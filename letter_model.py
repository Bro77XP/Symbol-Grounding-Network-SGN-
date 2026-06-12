import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Tuple, Optional

MODEL_PATH = "letter_model.pt"
NUM_CLASSES = 26
LETTERS = sorted("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class HardSwish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * F.relu6(x + 3, inplace=True) / 6.0


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = F.relu(self.fc1(s))
        s = self.fc2(s).sigmoid().view(b, c, 1, 1)
        return x * s


class DepthwiseConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size, stride, padding, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu6(self.bn(self.conv(x)), inplace=True)


class PointwiseConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str = "h-swish"):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = HardSwish() if activation == "h-swish" else nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class InvertedBottleneck(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, expansion: int = 1, se_reduction: int = 4):
        super().__init__()
        mid_ch = in_ch * expansion
        self.expand = PointwiseConv2d(in_ch, mid_ch, "relu") if expansion > 1 else nn.Identity()
        self.depthwise = DepthwiseConv2d(mid_ch)
        self.se = SqueezeExcitation(mid_ch, se_reduction)
        self.project = PointwiseConv2d(mid_ch, out_ch, "h-swish")
        self.use_residual = (in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.se(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


class MobileNetV3Small7x7(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            HardSwish(),
        )
        self.block1 = InvertedBottleneck(32, 64, expansion=3, se_reduction=4)
        self.block2 = InvertedBottleneck(64, 96, expansion=3, se_reduction=4)
        self.block3 = InvertedBottleneck(96, 128, expansion=2, se_reduction=4)
        self.block4 = InvertedBottleneck(128, 160, expansion=2, se_reduction=4)
        self.block5 = InvertedBottleneck(160, 192, expansion=2, se_reduction=4)
        self.block6 = InvertedBottleneck(192, 192, expansion=2, se_reduction=4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(192, embedding_dim),
            HardSwish(),
            nn.Dropout(0.2),
        )
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.pool(x).flatten(1)
        x = self.fc(x)
        return x


class FeatureFusion(nn.Module):
    def __init__(self, cnn_dim: int = 128, feat_dim: int = 19, out_dim: int = 128):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(cnn_dim + feat_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, cnn_emb: torch.Tensor, feat_vec: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([cnn_emb, feat_vec], dim=1)
        return self.project(combined)


class LetterClassifier(nn.Module):
    def __init__(self, in_dim: int = 128, num_classes: int = 26):
        super().__init__()
        self.head = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class WordGRUEncoder(nn.Module):
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers=num_layers,
            batch_first=True, bidirectional=True, dropout=0.2 if num_layers > 1 else 0,
        )
        self.project = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        else:
            out, _ = self.gru(x)
        return self.project(out)


class SentenceTransformerDecoder(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 3,
                 dim_feedforward: int = 512, vocab_size: int = 130, max_seq_len: int = 32):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor,
                tgt_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = tgt.size(1)
        positions = torch.arange(seq_len, device=tgt.device).unsqueeze(0)
        tgt_emb = self.embedding(tgt) + self.pos_embedding(positions)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_proj(out)

    def generate(self, memory: torch.Tensor, max_len: int = 20, temperature: float = 0.8,
                 sos_id: int = 0, eos_id: int = 1) -> torch.Tensor:
        device = memory.device
        batch_size = memory.size(0)
        generated = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        for _ in range(max_len):
            logits = self.forward(generated, memory)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return generated


class ShapeLangModel(nn.Module):
    def __init__(self, embedding_dim: int = 128, word_hidden: int = 128,
                 sent_d_model: int = 128,                  vocab_size: int = 177):
        super().__init__()
        self.cnn = MobileNetV3Small7x7(embedding_dim)
        self.fusion = FeatureFusion(embedding_dim, 19, embedding_dim)
        self.letter_classifier = LetterClassifier(embedding_dim, NUM_CLASSES)
        self.word_encoder = WordGRUEncoder(embedding_dim, word_hidden)
        self.sentence_decoder = SentenceTransformerDecoder(
            d_model=sent_d_model, vocab_size=vocab_size,
        )
        self.embedding_dim = embedding_dim
        self.word_hidden = word_hidden
        self.register_buffer("feat_mean", torch.zeros(19))
        self.register_buffer("feat_std", torch.ones(19))

    def forward_letter(self, grids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        cnn_emb = self.cnn(grids)
        fused = self.fusion(cnn_emb, features)
        logits = self.letter_classifier(fused)
        return logits, fused

    def forward_word(self, letter_embeddings: torch.Tensor,
                     lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.word_encoder(letter_embeddings, lengths)

    def forward_sentence(self, tgt_tokens: torch.Tensor,
                         memory: torch.Tensor) -> torch.Tensor:
        return self.sentence_decoder(tgt_tokens, memory)

    def encode_grids(self, grids: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        cnn_emb = self.cnn(grids)
        fused = self.fusion(cnn_emb, features)
        return fused

    def recognize_letter(self, grids: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, _ = self.forward_letter(grids, features)
        probs = F.softmax(logits, dim=-1)
        confs, preds = probs.max(dim=-1)
        return preds, confs

    def generate_sentence(self, word_embeddings: torch.Tensor,
                          max_len: int = 20, temperature: float = 0.8) -> torch.Tensor:
        memory = self.word_encoder(word_embeddings)
        memory = memory.unsqueeze(0) if memory.dim() == 2 else memory
        return self.sentence_decoder.generate(memory, max_len, temperature)


def train_letter_model(variants_per_letter: int = 200, epochs: int = 300,
                       lr: float = 1e-3, batch_size: int = 64,
                       patience: int = 0, device: str = "cpu") -> ShapeLangModel:
    from letter_grids import generate_augmented_dataset
    from feature_extractor import extract_features_batch

    print("Generating augmented training data...")
    X_grids, y_labels = generate_augmented_dataset(variants_per_letter)
    X_feats = extract_features_batch(X_grids)

    n = len(y_labels)
    indices = np.random.permutation(n)
    split = int(n * 0.85)
    train_idx, val_idx = indices[:split], indices[split:]

    feat_mean = X_feats[train_idx].mean(axis=0)
    feat_std = X_feats[train_idx].std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0
    X_feats = (X_feats - feat_mean) / feat_std

    X_grids_t = torch.tensor(X_grids, dtype=torch.float32).to(device)
    if X_grids_t.dim() == 3:
        X_grids_t = X_grids_t.unsqueeze(1)
    X_feats_t = torch.tensor(X_feats, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_labels, dtype=torch.long).to(device)

    model = ShapeLangModel().to(device)
    model.feat_mean.data = torch.tensor(feat_mean, device=device)
    model.feat_std.data = torch.tensor(feat_std, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    print(f"Training on {len(train_idx)} samples, validating on {len(val_idx)} samples")
    print(f"Device: {device}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        perm_train = torch.randperm(len(train_idx))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_idx), batch_size):
            batch_idx = train_idx[perm_train[i:i + batch_size]]
            grids = X_grids_t[batch_idx]
            feats = X_feats_t[batch_idx]
            labels = y_t[batch_idx]

            logits, _ = model.forward_letter(grids, feats)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_grids = X_grids_t[val_idx]
            val_feats = X_feats_t[val_idx]
            val_labels = y_t[val_idx]
            preds, _ = model.recognize_letter(val_grids, val_feats)
            val_acc = (preds == val_labels).float().mean().item()

        if epoch % 10 == 0 or epoch == 1:
            lr_now = scheduler.get_last_lr()[0]
            print(f"  Epoch {epoch:3d}/{epochs} - loss: {avg_loss:.4f} - val_acc: {val_acc:.1%} - lr: {lr_now:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif patience > 0:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH} (best val_acc: {best_val_acc:.1%})")
    return model


def load_letter_model(device: str = "cpu") -> ShapeLangModel:
    model = ShapeLangModel(vocab_size=177)
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        model_state = model.state_dict()
        for k in list(state.keys()):
            if k in model_state and state[k].shape == model_state[k].shape:
                model_state[k] = state[k]
        model.load_state_dict(model_state)
        model.eval()
        loaded = sum(1 for k in state if k in model_state and state[k].shape == model_state[k].shape)
        skipped = len(state) - loaded
        print(f"Loaded model from {MODEL_PATH} ({loaded} layers)")
        if skipped:
            print(f"  (Reinitialized {skipped} layers due to shape changes)")
    else:
        print(f"No model found at {MODEL_PATH}. Run 'train' first.")
    return model.to(device)


def recognize_letter_from_grid(grid: np.ndarray, model: ShapeLangModel,
                               device: str = "cpu") -> Tuple[str, float]:
    from feature_extractor import extract_features

    model.eval()
    features = extract_features(grid)
    features = (features - model.feat_mean.cpu().numpy()) / model.feat_std.cpu().numpy()
    grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    feat_t = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        preds, confs = model.recognize_letter(grid_t, feat_t)

    idx = preds.item()
    conf = confs.item()
    return LETTERS[idx], conf


def recognize_word_from_grids(grids: list, model: ShapeLangModel,
                              device: str = "cpu") -> str:
    word = ""
    for grid in grids:
        letter, _ = recognize_letter_from_grid(grid, model, device)
        word += letter
    return word


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    model = train_letter_model(epochs=50, device=device)

    from letter_grids import get_grid
    for letter in ["A", "B", "C", "D", "E"]:
        grid = get_grid(letter)
        pred, conf = recognize_letter_from_grid(grid, model, device)
        print(f"  {letter} -> {pred} (confidence: {conf:.1%})")
