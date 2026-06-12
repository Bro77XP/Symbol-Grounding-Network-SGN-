# Symbol Grounding Network (SGN)

An AI that learns language from the **visual shapes of alphabet letters** using a MobileNetV3-Large-style CNN fused with hand-crafted topological features.

## How It Works

1. **26 binary 7×7 grids** define each letter's canonical shape (A–Z)
2. **19 topological features** are extracted per grid (pure numpy): symmetries, centroids, density, holes, endpoints, branch points, peripheral pixels, pixel variance, etc.
3. A **MobileNetV3-Large CNN** (~2.05M parameters, 6 inverted-bottleneck blocks with SE + h-swish) learns from augmented versions (shift, noise, cutout)
4. CNN features (128-dim) + topological features (19-dim) are **fused** and fed to multi-task heads:
   - **Letter classifier**: 26-way classification
   - **BiGRU word encoder**: maps fused features to word embeddings
   - **TransformerDecoder sentence model**: 4 heads, 3 layers, 128-dim
5. A **CFG grammar** with 184-word vocabulary, POS tags, auxiliary verbs, and question rules enables sentence parsing and generation

## Commands

| Command | Description |
|---|---|
| `train [epochs] [variants]` | Train the neural network |
| `show <letter>` | Display grid + features for a letter |
| `read` | Enter letters one-by-one to build a word |
| `word <word>` | Spell a word using stored letter grids |
| `recognize <word>` | Full breakdown: grids, features, CNN predictions |
| `speak <sentence>` | Parse, auto-learn topics, and respond |
| `generate` | Generate a random sentence |
| `parse <sentence>` | Show parse tree |
| `learn <topic>` | Search DuckDuckGo/Wikipedia to learn a topic |
| `ask <question>` | Answer using learned knowledge |
| `topics` | List all learned topics |
| `forget <topic>` | Remove a learned topic |
| `vocab` | List all vocabulary words |
| `features` | Show topological feature names |
| `status` | Model and system info |
| `help` | Show this message |
| `quit` | Exit |

## Quick Start

```bash
python testidea.py
shapelang> train 200 300
shapelang> recognize CAT
shapelang> speak the cat ran
shapelang> learn cat
shapelang> ask what is a cat
```

## Files

| File | Purpose |
|---|---|
| `letter_grids.py` | 26 letter grids + augmentation (shift, noise, cutout) |
| `feature_extractor.py` | 19 topological feature extractors |
| `letter_model.py` | ShapeLangModel (CNN + fusion + heads), training loop |
| `language_engine.py` | CFG parser, vocabulary, sentence generation, conversation |
| `web_learner.py` | DuckDuckGo/Wikipedia search, knowledge base |
| `train.py` | Standalone training script (calls `letter_model.train_letter_model()`) |
| `testidea.py` | Interactive CLI |
| `letter_model.pt` | Trained model checkpoint |
