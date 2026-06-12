import sys
import torch
import numpy as np
from letter_grids import get_all_grids, display_grid
from feature_extractor import extract_features, FEATURE_NAMES
from letter_model import train_letter_model, recognize_letter_from_grid, LETTERS

BANNER = """
  ____  _          _ _                 _
 / ___|| |__   ___| | | __ _ _ __   __| | ___  _ __
 \\___ \\| '_ \\ / _ \\ | |/ _` | '_ \\ / _` |/ _ \\| '__|
  ___) | | | |  __/ | | (_| | | | | (_| | (_) | |
 |____/|_| |_|\\___|_|_|\\__,_|_| |_|\\__,_|\\___/|_|
  Training Module
"""

def main():
    print(BANNER)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    variants = 200
    epochs = 200
    if len(sys.argv) > 1:
        try:
            epochs = int(sys.argv[1])
        except ValueError:
            pass
    if len(sys.argv) > 2:
        try:
            variants = int(sys.argv[2])
        except ValueError:
            pass

    print(f"Variants per letter: {variants}")
    print(f"Epochs: {epochs}")
    print()

    print("=" * 60)
    print("Architecture: MobileNetV3-Small + SE + Feature Fusion + Multi-Task")
    print("=" * 60)
    print()

    model = train_letter_model(
        variants_per_letter=variants,
        epochs=epochs,
        device=device,
    )

    print("\nStep 5: Testing on clean letter grids...")
    model.eval()
    grids = get_all_grids()
    correct = 0
    total = 0
    for letter in sorted(grids.keys()):
        grid = grids[letter]
        pred, conf = recognize_letter_from_grid(grid, model, device)
        is_correct = pred == letter
        correct += int(is_correct)
        total += 1
        status = "OK" if is_correct else "FAIL"
        print(f"    {letter} -> {pred} (conf: {conf:.1%}) [{status}]")

    print(f"\n  Clean grid accuracy: {correct}/{total} ({correct/total:.1%})")
    print("\nDone! Run 'python testidea.py' to use the interactive system.")


if __name__ == "__main__":
    main()
