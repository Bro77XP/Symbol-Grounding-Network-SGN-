import numpy as np
from typing import Dict, Tuple

LETTER_GRIDS: Dict[str, np.ndarray] = {
    "A": np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float32),
    "B": np.array([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0],
    ], dtype=np.float32),
    "C": np.array([
        [0, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 1, 1, 0],
    ], dtype=np.float32),
    "D": np.array([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0],
    ], dtype=np.float32),
    "E": np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.float32),
    "F": np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32),
    "G": np.array([
        [0, 0, 1, 1, 1, 1, 0],
        [0, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ], dtype=np.float32),
    "H": np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float32),
    "I": np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.float32),
    "J": np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ], dtype=np.float32),
    "K": np.array([
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
    ], dtype=np.float32),
    "L": np.array([
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.float32),
    "M": np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float32),
    "N": np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float32),
    "O": np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
    ], dtype=np.float32),
    "P": np.array([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
    ], dtype=np.float32),
    "Q": np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
    ], dtype=np.float32),
    "R": np.array([
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 1, 0],
        [1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 0],
    ], dtype=np.float32),
    "S": np.array([
        [0, 1, 1, 1, 1, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ], dtype=np.float32),
    "T": np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.float32),
    "U": np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 1, 0],
    ], dtype=np.float32),
    "V": np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.float32),
    "W": np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float32),
    "X": np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 0, 0, 0, 0, 0, 1],
    ], dtype=np.float32),
    "Y": np.array([
        [1, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=np.float32),
    "Z": np.array([
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
    ], dtype=np.float32),
}


def get_all_grids() -> Dict[str, np.ndarray]:
    return {k: v.copy() for k, v in LETTER_GRIDS.items()}


def get_grid(letter: str) -> np.ndarray:
    letter = letter.upper()
    if letter not in LETTER_GRIDS:
        raise ValueError(f"Unknown letter: {letter}")
    return LETTER_GRIDS[letter].copy()


def display_grid(grid: np.ndarray) -> str:
    lines = []
    for row in grid:
        line = ""
        for val in row:
            line += "██" if val > 0.5 else "  "
        lines.append(line)
    return "\n".join(lines)


def grid_to_text(grid: np.ndarray) -> str:
    lines = []
    for row in grid:
        line = ""
        for val in row:
            line += "1" if val > 0.5 else "0"
        lines.append(line)
    return "\n".join(lines)


def text_to_grid(text: str) -> np.ndarray:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    grid = np.zeros((7, 7), dtype=np.float32)
    for i, line in enumerate(lines[:7]):
        for j, ch in enumerate(line[:7]):
            if ch in ("1", "█", "#"):
                grid[i, j] = 1.0
    return grid


def add_noise(grid: np.ndarray, level: float = 0.1) -> np.ndarray:
    noisy = grid.copy()
    mask = np.random.random(noisy.shape) < level
    noisy[mask] = 1.0 - noisy[mask]
    return noisy


def random_rotation(grid: np.ndarray) -> np.ndarray:
    k = np.random.choice([0, 1, 2, 3])
    return np.rot90(grid, k=k).copy()


def random_flip(grid: np.ndarray) -> np.ndarray:
    if np.random.random() < 0.5:
        grid = np.flipud(grid)
    if np.random.random() < 0.5:
        grid = np.fliplr(grid)
    return grid.copy()


def random_shift(grid: np.ndarray, max_shift: int = 1) -> np.ndarray:
    shifted = np.zeros_like(grid)
    dx = np.random.randint(-max_shift, max_shift + 1)
    dy = np.random.randint(-max_shift, max_shift + 1)
    h, w = grid.shape
    src_y1 = max(0, dy)
    src_y2 = min(h, h + dy)
    src_x1 = max(0, dx)
    src_x2 = min(w, w + dx)
    dst_y1 = max(0, -dy)
    dst_y2 = min(h, h - dy)
    dst_x1 = max(0, -dx)
    dst_x2 = min(w, w - dx)
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = grid[src_y1:src_y2, src_x1:src_x2]
    return shifted


def mixup(grid_a: np.ndarray, grid_b: np.ndarray, alpha: float = 0.8) -> np.ndarray:
    lam = np.random.beta(alpha, alpha)
    return (lam * grid_a + (1 - lam) * grid_b).clip(0, 1)


def cutout(grid: np.ndarray, size: int = 2) -> np.ndarray:
    g = grid.copy()
    h, w = g.shape
    cy = np.random.randint(0, h)
    cx = np.random.randint(0, w)
    y1 = max(0, cy - size // 2)
    y2 = min(h, cy + size // 2 + 1)
    x1 = max(0, cx - size // 2)
    x2 = min(w, cx + size // 2 + 1)
    g[y1:y2, x1:x2] = 0.0
    return g


def augment_grid(grid: np.ndarray) -> np.ndarray:
    g = grid.copy()
    if np.random.random() < 0.3:
        g = random_shift(g, max_shift=1)
    if np.random.random() < 0.15:
        g = add_noise(g, level=0.04)
    if np.random.random() < 0.15:
        g = cutout(g, size=2)
    return g


def generate_augmented_dataset(
    variants_per_letter: int = 200,
) -> Tuple[list, list]:
    grids = get_all_grids()
    X, y = [], []
    letters = sorted(grids.keys())
    for letter in letters:
        base = grids[letter]
        # multiple clean copies
        for _ in range(max(1, variants_per_letter // 4)):
            X.append(base.reshape(1, 7, 7))
            y.append(letters.index(letter))
        # augmented variants
        for _ in range(variants_per_letter - max(1, variants_per_letter // 4)):
            aug = augment_grid(base)
            X.append(aug.reshape(1, 7, 7))
            y.append(letters.index(letter))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


if __name__ == "__main__":
    grids = get_all_grids()
    print(f"Loaded {len(grids)} letter grids (7x7)\n")
    for letter in sorted(grids.keys()):
        print(f"  {letter}:")
        print(display_grid(grids[letter]))
        print()
