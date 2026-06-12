import numpy as np
from typing import Dict


def _count_neighbors(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    padded = np.pad(grid, 1, mode="constant", constant_values=0)
    neighbor_count = np.zeros_like(grid, dtype=np.float32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            neighbor_count += padded[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
    return neighbor_count


def endpoint_count(grid: np.ndarray) -> float:
    binary = (grid > 0.5).astype(np.float32)
    nc = _count_neighbors(binary)
    return float(np.sum((binary > 0) & (nc == 1)))


def intersection_count(grid: np.ndarray) -> float:
    binary = (grid > 0.5).astype(np.float32)
    nc = _count_neighbors(binary)
    return float(np.sum((binary > 0) & (nc >= 3)))


def passthrough_count(grid: np.ndarray) -> float:
    binary = (grid > 0.5).astype(np.float32)
    nc = _count_neighbors(binary)
    return float(np.sum((binary > 0) & (nc == 2)))


def flood_fill(grid: np.ndarray, start_y: int, start_x: int, visited: np.ndarray) -> int:
    h, w = grid.shape
    stack = [(start_y, start_x)]
    pixels = []
    while stack:
        y, x = stack.pop()
        if y < 0 or y >= h or x < 0 or x >= w:
            continue
        if visited[y, x]:
            continue
        if grid[y, x] <= 0.5:
            continue
        visited[y, x] = True
        pixels.append((y, x))
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                stack.append((y + dy, x + dx))
    return len(pixels)


def connected_components(grid: np.ndarray) -> float:
    binary = (grid > 0.5).astype(np.float32)
    visited = np.zeros_like(binary, dtype=bool)
    count = 0
    h, w = binary.shape
    for y in range(h):
        for x in range(w):
            if binary[y, x] > 0.5 and not visited[y, x]:
                flood_fill(binary, y, x, visited)
                count += 1
    return float(count)


def euler_number(grid: np.ndarray) -> float:
    binary = (grid > 0.5).astype(np.int32)
    h, w = binary.shape
    p = np.zeros(16, dtype=np.float32)
    for y in range(h - 1):
        for x in range(w - 1):
            val = (
                binary[y, x] * 8
                + binary[y, x + 1] * 4
                + binary[y + 1, x + 1] * 2
                + binary[y + 1, x] * 1
            )
            p[val] += 1
    euler = (
        (p[1] + p[3] + p[5] + p[7] + p[9] + p[11] + p[13] + p[15])
        - (p[2] + p[6] + p[10] + p[14])
        + (p[4] + p[12])
        - (p[8])
    )
    return float(euler)


def hole_count(grid: np.ndarray) -> float:
    cc = connected_components(grid)
    euler = euler_number(grid)
    return max(0.0, cc - euler)


def horizontal_symmetry(grid: np.ndarray) -> float:
    flipped = np.flipud(grid)
    total = grid.size
    matches = np.sum(grid == flipped)
    return float(matches / total) if total > 0 else 0.0


def vertical_symmetry(grid: np.ndarray) -> float:
    flipped = np.fliplr(grid)
    total = grid.size
    matches = np.sum(grid == flipped)
    return float(matches / total) if total > 0 else 0.0


def rotational_symmetry(grid: np.ndarray) -> float:
    flipped = np.fliplr(np.flipud(grid))
    total = grid.size
    matches = np.sum(grid == flipped)
    return float(matches / total) if total > 0 else 0.0


def pixel_density(grid: np.ndarray) -> float:
    return float(np.mean(grid > 0.5))


def bounding_box_width(grid: np.ndarray) -> float:
    binary = grid > 0.5
    cols = np.any(binary, axis=0)
    if not np.any(cols):
        return 0.0
    indices = np.where(cols)[0]
    return float(indices[-1] - indices[0] + 1)


def bounding_box_height(grid: np.ndarray) -> float:
    binary = grid > 0.5
    rows = np.any(binary, axis=1)
    if not np.any(rows):
        return 0.0
    indices = np.where(rows)[0]
    return float(indices[-1] - indices[0] + 1)


def aspect_ratio(grid: np.ndarray) -> float:
    w = bounding_box_width(grid)
    h = bounding_box_height(grid)
    return w / h if h > 0 else 0.0


def center_of_mass_x(grid: np.ndarray) -> float:
    binary = grid > 0.5
    if not np.any(binary):
        return 0.5
    rows, cols = np.where(binary)
    return float(np.mean(cols) / (grid.shape[1] - 1)) if grid.shape[1] > 1 else 0.5


def center_of_mass_y(grid: np.ndarray) -> float:
    binary = grid > 0.5
    if not np.any(binary):
        return 0.5
    rows, cols = np.where(binary)
    return float(np.mean(rows) / (grid.shape[0] - 1)) if grid.shape[0] > 1 else 0.5


def peripheral_pixels(grid: np.ndarray) -> float:
    """Count of on-pixels touching the grid border."""
    binary = (grid > 0.5).astype(np.float32)
    top = binary[0, :].sum()
    bottom = binary[-1, :].sum()
    left = binary[:, 0].sum()
    right = binary[:, -1].sum()
    return float(top + bottom + left + right)


def centroid_distance(grid: np.ndarray) -> float:
    """Distance from center of mass to grid center (normalized)."""
    binary = grid > 0.5
    if not np.any(binary):
        return 0.0
    rows, cols = np.where(binary)
    cx = np.mean(cols) / (grid.shape[1] - 1)
    cy = np.mean(rows) / (grid.shape[0] - 1)
    return float(np.sqrt((cx - 0.5)**2 + (cy - 0.5)**2) / 0.7071)


def diagonal_symmetry(grid: np.ndarray) -> float:
    """Symmetry score along main diagonal (top-left to bottom-right)."""
    flipped = np.rot90(grid)
    total = grid.size
    matches = np.sum(grid == flipped)
    return float(matches / total) if total > 0 else 0.0


def pixel_variance(grid: np.ndarray) -> float:
    """Variance of pixel intensities (measures spread)."""
    binary = (grid > 0.5).astype(np.float32)
    return float(np.var(binary))


FEATURE_NAMES = [
    "endpoints",
    "intersections",
    "passthroughs",
    "connected_components",
    "holes",
    "euler_number",
    "horizontal_symmetry",
    "vertical_symmetry",
    "rotational_symmetry",
    "pixel_density",
    "bbox_width",
    "bbox_height",
    "aspect_ratio",
    "com_x",
    "com_y",
    "peripheral_pixels",
    "centroid_distance",
    "diagonal_symmetry",
    "pixel_variance",
]


def extract_features(grid: np.ndarray) -> np.ndarray:
    features = np.array([
        endpoint_count(grid),
        intersection_count(grid),
        passthrough_count(grid),
        connected_components(grid),
        hole_count(grid),
        euler_number(grid),
        horizontal_symmetry(grid),
        vertical_symmetry(grid),
        rotational_symmetry(grid),
        pixel_density(grid),
        bounding_box_width(grid) / 7.0,
        bounding_box_height(grid) / 7.0,
        aspect_ratio(grid),
        center_of_mass_x(grid),
        center_of_mass_y(grid),
        peripheral_pixels(grid) / 28.0,
        centroid_distance(grid),
        diagonal_symmetry(grid),
        pixel_variance(grid),
    ], dtype=np.float32)
    return features


def extract_features_batch(grids: np.ndarray) -> np.ndarray:
    batch_size = grids.shape[0]
    features = np.zeros((batch_size, 19), dtype=np.float32)
    for i in range(batch_size):
        g = grids[i]
        if g.ndim == 3:
            g = g[0]
        features[i] = extract_features(g)
    return features


if __name__ == "__main__":
    from letter_grids import get_all_grids, display_grid

    grids = get_all_grids()
    print("Feature extraction for all letters:\n")
    for letter in sorted(grids.keys()):
        grid = grids[letter]
        feats = extract_features(grid)
        print(f"  {letter}:")
        print(display_grid(grid))
        for name, val in zip(FEATURE_NAMES, feats):
            print(f"    {name:25s}: {val:.4f}")
        print()
