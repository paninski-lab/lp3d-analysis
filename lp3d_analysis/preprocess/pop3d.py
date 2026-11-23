"""Helpers for converting the 3D-POP multi-animal, multi-view pigeon dataset.

Raw layout (per sequence, under ``<raw>/Pigeon{02,05}/Sequence<N>_n<NN>_<date>/``):
    Videos/<seq>-Cam{1..4}.mp4
    Annotation/<seq>-Cam{1..4}-{BBox,Keypoint2DFiltered,Keypoint3D}.csv
    CalibrationInfo/<seq>-Cam{1..4}-{Extrinsics,Intrinsics}.p   (OpenCV; world units are mm)

The ``Annotation`` files (not ``TrainingSplit``) are used; there the ``frame`` column is the
row index and equals the video frame index exactly.

Notes on the data:
    - 3D keypoints are in a shared world frame (mm), z is up. The four per-camera Keypoint3D files
      can differ by about one frame of motion (cams {1,4} vs {2,3}); frames where they disagree
      are dropped because the 2D labels derive from each camera's own 3D file.
    - Per-bird columns are ``<bird>_{bp,hd}_<kp>_{x,y[,z]}`` and ``<bird>_BBox_{x,y,w,h}``.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

KEYPOINTS = [
    'hd_beak',
    'hd_nose',
    'hd_leftEye',
    'hd_rightEye',
    'bp_leftShoulder',
    'bp_rightShoulder',
    'bp_topKeel',
    'bp_bottomKeel',
    'bp_tail',
]
KP_IDX = {k: i for i, k in enumerate(KEYPOINTS)}
CAMS = [1, 2, 3, 4]
MAX_3D_DISCREPANCY_MM = 3.0


def n_birds(seq: str) -> int:
    """Number of birds, parsed from a sequence name like ``Sequence13_n02_13072022``."""
    return int(re.search(r'_n(\d+)_', seq).group(1))


def seq_dir(raw_dir: Path, seq: str) -> Path:
    return Path(raw_dir) / f'Pigeon{n_birds(seq):02d}' / seq


def list_sequences(raw_dir: Path, exclude=()) -> list:
    """Sequence names for the 2- and 5-bird sessions, sorted by name."""
    out = []
    for pigeon_dir in sorted(Path(raw_dir).glob('Pigeon0[25]')):
        out += [p.name for p in sorted(pigeon_dir.glob('Sequence*')) if p.is_dir()]
    return [s for s in out if s not in set(exclude)]


def _bird_ids(columns) -> list:
    return sorted(c[: -len('_bp1_x')] for c in columns if c.endswith('_bp1_x'))


def load_sequence(raw_dir: Path, seq: str) -> dict:
    """Load annotations for one sequence.

    Returns
    -------
    dict with
        birds: list of bird ids (fixed order; used as the instance order everywhere)
        kp3d: (4, T, B, 9, 3) world mm, one slice per camera file
        kp2d: (4, T, B, 9, 2) filtered 2D keypoints, pixels
        bbox: (4, T, B, 4) x, y, w, h in pixels
    """
    stem = seq_dir(raw_dir, seq) / 'Annotation' / seq
    k3 = [pd.read_csv(f'{stem}-Cam{c}-Keypoint3D.csv') for c in CAMS]
    k2 = [pd.read_csv(f'{stem}-Cam{c}-Keypoint2DFiltered.csv') for c in CAMS]
    bb = [pd.read_csv(f'{stem}-Cam{c}-BBox.csv') for c in CAMS]
    n = len(k3[0])
    for d in k3 + k2 + bb:
        assert len(d) == n and (d['frame'].values == np.arange(n)).all(), seq
    birds = _bird_ids(k3[0].columns)
    assert birds == _bird_ids(k2[0].columns)
    assert birds == sorted(c[: -len('_BBox_x')] for c in bb[0].columns if c.endswith('_BBox_x'))

    def stack(dfs, parts, coords):
        return np.stack(
            [
                np.stack(
                    [
                        np.stack([d[[f'{b}_{k}_{a}' for a in coords]].values for k in parts], 1)
                        for b in birds
                    ],
                    1,
                )
                for d in dfs
            ]
        )

    kp3d = stack(k3, KEYPOINTS, 'xyz')
    kp2d = stack(k2, KEYPOINTS, 'xy')
    bbox = np.stack(
        [np.stack([d[[f'{b}_BBox_{a}' for a in 'xywh']].values for b in birds], 1) for d in bb]
    )
    return dict(birds=birds, kp3d=kp3d, kp2d=kp2d, bbox=bbox)


def candidate_mask(data: dict, max_3d_mm: float = MAX_3D_DISCREPANCY_MM) -> np.ndarray:
    """Boolean (T,) mask of time points where *all* birds are fully usable.

    Requires, for every bird: all 9 filtered 2D keypoints present in all 4 views, a bbox with
    positive size in all 4 views, a complete 3D pose in all 4 files, and the 4 files' 3D
    keypoints agreeing to within ``max_3d_mm`` (Euclidean, worst keypoint).
    """
    kp2d, kp3d, bbox = data['kp2d'], data['kp3d'], data['bbox']
    ok2d = ~np.isnan(kp2d).any(axis=(0, 2, 3, 4))
    okbb = (~np.isnan(bbox).any(axis=(0, 2, 3))) & (bbox[..., 2:] > 0).all(axis=(0, 2, 3))
    ok3d = ~np.isnan(kp3d).any(axis=(0, 2, 3, 4))
    disc = np.zeros(kp3d.shape[1])
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.linalg.norm(kp3d[i] - kp3d[j], axis=-1).reshape(kp3d.shape[1], -1)
            disc = np.fmax(disc, np.nanmax(np.where(np.isnan(d), 0, d), axis=1))
    return ok2d & okbb & ok3d & (disc <= max_3d_mm)


def eligible_mask(
    data: dict, radius: int = 1, max_3d_mm: float = MAX_3D_DISCREPANCY_MM
) -> np.ndarray:
    """``candidate_mask`` AND the same mask at t-radius..t+radius (and not at the sequence ends).

    Neighbouring frames failing the per-frame filter (typically NaN filtered 2D labels) is a
    marker of fast motion/motion blur, where head keypoints are least reliable.
    """
    m = candidate_mask(data, max_3d_mm)
    out = m.copy()
    for k in range(1, radius + 1):
        out[k:] &= m[:-k]  # t-k passes
        out[:-k] &= m[k:]  # t+k passes
        out[:k] = False
        out[-k:] = False
    return out


def ego_pose(pts: np.ndarray):
    """Ego-centric pose descriptor.

    Parameters
    ----------
    pts: (..., 9, 3) world coordinates in mm, z up

    Returns
    -------
    ego: (..., 9, 3) keypoints centered on the bird's centroid, rotated about the vertical axis so
        the heading (tail -> shoulder midpoint, projected to the ground plane) is +x, and divided
        by body length
    length: (...,) body length in mm
    """
    q = pts - pts.mean(-2, keepdims=True)
    shoulders = (
        pts[..., KP_IDX['bp_leftShoulder'], :] + pts[..., KP_IDX['bp_rightShoulder'], :]
    ) / 2
    heading = shoulders - pts[..., KP_IDX['bp_tail'], :]
    length = np.linalg.norm(heading, axis=-1)
    theta = np.arctan2(heading[..., 1], heading[..., 0])
    c, s = np.cos(-theta)[..., None], np.sin(-theta)[..., None]
    x = q[..., 0] * c - q[..., 1] * s
    y = q[..., 0] * s + q[..., 1] * c
    return np.stack([x, y, q[..., 2]], -1) / length[..., None, None], length


def separation(pts: np.ndarray) -> float:
    """Inter-bird separation for one time point; pts is (B, 9, 3).

    2 birds: centroid distance. 5 birds: mean nearest-neighbour centroid distance.
    """
    cen = pts.mean(-2)
    d = np.linalg.norm(cen[:, None] - cen[None], axis=-1) + np.eye(len(cen)) * 1e12
    return float(d.min(1).mean())


def kmeans(x: np.ndarray, k: int, seed: int = 0, n_iter: int = 50):
    """Plain k-means (k-means++ init). Returns (labels, centers)."""
    rng = np.random.default_rng(seed)
    centers = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d2 = np.min([((x - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(n_iter):
        labels = np.argmin(((x[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        centers = np.array(
            [x[labels == j].mean(0) if (labels == j).any() else centers[j] for j in range(k)]
        )
    return labels, centers


def pca_project(x: np.ndarray, n_components: int = 10):
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:n_components].T


def select_time_points(
    seq_of: np.ndarray,
    frame_of: np.ndarray,
    separation_of: np.ndarray,
    cluster_counts: np.ndarray,
    n_select: int,
    n_bins: int = 5,
    min_gap: int = 60,
    max_per_seq_factor: float = 1.5,
    seed: int = 0,
) -> np.ndarray:
    """Greedy selection of whole time points, stratified by separation, covering pose clusters.

    Parameters
    ----------
    seq_of, frame_of: (T,) candidate time points
    separation_of: (T,) inter-bird separation metric (stratification axis)
    cluster_counts: (T, K) number of instances of each pose cluster at each time point
    n_select: number of time points to pick
    n_bins: quantile bins of separation; each gets ``n_select / n_bins`` picks
    min_gap: minimum frames between two picks in the same sequence
    max_per_seq_factor: per-sequence cap = ceil(factor * n_select / n_sequences)
    seed: tie-breaking noise

    Returns
    -------
    indices into the candidate arrays, in pick order
    """
    rng = np.random.default_rng(seed)
    n_seq = len(set(seq_of))
    cap = int(np.ceil(max_per_seq_factor * n_select / n_seq))
    edges = np.quantile(separation_of, np.linspace(0, 1, n_bins + 1))
    bin_of = np.clip(np.searchsorted(edges, separation_of, side='right') - 1, 0, n_bins - 1)
    quota = np.full(n_bins, n_select // n_bins)
    quota[: n_select - quota.sum()] += 1

    freq = cluster_counts.sum(0).astype(float)
    target = np.sqrt(freq)
    target = target / target.sum() * n_select * cluster_counts.sum(1).mean()
    current = np.zeros_like(target)
    picked, per_seq = [], {s: [] for s in set(seq_of)}
    b = 0
    while len(picked) < n_select:
        while quota[b % n_bins] == 0:
            b += 1
        bin_id = b % n_bins
        w = (target - current) / np.maximum(target, 1e-9)
        score = cluster_counts @ w + rng.uniform(0, 1e-6, len(seq_of))
        allowed = bin_of == bin_id
        for s, fr in per_seq.items():
            m = seq_of == s
            if len(fr) >= cap:
                allowed &= ~m
            elif fr:
                allowed &= ~(m & (np.abs(frame_of[:, None] - np.array(fr)[None]) < min_gap).any(1))
        score[~allowed] = -np.inf
        i = int(np.argmax(score))
        if not np.isfinite(score[i]):
            raise RuntimeError(f'no feasible candidate left in bin {bin_id}')
        picked.append(i)
        per_seq[seq_of[i]].append(int(frame_of[i]))
        current += cluster_counts[i]
        quota[bin_id] -= 1
        b += 1
    return np.array(picked)


def load_calibration(raw_dir: Path, seq: str) -> list:
    """Per-camera OpenCV calibration: list of dicts with rvec (3,), tvec (3,), K (3,3), dist (5,).

    World units are mm; rvec/tvec map world -> camera (``cv2.projectPoints`` convention).
    """
    import pickle

    out = []
    for c in CAMS:
        base = seq_dir(raw_dir, seq) / 'CalibrationInfo' / f'{seq}-Cam{c}'
        with open(f'{base}-Extrinsics.p', 'rb') as f:
            rvec, tvec = pickle.load(f)
        with open(f'{base}-Intrinsics.p', 'rb') as f:
            k_mat, dist = pickle.load(f)
        out.append(
            dict(
                rvec=np.asarray(rvec, float).ravel(),
                tvec=np.asarray(tvec, float).ravel(),
                K=np.asarray(k_mat, float),
                dist=np.asarray(dist, float).ravel(),
            )
        )
    return out


def triangulate(points2d: np.ndarray, calib: list) -> np.ndarray:
    """Linear (DLT) triangulation. points2d: (4, N, 2) pixels (NaN = missing) -> (N, 3) world mm."""
    import cv2

    rows = []
    for c, cal in enumerate(calib):
        norm = cv2.undistortPoints(
            np.nan_to_num(points2d[c]).reshape(-1, 1, 2), cal['K'], cal['dist']
        ).reshape(-1, 2)
        rot, _ = cv2.Rodrigues(cal['rvec'])
        proj = np.concatenate([rot, cal['tvec'][:, None]], 1)
        valid = ~np.isnan(points2d[c]).any(-1)
        r0 = (norm[:, 0:1] * proj[2] - proj[0]) * valid[:, None]
        r1 = (norm[:, 1:2] * proj[2] - proj[1]) * valid[:, None]
        rows += [r0, r1]
    a = np.stack(rows, 1)  # (N, 8, 4)
    _, _, vt = np.linalg.svd(a)
    x = vt[:, -1]
    return x[:, :3] / x[:, 3:4]


def reproject(points3d: np.ndarray, calib: list) -> np.ndarray:
    """(N, 3) world mm -> (4, N, 2) pixels."""
    import cv2

    return np.stack(
        [
            cv2.projectPoints(
                points3d.reshape(-1, 1, 3), cal['rvec'], cal['tvec'], cal['K'], cal['dist']
            )[0].reshape(-1, 2)
            for cal in calib
        ]
    )


def reprojection_error(kp2d: np.ndarray, calib: list) -> np.ndarray:
    """Triangulate-then-reproject error in pixels.

    kp2d: (4, T, B, 9, 2) -> (4, T, B, 9) euclidean pixel error per view/keypoint.
    """
    _, t, b, k, _ = kp2d.shape
    flat = kp2d.reshape(4, -1, 2)
    proj = reproject(triangulate(flat, calib), calib)
    return np.linalg.norm(proj - flat, axis=-1).reshape(4, t, b, k)


BBOX_SCALE = 2.0  # multiplier on the larger keypoint-extent dimension


def keypoint_bbox(xy: np.ndarray, scale: float = BBOX_SCALE) -> tuple:
    """Square bbox (x, y, h, w) in original pixels for one bird in one view.

    Centered on the midpoint of the tight extent of ``xy`` (n_keypoints, 2), with side ``scale``
    times the larger of that extent's width/height (same recipe as scripts/preprocess/r24m-crop.py).
    """
    x_min, y_min = np.nanmin(xy, axis=0)
    x_max, y_max = np.nanmax(xy, axis=0)
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = max(x_max - x_min, y_max - y_min) * scale
    return (
        int(round(cx - side / 2)),
        int(round(cy - side / 2)),
        int(round(side)),
        int(round(side)),
    )


def crop_array(frame: np.ndarray, x: int, y: int, h: int, w: int) -> np.ndarray:
    """Crop ``frame`` to the box (x, y, x + w, y + h), padding out-of-bounds regions with black."""
    height, width = frame.shape[:2]
    top, left = max(0, -y), max(0, -x)
    bottom, right = max(0, y + h - height), max(0, x + w - width)
    padded = np.zeros((height + top + bottom, width + left + right, frame.shape[2]), frame.dtype)
    padded[top : top + height, left : left + width] = frame
    return padded[y + top : y + top + h, x + left : x + left + w]
