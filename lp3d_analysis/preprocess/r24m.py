"""Helper functions for preprocessing the PAIR-R24M dataset."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import toml
from scipy.spatial.transform import Rotation
from sklearn.cluster import KMeans


def load_camera_calibration(calib_dir: Path, cam_name: str, video_size: tuple) -> dict:
    """Convert one camera's raw calibration json into an anipose/aniposelib toml camera entry.

    README.txt warns that the raw json's intrinsicMatrix uses Matlab's row-vector convention,
    which is the transpose of OpenCV's camera matrix. The same row-vector/column-vector
    distinction applies to rotationMatrix (standard for Matlab's Camera Calibrator extrinsics,
    and not called out separately in the README): if X_cam = X_world @ R_matlab + t (row
    vectors), then in OpenCV's column-vector convention X_cam = R_matlab.T @ X_world + t, i.e.
    rvec_opencv = rotvec(R_matlab.T) = -rotvec(R_matlab). This mirrors how rat7m's own
    Matlab-derived camera parameters are handled in
    lp3d_analysis.preprocess.bundle_adjust.convertParams.
    """
    cam_idx = cam_name.replace('Camera', '')
    json_path = calib_dir / f'camera{cam_idx}_calibration.json'
    with open(json_path) as f:
        raw = json.load(f)

    K = np.array(raw['intrinsicMatrix']).T
    rvec = -Rotation.from_matrix(np.array(raw['rotationMatrix'])).as_rotvec()
    tvec = np.array(raw['translationVector'])
    k1, k2, k3 = raw['radialDistortion']
    p1, p2 = raw['tangentialDistortion']

    return {
        'name': cam_name,
        'size': list(video_size),
        'matrix': K.tolist(),
        'distortions': [k1, k2, p1, p2, k3],
        'rotation': rvec.tolist(),
        'translation': tvec.tolist(),
    }


def write_session_calibration_toml(
    session_dir: Path,
    cam_names: list,
    video_size: tuple,
    save_path: Path,
) -> None:
    """Convert all cameras' raw calibration jsons for a session into a single anipose toml."""
    calib_dir = session_dir / 'calibration'
    data = {
        f'cam_{i}': load_camera_calibration(calib_dir, cam_name, video_size)
        for i, cam_name in enumerate(cam_names)
    }
    data['metadata'] = {}
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        toml.dump(data, f)


def project_points(points_3d: np.ndarray, calib: dict) -> np.ndarray:
    """Project (n, 3) world points into one camera's image plane.

    `calib` is one camera's entry from an anipose toml (as loaded by `toml.load`), i.e. a dict
    with 'rotation' (Rodrigues vector), 'translation', 'matrix' (3x3 intrinsics), and
    'distortions' (5 OpenCV coefficients: k1, k2, p1, p2, k3).
    """
    rvec = np.array(calib['rotation'], dtype=np.float64)
    tvec = np.array(calib['translation'], dtype=np.float64)
    K = np.array(calib['matrix'], dtype=np.float64)
    dist = np.array(calib['distortions'], dtype=np.float64)
    points_2d, _ = cv2.projectPoints(points_3d.astype(np.float64), rvec, tvec, K, dist)
    return points_2d.reshape(-1, 2)


def get_frame_from_chunks(video_dir: Path, frame_idx: int, chunk_size: int = 3500) -> np.ndarray:
    """Read a single (BGR, as returned by cv2) frame from a directory of chunked videos
    (e.g. videos/Camera1/)."""
    chunk_start = (frame_idx // chunk_size) * chunk_size
    video_file = video_dir / f'{chunk_start}.mp4'
    offset = frame_idx - chunk_start
    cap = cv2.VideoCapture(str(video_file))
    cap.set(cv2.CAP_PROP_POS_FRAMES, offset)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f'could not read frame {offset} from {video_file}')
    return frame


def export_frames_from_chunks(
    video_dir: Path,
    save_dir: Path,
    frame_idxs: np.ndarray,
    n_digits: int = 8,
    format: str = 'png',
    chunk_size: int = 3500,
) -> None:
    """Export individual frames (no context frames) from a directory of chunked videos."""
    save_dir.mkdir(parents=True, exist_ok=True)
    for frame_idx in frame_idxs:
        frame = get_frame_from_chunks(video_dir, int(frame_idx), chunk_size=chunk_size)
        cv2.imwrite(str(save_dir / f'img{str(int(frame_idx)).zfill(n_digits)}.{format}'), frame)


def load_absolute_positions(
    session_dir: Path,
    keypoints: list,
    frame_idxs: np.ndarray,
    n_frames_per_session: int,
    marker_file: str = 'markerDataset.csv',
) -> dict:
    """Load absolute 3D keypoint positions for both animals at specific frame indices.

    Returns
    -------
    dict with keys 'an1', 'an2', each an array of shape (len(frame_idxs), n_keypoints, 3),
    row-aligned with `frame_idxs`
    """
    coord_cols = {
        an: [f'absolutePosition_{an}_{kp}_{c}' for kp in keypoints for c in ['x', 'y', 'z']]
        for an in ['an1', 'an2']
    }
    usecols = coord_cols['an1'] + coord_cols['an2']
    df = pd.read_csv(session_dir / marker_file, usecols=usecols, nrows=n_frames_per_session)
    df = df.loc[frame_idxs]
    n_frames = len(frame_idxs)
    return {
        an: df[coord_cols[an]].to_numpy(dtype=float).reshape(n_frames, len(keypoints), 3)
        for an in ['an1', 'an2']
    }


def load_good_poses(
    session_dir: Path,
    keypoints: list,
    n_frames_per_session: int,
    marker_file: str = 'markerDataset.csv',
) -> tuple:
    """Load per-frame 3D poses for both animals, restricted to frames with good mocap for both.

    Returns
    -------
    poses : np.ndarray, shape (n_good, n_keypoints * 3 * 2)
        both animals' absolute 3D keypoint coordinates concatenated (an1 first, then an2)
    good_idxs : np.ndarray, shape (n_good,)
        original frame/row indices (into the video / markerDataset.csv) for each pose
    """
    coord_cols = {
        an: [f'absolutePosition_{an}_{kp}_{c}' for kp in keypoints for c in ['x', 'y', 'z']]
        for an in ['an1', 'an2']
    }
    usecols = coord_cols['an1'] + coord_cols['an2'] + ['goodFrame_an1', 'goodFrame_an2']
    df = pd.read_csv(session_dir / marker_file, usecols=usecols, nrows=n_frames_per_session)
    is_good = (df['goodFrame_an1'] == 1) & (df['goodFrame_an2'] == 1)
    good_idxs = df.index[is_good].to_numpy()
    poses = df.loc[is_good, coord_cols['an1'] + coord_cols['an2']].to_numpy(dtype=float)
    return poses, good_idxs


def select_prototype_frames_kmeans(
    poses: np.ndarray,
    good_idxs: np.ndarray,
    n_frames: int,
    context_frames: int = 2,
    max_seed_tries: int = 50,
) -> np.ndarray:
    """Select a diverse subset of frames via kmeans clustering on their poses.

    One frame closest to each of `n_frames` kmeans cluster centers is kept as a "prototype",
    mirroring the approach used for rat7m (lp3d_analysis.preprocess / scripts/preprocess/rat7m.py).
    Retries with a different kmeans seed if any two selected frames are close enough that their
    +/- context_frames export windows would overlap.

    Parameters
    ----------
    poses: shape (n_good_frames, n_features), row-aligned with `good_idxs`
    good_idxs: original frame/row indices corresponding to each row of `poses`
    n_frames: number of prototype frames to select
    context_frames: number of frames exported on either side of a selected frame; used only to
        check selected frames aren't too close together
    max_seed_tries: give up retrying for non-overlapping frames after this many seeds

    Returns
    -------
    np.ndarray of selected frame indices, sorted
    """
    if len(good_idxs) <= n_frames:
        return np.sort(good_idxs)

    def find_prototypes(seed):
        kmeans = KMeans(n_clusters=n_frames, random_state=seed, n_init=10)
        kmeans.fit(poses)
        centers = kmeans.cluster_centers_
        dists = np.linalg.norm(poses[:, None, :] - centers[None, :, :], axis=2)
        idxs_prototypes = np.argmin(dists, axis=0)
        return np.sort(good_idxs[idxs_prototypes])

    seed = 0
    idxs_prototypes = find_prototypes(seed)
    min_spacing = 2 * context_frames
    while np.any(np.diff(idxs_prototypes) <= min_spacing) and seed < max_seed_tries:
        seed += 1
        print(f'context frames overlapping with selected frames; trying seed = {seed}')
        idxs_prototypes = find_prototypes(seed)

    return idxs_prototypes
