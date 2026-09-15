"""Preprocessing script for the PAIR-R24M dataset.

Data is located at /media/mattw/multiview-data/_raw/pair-r24m/PAIR-R24M-Dataset
(see README.txt there for a description of the raw data).

This is a multi-animal (2 rats/frame) dataset. Sessions are restricted to metadata.csv's
calibration=4 viewset (a consistent 6-camera setup); see the comments next to
dataset_info['r24m']['InD_animals'] / ['OOD_animals'] for the RecordingIndex -> session mapping.

Current phase: validation only. Before doing any extraction we need to confirm that
    (1) there are enough frames where both animals have good mocap (goodFrame_an1 and
        goodFrame_an2 both == 1), and
    (2) for every session, the videos from all 6 camera views have a consistent frame count that
        matches the number of rows in markerDataset.csv.

Re: (2) -- every session's final video chunk (starting at frame 105000) is shorter than the
nominal 3500 frames, and by a slightly different amount per camera in some sessions, which would
break camera synchronization if used. Every *other* chunk (0 through 101500, 30 chunks x 3500
frames) is confirmed exactly 3500 frames and perfectly aligned across all 6 cameras in every
session, so the fix is to drop each session's final chunk entirely and only use the first
N_FRAMES_PER_SESSION (105000) frames / markerDataset.csv rows.

Calibration -> anipose toml conversion + reprojection sanity check is implemented below
(run_calibration_check). Not yet implemented: kmeans-based frame selection across both animals'
poses, full multi-animal frame extraction, and a separate crop/split script to produce a
single-animal dataset.
"""
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml

from lp3d_analysis import dataset_info
from lp3d_analysis.preprocess.r24m import (
    export_frames_from_chunks,
    get_frame_from_chunks,
    load_absolute_positions,
    load_good_poses,
    project_points,
    select_prototype_frames_kmeans,
    write_session_calibration_toml,
)


# ----------------------------------------------------------------------------------------
# path info
# ----------------------------------------------------------------------------------------
base_dir = Path('/media/mattw/multiview-data/_raw/pair-r24m/PAIR-R24M-Dataset')
data_dir_proc = Path('/media/mattw/multiview-data/r24m')

# collect dataset info
data_info = dataset_info['r24m']
camera_names = data_info['cam_names']
keypoints = data_info['keypoints']
InD_animals = data_info['InD_animals']
OOD_animals = data_info['OOD_animals']
skeleton = data_info['skeleton']
skeleton_colors = data_info['skeleton_colors']
crop_size = data_info['crop_size']

marker_file = 'markerDataset.csv'
video_size = (1328, 1048)  # (width, height); consistent across all r24m videos

# every session's final video chunk is dropped (see module docstring); frames/csv rows beyond
# this are not used
N_FRAMES_PER_SESSION = 105000

# set flags
run_check_good_frames = False
run_check_frame_counts = False
run_calibration_check = False
run_select_frames = False
run_extract_calibrations = False
run_extract_labeled_frames = False
run_calibration_check_all = False
run_prune_bad_frames = False
run_migrate_labels = False
run_collate_collected_data = True

# session/seed used for the single-frame calibration reprojection sanity check
calibration_check_session = InD_animals[0]
calibration_check_seed = 0

# kmeans frame selection: over-extract candidates now, manually cull bad reprojections later
# (same workflow used for rat7m)
frames_per_session = 75
context_frames = 0  # frames on either side of a selected frame that will later be exported
overwrite_frame_selection = False

labels_file = 'labels.pqt'
# lightning-pose-style multi-animal label columns: no separate "individuals" level, bodyparts
# are flattened as "{keypoint}_{animal}" (e.g. "SpineM_an1")
flat_bodyparts = [f'{kp}_{an}' for an in ['an1', 'an2'] for kp in keypoints]
pdindex = pd.MultiIndex.from_product(
    [['mocap'], flat_bodyparts, ['x', 'y']],
    names=['scorer', 'bodyparts', 'coords'],
)


def count_csv_rows(csv_file: Path) -> int:
    """Count data rows in a csv file (excludes header) without loading it into memory."""
    with open(csv_file) as f:
        return sum(1 for _ in f) - 1


def get_video_frame_count(video_file: Path) -> int:
    """Number of frames in a video file, from container metadata (fast, no decoding)."""
    result = subprocess.run(
        [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', str(video_file),
        ],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def report_good_frames_overlap(sessions: list) -> None:
    """Report, per session, how many frames have good mocap for both animals."""
    print('=== goodFrame_an1 & goodFrame_an2 overlap ===')
    total_good = 0
    total_rows = 0
    for session in sessions:
        csv_file = base_dir / session / marker_file
        df = pd.read_csv(csv_file, usecols=['goodFrame_an1', 'goodFrame_an2'])
        n_good = int(((df['goodFrame_an1'] == 1) & (df['goodFrame_an2'] == 1)).sum())
        n_rows = len(df)
        total_good += n_good
        total_rows += n_rows
        print(f'{session}: {n_good} / {n_rows} frames good for both animals ({100 * n_good / n_rows:.1f}%)')
    print(f'TOTAL: {total_good} / {total_rows} frames good for both animals ({100 * total_good / total_rows:.1f}%)')
    print()


def get_camera_frame_count_excl_final_chunk(session_dir: Path, camera_name: str) -> int:
    """Total frames across all video chunks for one camera in a session, excluding the final
    (short, not-guaranteed-synced) chunk."""
    video_dir = session_dir / 'videos' / camera_name
    video_files = sorted(video_dir.glob('*.mp4'), key=lambda f: int(f.stem))
    return sum(get_video_frame_count(f) for f in video_files[:-1])


def report_frame_count_consistency(sessions: list) -> None:
    """Report, per session, whether all camera views agree on frame count and match
    N_FRAMES_PER_SESSION, after dropping each session's final video chunk."""
    print(f'=== frame count consistency (final chunk dropped, expect {N_FRAMES_PER_SESSION}/cam) ===')
    for session in sessions:
        session_dir = base_dir / session
        n_rows = count_csv_rows(session_dir / marker_file)
        cam_counts = {
            cam: get_camera_frame_count_excl_final_chunk(session_dir, cam) for cam in camera_names
        }
        counts_set = set(cam_counts.values())
        matches_expected = counts_set == {N_FRAMES_PER_SESSION}
        has_enough_csv_rows = n_rows >= N_FRAMES_PER_SESSION
        status = 'OK' if (matches_expected and has_enough_csv_rows) else 'MISMATCH'
        print(f'{session}: marker rows={n_rows}, camera frame counts (excl. final chunk)={cam_counts} -> {status}')
    print()


def check_calibration_reprojection(session: str, seed: int = 0) -> None:
    """Convert a session's raw calibration to an anipose toml, then sanity-check it by
    reprojecting the absolute 3D marker positions of a random good-mocap frame onto each
    camera view and plotting the result on top of the actual video frame."""
    session_dir = base_dir / session

    calib_save_path = data_dir_proc / 'calibrations' / f'{session}.toml'
    write_session_calibration_toml(session_dir, camera_names, video_size, calib_save_path)
    calib = toml.load(calib_save_path)
    print(f'wrote calibration to {calib_save_path}')

    # pick a random frame with good mocap for both animals, within the usable frame range
    coord_cols = {
        an: [f'absolutePosition_{an}_{kp}_{c}' for kp in keypoints for c in ['x', 'y', 'z']]
        for an in ['an1', 'an2']
    }
    usecols = coord_cols['an1'] + coord_cols['an2'] + ['goodFrame_an1', 'goodFrame_an2']
    df = pd.read_csv(session_dir / marker_file, usecols=usecols, nrows=N_FRAMES_PER_SESSION)
    good_idxs = df.index[(df['goodFrame_an1'] == 1) & (df['goodFrame_an2'] == 1)].to_numpy()

    rng = np.random.default_rng(seed)
    frame_idx = int(rng.choice(good_idxs))
    row = df.loc[frame_idx]
    points_3d = {
        an: row[coord_cols[an]].to_numpy(dtype=float).reshape(-1, 3) for an in ['an1', 'an2']
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for c, (cam_name, ax) in enumerate(zip(camera_names, axes.flatten())):
        cam_calib = calib[f'cam_{c}']
        frame = get_frame_from_chunks(session_dir / 'videos' / cam_name, frame_idx)
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for an, color in [('an1', 'red'), ('an2', 'cyan')]:
            points_2d = project_points(points_3d[an], cam_calib)
            ax.scatter(points_2d[:, 0], points_2d[:, 1], c=color, s=15)
            for i0, i1 in skeleton:
                ax.plot(points_2d[[i0, i1], 0], points_2d[[i0, i1], 1], c=color, linewidth=1)
        ax.set_title(cam_name)
        ax.axis('off')
    fig.suptitle(f'{session} frame {frame_idx}')

    save_file = data_dir_proc / 'calibration-check' / f'{session}_frame{frame_idx}.png'
    save_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'saved calibration reprojection check to {save_file}')


def select_frames_for_sessions(sessions: list, n_frames: int, overwrite: bool = False) -> dict:
    """Select kmeans-diverse prototype frames (both animals good) for each session.

    Selected indices are cached to data_dir_proc/labeled-data/<session>_idxs.npy so re-running
    without `overwrite` is cheap.
    """
    selected = {}
    for session in sessions:
        idxs_file = data_dir_proc / 'labeled-data' / f'{session}_idxs.npy'
        if idxs_file.exists() and not overwrite:
            idxs = np.load(idxs_file)
            print(f'{session}: loaded {len(idxs)} cached frame idxs from {idxs_file}')
        else:
            session_dir = base_dir / session
            poses, good_idxs = load_good_poses(session_dir, keypoints, N_FRAMES_PER_SESSION, marker_file)
            idxs = select_prototype_frames_kmeans(
                poses, good_idxs, n_frames, context_frames=context_frames,
            )
            idxs_file.parent.mkdir(parents=True, exist_ok=True)
            np.save(idxs_file, idxs)
            print(f'{session}: selected {len(idxs)} / {len(good_idxs)} good frames -> {idxs_file}')
        selected[session] = idxs
    return selected


def extract_all_calibrations(sessions: list) -> None:
    """Write an anipose toml calibration for every session."""
    for session in sessions:
        save_path = data_dir_proc / 'calibrations' / f'{session}.toml'
        write_session_calibration_toml(base_dir / session, camera_names, video_size, save_path)
        print(f'{session}: wrote calibration to {save_path}')


def extract_labeled_frames(sessions: list) -> None:
    """Export each session's selected (kmeans) frames and their 2D reprojected labels.

    Frames are saved to labeled-data/<session>_<camera>/imgXXXXXXXX.png, and each camera
    directory gets a labels.pqt with both animals' reprojected 2D keypoints (see `pdindex`).
    """
    for session in sessions:
        idxs_file = data_dir_proc / 'labeled-data' / f'{session}_idxs.npy'
        frame_idxs = np.load(idxs_file)
        n_frames = len(frame_idxs)

        calib = toml.load(data_dir_proc / 'calibrations' / f'{session}.toml')
        session_dir = base_dir / session
        positions = load_absolute_positions(
            session_dir, keypoints, frame_idxs, N_FRAMES_PER_SESSION, marker_file,
        )

        print(f'{session}: exporting {n_frames} frames x {len(camera_names)} cameras')
        for c, cam_name in enumerate(camera_names):
            save_dir = data_dir_proc / 'labeled-data' / f'{session}_{cam_name}'
            export_frames_from_chunks(
                session_dir / 'videos' / cam_name, save_dir, frame_idxs,
            )

            cam_calib = calib[f'cam_{c}']
            pts2d_by_animal = []
            for an in ['an1', 'an2']:
                flat_3d = positions[an].reshape(-1, 3)
                flat_2d = project_points(flat_3d, cam_calib)
                pts2d_by_animal.append(flat_2d.reshape(n_frames, -1))
            data = np.concatenate(pts2d_by_animal, axis=1)

            row_index = [
                f'labeled-data/{session}_{cam_name}/img{str(int(i)).zfill(8)}.png'
                for i in frame_idxs
            ]
            df_labels = pd.DataFrame(data, columns=pdindex, index=row_index)
            df_labels.to_parquet(save_dir / labels_file)
        print(f'{session}: done')


def generate_calibration_check_plots(sessions: list) -> None:
    """Save a per-camera reprojection check plot for every selected frame in every session."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    for ax in axes_flat[len(camera_names):]:
        ax.axis('off')

    for session in sessions:
        idxs_file = data_dir_proc / 'labeled-data' / f'{session}_idxs.npy'
        frame_idxs = np.load(idxs_file)

        labels = {
            cam_name: pd.read_parquet(
                data_dir_proc / 'labeled-data' / f'{session}_{cam_name}' / labels_file
            )
            for cam_name in camera_names
        }

        save_dir = data_dir_proc / 'calibration-check' / session
        save_dir.mkdir(parents=True, exist_ok=True)

        n_kp = len(keypoints)
        for frame_idx in frame_idxs:
            frame_name = f'img{str(int(frame_idx)).zfill(8)}.png'
            for ax, cam_name in zip(axes_flat, camera_names):
                ax.clear()
                img_path = data_dir_proc / 'labeled-data' / f'{session}_{cam_name}' / frame_name
                frame = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
                row = labels[cam_name].loc[f'labeled-data/{session}_{cam_name}/{frame_name}']
                ax.imshow(frame)
                row_vals = row.to_numpy()
                points_2d = {
                    'an1': row_vals[:2 * n_kp].reshape(n_kp, 2),
                    'an2': row_vals[2 * n_kp:].reshape(n_kp, 2),
                }
                for an, color in [('an1', 'red'), ('an2', 'cyan')]:
                    pts = points_2d[an]
                    ax.scatter(pts[:, 0], pts[:, 1], c=color, s=15)
                    for i0, i1 in skeleton:
                        ax.plot(pts[[i0, i1], 0], pts[[i0, i1], 1], c=color, linewidth=1)
                ax.set_title(cam_name)
                ax.axis('off')
            fig.suptitle(f'{session} frame {frame_idx}')
            plt.savefig(save_dir / f'frame{int(frame_idx):08d}.png', dpi=120, bbox_inches='tight')
        print(f'{session}: saved {len(frame_idxs)} calibration-check plots to {save_dir}')
    plt.close(fig)


def prune_bad_frames(sessions: list) -> None:
    """Propagate manual deletions from calibration-check/ back into labeled-data/.

    A frame is considered "bad" (manually culled) if its check plot
    (calibration-check/<session>/frame########.png) no longer exists. For every such frame,
    removes the corresponding image from every camera's labeled-data directory and drops the
    matching row from that camera's labels.pqt, then rewrites <session>_idxs.npy to the
    surviving frame indices.
    """
    for session in sessions:
        idxs_file = data_dir_proc / 'labeled-data' / f'{session}_idxs.npy'
        original_idxs = np.load(idxs_file)

        check_dir = data_dir_proc / 'calibration-check' / session
        surviving_idxs = np.array([
            idx for idx in original_idxs
            if (check_dir / f'frame{int(idx):08d}.png').exists()
        ])
        removed_idxs = np.setdiff1d(original_idxs, surviving_idxs)

        if len(removed_idxs) == 0:
            print(f'{session}: no frames removed')
            continue

        for cam_name in camera_names:
            save_dir = data_dir_proc / 'labeled-data' / f'{session}_{cam_name}'

            for idx in removed_idxs:
                img_file = save_dir / f'img{int(idx):08d}.png'
                if img_file.exists():
                    img_file.unlink()

            labels_path = save_dir / labels_file
            df = pd.read_parquet(labels_path)
            drop_rows = [
                f'labeled-data/{session}_{cam_name}/img{int(idx):08d}.png' for idx in removed_idxs
            ]
            df = df.drop(index=drop_rows, errors='ignore')
            df.to_parquet(labels_path)

        np.save(idxs_file, surviving_idxs)
        print(
            f'{session}: removed {len(removed_idxs)} / {len(original_idxs)} frames, '
            f'{len(surviving_idxs)} remain'
        )


def migrate_labels_to_flat_bodyparts(sessions: list) -> None:
    """One-time migration: relabel existing labels.pqt columns from the 4-level
    (scorer, individuals, bodyparts, coords) scheme to the lightning-pose-style flat scheme
    (scorer, bodyparts, coords) with bodyparts named "{keypoint}_{animal}". Values are
    unchanged -- this only renames columns."""
    for session in sessions:
        for cam_name in camera_names:
            path = data_dir_proc / 'labeled-data' / f'{session}_{cam_name}' / labels_file
            df = pd.read_parquet(path)
            if 'individuals' not in df.columns.names:
                print(f'{session} {cam_name}: already flat, skipping')
                continue
            new_cols = pd.MultiIndex.from_tuples(
                [(scorer, f'{bp}_{ind}', coord) for scorer, ind, bp, coord in df.columns],
                names=['scorer', 'bodyparts', 'coords'],
            )
            df.columns = new_cols
            df.to_parquet(path)
            print(f'{session} {cam_name}: migrated to flat bodyparts')


def collate_collected_data(ind_sessions: list, ood_sessions: list) -> None:
    """Concatenate labels.pqt into CollectedData_<camera>.csv (InD) and
    CollectedData_<camera>_new.csv (OOD) per camera, at the dataset root, in lightning-pose
    format (flat "{keypoint}_{animal}" bodyparts)."""
    for cam_name in camera_names:
        for sessions, suffix in [(ind_sessions, ''), (ood_sessions, '_new')]:
            dfs = [
                pd.read_parquet(
                    data_dir_proc / 'labeled-data' / f'{session}_{cam_name}' / labels_file
                )
                for session in sessions
            ]
            df = pd.concat(dfs, axis=0)
            save_path = data_dir_proc / f'CollectedData_{cam_name}{suffix}.csv'
            df.to_csv(save_path)
            print(f'{cam_name}{suffix}: collated {len(df)} frames across {len(sessions)} sessions -> {save_path}')


if __name__ == '__main__':

    sessions = InD_animals + OOD_animals

    if run_check_good_frames:
        report_good_frames_overlap(sessions)

    if run_check_frame_counts:
        report_frame_count_consistency(sessions)

    if run_calibration_check:
        check_calibration_reprojection(calibration_check_session, seed=calibration_check_seed)

    if run_select_frames:
        selected_idxs = select_frames_for_sessions(
            sessions, frames_per_session, overwrite=overwrite_frame_selection,
        )

    if run_extract_calibrations:
        extract_all_calibrations(sessions)

    if run_extract_labeled_frames:
        extract_labeled_frames(sessions)

    if run_calibration_check_all:
        generate_calibration_check_plots(sessions)

    if run_prune_bad_frames:
        prune_bad_frames(sessions)

    if run_migrate_labels:
        migrate_labels_to_flat_bodyparts(sessions)

    if run_collate_collected_data:
        collate_collected_data(InD_animals, OOD_animals)
