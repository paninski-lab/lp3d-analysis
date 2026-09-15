"""Crop the r24m dataset into a single-animal dataset, r24m-crop.

Each multi-animal frame in r24m/labeled-data/<session>_<camera>/ is split into two single-animal
crops, one per animal, saved to r24m-crop/labeled-data/<session>_<animal>_<camera>/ (i.e. sessions
are effectively doubled: <session>_an1 and <session>_an2 are treated as separate "sessions").

For each animal's crop: the bounding box is centered on the mean (x, y) of that animal's own 12
keypoints in that view/frame, sized to 1.2x the max of the tight keypoint extent's width/height
(a square box), then resized to dataset_info['r24m']['crop_size']. Out-of-bounds regions are
padded with black, matching the convention used elsewhere in this repo (e.g.
scripts/preprocess/rat-pup.py).

Outputs (mirroring the InD/OOD "_new" suffix convention used for r24m's own CollectedData files):
    labeled-data/<session>_<animal>_<camera>/imgXXXXXXXX.png
    CollectedData_<camera>.csv / CollectedData_<camera>_new.csv (single-animal keypoint names)
    bboxes_<camera>.csv / bboxes_<camera>_new.csv (columns: x, y, h, w; original-image pixel coords)
    calibrations/<session>_<animal>.toml (identical copies of r24m's per-session calibration)
"""
import shutil
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lp3d_analysis import dataset_info

# ----------------------------------------------------------------------------------------
# path info
# ----------------------------------------------------------------------------------------
src_dir = Path('/media/mattw/multiview-data/r24m')
dst_dir = Path('/media/mattw/multiview-data/r24m-crop')

data_info = dataset_info['r24m']
camera_names = data_info['cam_names']
keypoints = data_info['keypoints']
InD_animals = data_info['InD_animals']
OOD_animals = data_info['OOD_animals']
skeleton = data_info['skeleton']
skeleton_colors = data_info['skeleton_colors']
crop_size = data_info['crop_size']  # (width, height)

animal_ids = ['an1', 'an2']
labels_file = 'labels.pqt'
bbox_scale = 1.4  # multiplier on the max keypoint-extent dimension

run_crop = True
run_copy_calibrations = True
run_check_cropped_labels = True


def compute_bbox(xy: np.ndarray) -> tuple:
    """Square bbox (x, y, h, w): centered on the midpoint of the tight extent of `xy`
    (n_keypoints, 2) -- i.e. the average of the most extreme keypoints in x and in y -- sized to
    bbox_scale times the max of that extent's width/height."""
    x_min, y_min = np.nanmin(xy, axis=0)
    x_max, y_max = np.nanmax(xy, axis=0)
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = max(x_max - x_min, y_max - y_min) * bbox_scale
    return (
        int(round(center_x - side / 2)),
        int(round(center_y - side / 2)),
        int(round(side)),
        int(round(side)),
    )


def crop_array(frame: np.ndarray, x: int, y: int, h: int, w: int) -> np.ndarray:
    """Crop `frame` to the box (x, y, x + w, y + h), padding out-of-bounds regions with black."""
    height, width = frame.shape[:2]
    top_pad, left_pad = max(0, -y), max(0, -x)
    bottom_pad, right_pad = max(0, y + h - height), max(0, x + w - width)
    padded = np.zeros(
        (height + top_pad + bottom_pad, width + left_pad + right_pad, frame.shape[2]),
        dtype=frame.dtype,
    )
    padded[top_pad:top_pad + height, left_pad:left_pad + width] = frame
    y0, x0 = y + top_pad, x + left_pad
    return padded[y0:y0 + h, x0:x0 + w]


def crop_sessions(sessions: list, suffix: str) -> None:
    """Crop every session's frames (both animals, all cameras) and write per-camera
    CollectedData_<camera>{suffix}.csv and bboxes_<camera>{suffix}.csv."""
    for cam_name in camera_names:
        label_rows = {}
        bbox_rows = {}

        for session in sessions:
            df = pd.read_parquet(src_dir / 'labeled-data' / f'{session}_{cam_name}' / labels_file)
            for img_path, row in df.iterrows():
                frame_name = Path(img_path).name
                frame = cv2.imread(str(src_dir / img_path))

                for an in animal_ids:
                    xy = np.array([
                        [row[('mocap', f'{kp}_{an}', 'x')], row[('mocap', f'{kp}_{an}', 'y')]]
                        for kp in keypoints
                    ])
                    x, y, h, w = compute_bbox(xy)

                    cropped = crop_array(frame, x, y, h, w)
                    resized = cv2.resize(cropped, crop_size, interpolation=cv2.INTER_LINEAR)

                    save_dir = dst_dir / 'labeled-data' / f'{session}_{an}_{cam_name}'
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir / frame_name), resized)

                    xy_cropped = xy.copy()
                    xy_cropped[:, 0] = (xy_cropped[:, 0] - x) / w * crop_size[0]
                    xy_cropped[:, 1] = (xy_cropped[:, 1] - y) / h * crop_size[1]

                    new_path = f'labeled-data/{session}_{an}_{cam_name}/{frame_name}'
                    label_rows[new_path] = xy_cropped.flatten()
                    bbox_rows[new_path] = [x, y, h, w]

        pdindex = pd.MultiIndex.from_product(
            [['mocap'], keypoints, ['x', 'y']], names=['scorer', 'bodyparts', 'coords'],
        )
        df_labels = pd.DataFrame.from_dict(label_rows, orient='index', columns=pdindex)
        df_labels.to_csv(dst_dir / f'CollectedData_{cam_name}{suffix}.csv')

        df_bboxes = pd.DataFrame.from_dict(
            bbox_rows, orient='index', columns=['x', 'y', 'h', 'w'],
        )
        df_bboxes.to_csv(dst_dir / f'bboxes_{cam_name}{suffix}.csv')

        print(
            f'{cam_name}{suffix}: cropped {len(label_rows)} animal-frames '
            f'({len(sessions)} sessions x {len(animal_ids)} animals)'
        )


def copy_calibrations(sessions: list) -> None:
    """Duplicate each session's calibration toml once per animal (same physical calibration,
    named to match the doubled-session labeled-data convention)."""
    save_dir = dst_dir / 'calibrations'
    save_dir.mkdir(parents=True, exist_ok=True)
    for session in sessions:
        src_calib = src_dir / 'calibrations' / f'{session}.toml'
        for an in animal_ids:
            shutil.copy(src_calib, save_dir / f'{session}_{an}.toml')
    print(f'copied calibrations for {len(sessions)} sessions x {len(animal_ids)} animals')


def generate_labeled_data_check() -> None:
    """Plot CollectedData_* keypoints on the cropped images, one grid-of-cameras figure per
    frame, saved to labeled-data-check/. Mirrors the mechanics of
    scripts/preprocess/r24m.py's generate_calibration_check_plots (same figure type: real image
    + skeleton overlay, one panel per camera, one file per frame), but single-animal (no an1/an2
    color split -- per-limb skeleton_colors are used instead).
    """
    csv_files = {
        cam_name: pd.concat([
            pd.read_csv(dst_dir / f'CollectedData_{cam_name}.csv', header=[0, 1, 2], index_col=0),
            pd.read_csv(
                dst_dir / f'CollectedData_{cam_name}_new.csv', header=[0, 1, 2], index_col=0,
            ),
        ])
        for cam_name in camera_names
    }
    base_cam = camera_names[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()
    for ax in axes_flat[len(camera_names):]:
        ax.axis('off')

    n_frames = len(csv_files[base_cam].index)
    for count, index in enumerate(csv_files[base_cam].index):
        pieces = index.split('/')
        frame_name = pieces[2]
        session_an = pieces[1].replace(f'_{base_cam}', '')

        for ax, cam_name in zip(axes_flat, camera_names):
            ax.clear()
            row_index = index.replace(base_cam, cam_name)
            img_path = dst_dir / 'labeled-data' / f'{session_an}_{cam_name}' / frame_name
            frame = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
            xy = csv_files[cam_name].loc[row_index].to_numpy().reshape(-1, 2)

            ax.imshow(frame)
            ax.scatter(xy[:, 0], xy[:, 1], c='white', s=15)
            for (i0, i1), color in zip(skeleton, skeleton_colors):
                ax.plot(xy[[i0, i1], 0], xy[[i0, i1], 1], c=color, linewidth=1)
            ax.set_title(cam_name)
            ax.axis('off')

        fig.suptitle(session_an)
        save_dir = dst_dir / 'labeled-data-check' / session_an
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / frame_name, dpi=120, bbox_inches='tight')

        if (count + 1) % 100 == 0:
            print(f'{count + 1} / {n_frames} frames plotted')
    plt.close(fig)
    print(f'saved {n_frames} labeled-data-check plots to {dst_dir / "labeled-data-check"}')


if __name__ == '__main__':

    sessions = InD_animals + OOD_animals

    if run_crop:
        crop_sessions(InD_animals, suffix='')
        crop_sessions(OOD_animals, suffix='_new')

    if run_copy_calibrations:
        copy_calibrations(sessions)

    if run_check_cropped_labels:
        generate_labeled_data_check()
