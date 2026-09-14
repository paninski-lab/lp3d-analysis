"""Crop the rat-pup-2 dataset to per-frame square bounding boxes around the animal.

Three modes, selected with --labels, --videos, or --videos-detector:

--labels: for each labeled frame, a square bounding box is computed that tightly encloses the
    ground-truth keypoints, centered on the midpoint of the (possibly non-square) keypoint
    extent -- i.e. the shorter side is centered within the box defined by the longer side. This
    box is then padded by 20% (10% on each side) as a buffer. Frames are cropped to this box,
    with any region falling outside the original image padded with black pixels. Cropping is done
    independently per camera view: each `CollectedData_<camera>.csv` file is cropped using only
    its own frames' keypoints, producing a matching `CollectedData_<camera>.csv` and
    `bbox_<camera>.csv` in the output dataset.

--videos: the video `spliced2_<camera>.mp4` is a frame-by-frame re-encoding of the same frames
    used to build `CollectedData_<camera>.csv` (frame i of the video == row i of the CSV), so no
    detector model is needed. Each video frame is cropped using the exact same bbox logic as
    --labels (ground-truth keypoints for that frame), and the (square) crop is then resized to a
    fixed 320x320 for the saved video. The bbox CSV still records each frame's true (pre-resize)
    crop box in original-video pixel coordinates -- accurate for that frame's actual crop -- so
    that, together with the fixed 320 output size, keypoints can be mapped between the resized
    video and the original frame.

--videos-detector: for each camera's full video, a pretrained detector model is used (via the
    `litpose` CLI:
    https://lightning-pose.readthedocs.io/en/latest/source/user_guide_advanced/cropzoom_pipeline.html)
    to predict keypoints, from which a per-frame bounding box is computed with the same 20%
    buffer used above (`--crop_ratio 1.2`), matching the crop statistics the pose model will be
    trained on. Since the video is a splice of non-contiguous frames, bbox smoothing is skipped.
    The video is cropped to these boxes and every (square) frame is resized to a fixed
    320x320, producing a cropped video and matching bbox CSV per camera in the output dataset.

Bounding box file format:
https://lightning-pose.readthedocs.io/en/latest/source/directory_structure_reference/bounding_box_file_format.html
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ----------------------------------------------------------------------------------------
# path info
# ----------------------------------------------------------------------------------------
src_dir = Path('/media/mattw/multiview-data/rat-pup-2')
dst_dir = Path('/media/mattw/multiview-data/rat-pup-2-crop')
detector_model_dir = Path('/media/mattw/multiview/rat-pup-2/vits_dino_0')

camera_names = [f'Camera{i}' for i in range(1, 7)]

buffer_frac = 0.2  # total fractional increase in bbox side length (10% added to each side)
video_resize = 320  # final (square) pixel size of cropped video frames


def compute_bbox(xy: np.ndarray) -> tuple[int, int, int, int]:
    """Compute a padded square bounding box (x, y, h, w) around one frame's keypoints.

    `xy` has shape (n_keypoints, 2); NaN keypoints (unlabeled) are ignored. The box is
    centered on the midpoint of the tight keypoint extent and sized to its longer side, so the
    shorter side ends up centered on the animal. The box is then padded by `buffer_frac`.
    """
    x_min, x_max = np.nanmin(xy[:, 0]), np.nanmax(xy[:, 0])
    y_min, y_max = np.nanmin(xy[:, 1]), np.nanmax(xy[:, 1])
    center_x, center_y = (x_min + x_max) / 2, (y_min + y_max) / 2
    side = max(x_max - x_min, y_max - y_min) * (1 + buffer_frac)
    return (
        int(round(center_x - side / 2)),
        int(round(center_y - side / 2)),
        int(round(side)),
        int(round(side)),
    )


def process_camera(camera_name: str):
    labels_file = f'CollectedData_{camera_name}.csv'
    bbox_file = f'bbox_{camera_name}.csv'

    df = pd.read_csv(src_dir / labels_file, header=[0, 1, 2], index_col=0)

    bboxes = {}
    cropped_coords = np.full(df.shape, np.nan)

    for i, img_path in enumerate(df.index):
        xy = df.iloc[i].to_numpy().reshape(-1, 2)

        x, y, h, w = compute_bbox(xy)
        bboxes[img_path] = [x, y, h, w]

        # crop frame, padding out-of-bounds regions with black, and save
        dst_img_file = dst_dir / img_path
        dst_img_file.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(src_dir / img_path)
        img.crop((x, y, x + w, y + h)).save(dst_img_file)

        # remap labels into cropped-frame coordinates
        xy_cropped = xy - [x, y]
        cropped_coords[i] = xy_cropped.flatten()

    # save remapped labels
    df_cropped = pd.DataFrame(cropped_coords, columns=df.columns, index=df.index)
    df_cropped.to_csv(dst_dir / labels_file)

    # save bounding boxes (labeled-frames mode: path-keyed, one row per label)
    bbox_df = pd.DataFrame.from_dict(
        bboxes, orient='index', columns=['x', 'y', 'h', 'w'],
    )
    bbox_df.to_csv(dst_dir / bbox_file)

    print(f'{camera_name}: saved {len(df)} cropped frames + labels + bboxes')


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


def process_camera_video_from_labels(camera_name: str):
    video_file = src_dir / 'videos' / f'spliced2_{camera_name}.mp4'
    expected_name = f'spliced2_{camera_name}.mp4'
    if video_file.name != expected_name:
        raise ValueError(
            f'process_camera_video_from_labels assumes frame-for-frame correspondence with '
            f'CollectedData_{camera_name}.csv, which only holds for a video named '
            f'"{expected_name}" (got "{video_file.name}").'
        )

    df = pd.read_csv(src_dir / f'CollectedData_{camera_name}.csv', header=[0, 1, 2], index_col=0)

    cap = cv2.VideoCapture(str(video_file))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if n_frames != len(df):
        raise ValueError(
            f'{video_file.name} has {n_frames} frames but CollectedData_{camera_name}.csv has '
            f'{len(df)} rows; frame-for-frame correspondence cannot be assumed.'
        )

    dst_videos_dir = dst_dir / 'videos'
    dst_videos_dir.mkdir(parents=True, exist_ok=True)

    # cv2.VideoWriter can't encode H.264 in this environment (its bundled ffmpeg has no
    # libx264), so write frames to a temporary legacy-MPEG-4 file first, then re-encode
    # to H.264 below via the system ffmpeg (through moviepy). PyNvVideoCodec/NVDEC's
    # indexed random-access decoding (used at predict time) is unreliable on legacy
    # MPEG-4 Part 2 streams, so the final saved video must not be left in that codec.
    tmp_video_file = dst_videos_dir / f'_tmp_{video_file.name}'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(
        str(tmp_video_file), fourcc, fps, (video_resize, video_resize),
    )

    bboxes = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'failed to read frame {i} from {video_file.name}')

        xy = df.iloc[i].to_numpy().reshape(-1, 2)
        x, y, h, w = compute_bbox(xy)
        bboxes.append([x, y, h, w])

        cropped = crop_array(frame, x, y, h, w)
        resized = cv2.resize(cropped, (video_resize, video_resize), interpolation=cv2.INTER_LINEAR)
        writer.write(resized)

    cap.release()
    writer.release()

    # re-encode the temporary mpeg4 video to H.264 (see comment above) and discard it
    from moviepy import VideoFileClip

    clip = VideoFileClip(str(tmp_video_file))
    clip.write_videofile(str(dst_videos_dir / video_file.name), codec='libx264')
    clip.close()
    tmp_video_file.unlink()

    # save bounding boxes (video mode: positional, one row per video frame, path ignored);
    # h, w are each frame's true pre-resize crop size in original-video pixel coordinates
    bbox_df = pd.DataFrame(bboxes, columns=['x', 'y', 'h', 'w'])
    bbox_df.to_csv(dst_videos_dir / f'bbox_{camera_name}.csv')

    print(f'{camera_name}: saved {n_frames}-frame cropped video + bbox to {dst_videos_dir}')


def process_camera_video_detector(camera_name: str):
    video_file = src_dir / 'videos' / f'spliced2_{camera_name}.mp4'
    dst_videos_dir = dst_dir / 'videos'
    dst_videos_dir.mkdir(parents=True, exist_ok=True)

    # 1. run the pretrained detector model on the full video
    subprocess.run(
        ['litpose', 'predict', str(detector_model_dir), str(video_file)], check=True,
    )

    # 2. compute per-frame bboxes from the detector's predicted keypoints, using the same 20%
    # buffer used for the labeled dataset; no smoothing since the video splices together
    # non-contiguous frames
    subprocess.run(
        [
            'litpose', 'create_bbox', str(detector_model_dir), str(video_file),
            '--crop_ratio', str(1 + buffer_frac),
        ],
        check=True,
    )

    # 3. crop the video to the per-frame bboxes
    subprocess.run(
        ['litpose', 'crop', str(detector_model_dir), str(video_file)], check=True,
    )

    # 4. copy the bbox csv, and resize the (square) cropped video to a fixed size
    bbox_src = detector_model_dir / 'video_preds' / f'{video_file.stem}_bbox.csv'
    shutil.copy(bbox_src, dst_videos_dir / f'bbox_{camera_name}.csv')

    from moviepy import VideoFileClip

    cropped_video_src = detector_model_dir / 'cropped_videos' / f'cropped_{video_file.name}'
    clip = VideoFileClip(str(cropped_video_src))
    clip.resized(new_size=(video_resize, video_resize)).write_videofile(
        str(dst_videos_dir / video_file.name), codec='libx264',
    )

    print(f'{camera_name}: saved cropped video + bbox to {dst_videos_dir}')


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--labels', action='store_true',
        help='crop the labeled frame dataset using ground-truth keypoint bounding boxes',
    )
    group.add_argument(
        '--videos', action='store_true',
        help=(
            'crop the full videos using ground-truth keypoints, relying on frame-for-frame '
            'correspondence with the labeled dataset (no detector model needed)'
        ),
    )
    group.add_argument(
        '--videos-detector', action='store_true',
        help='crop the full videos using a pretrained detector model',
    )
    args = parser.parse_args()

    for camera_name in camera_names:
        if args.labels:
            process_camera(camera_name)
        elif args.videos:
            process_camera_video_from_labels(camera_name)
        else:
            process_camera_video_detector(camera_name)


if __name__ == '__main__':
    main()
