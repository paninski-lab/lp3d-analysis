"""Functions for video handling."""

from pathlib import Path
from typing import Optional
import os
import shutil
import subprocess

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd


# def save_video(save_file, tmp_dir, framerate=20, frame_pattern='frame_%06d.jpeg', crf=23):
#     """

#     Parameters
#     ----------
#     save_file : str
#         absolute path of filename (including extension)
#     tmp_dir : str
#         temporary directory that stores frames of video; this directory will be deleted
#     framerate : float, optional
#         framerate of final video
#     frame_pattern : str, optional
#         string pattern used for naming frames in tmp_dir
#     crf : 23
#         compression rate factor; smaller is less compression, larger filesize

#     """

#     if os.path.exists(save_file):
#         os.remove(save_file)

#     # make mp4 from images using ffmpeg
#     call_str = \
#         'ffmpeg -r %f -i %s -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -crf %i %s' % (
#             framerate, os.path.join(tmp_dir, frame_pattern), crf, save_file)
#     print(call_str)
#     subprocess.run(['/bin/bash', '-c', call_str], check=True)

#     # delete tmp directory
#     shutil.rmtree(tmp_dir)

def save_video(save_file, tmp_dir, framerate=20, frame_pattern='frame_%06d.jpeg', crf=23):
    """Save video from a directory of frames using ffmpeg.
    
    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    tmp_dir : str
        temporary directory that stores frames of video
    framerate : float, optional
        framerate of final video
    frame_pattern : str, optional
        string pattern used for naming frames in tmp_dir
    crf : int, optional
        compression rate factor; smaller is less compression, larger filesize
    """
    # Verify directories exist
    if not os.path.exists(tmp_dir):
        raise RuntimeError(f"Temporary directory does not exist: {tmp_dir}")
    
    # Create a temporary file listing all frames
    frame_files = sorted([f for f in os.listdir(tmp_dir) if f.endswith('.jpeg')])
    if not frame_files:
        raise RuntimeError(f"No jpeg files found in {tmp_dir}")
    
    print(f"Found {len(frame_files)} frames in temporary directory")
    
    # Create a temporary file listing all frames
    list_file = os.path.join(tmp_dir, "frames.txt")
    with open(list_file, 'w') as f:
        for frame in frame_files:
            f.write(f"file '{os.path.join(tmp_dir, frame)}'\n")
    
    # Remove existing output file if it exists
    if os.path.exists(save_file):
        os.remove(save_file)
    
    # Construct ffmpeg command using the concat demuxer
    call_str = (
        f'ffmpeg -y -f concat -safe 0 -r {framerate} -i "{list_file}" '
        f'-c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" '
        f'-crf {crf} "{save_file}"'
    )
    
    print(f"Executing ffmpeg command:\n{call_str}")
    
    try:
        # Run ffmpeg with full error output
        result = subprocess.run(
            ['/bin/bash', '-c', call_str],
            check=True,
            capture_output=True,
            text=True
        )
        print("Video creation successful")
        
        # Verify the video file was created
        if not os.path.exists(save_file):
            raise RuntimeError(f"ffmpeg completed but output file not found: {save_file}")
            
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg command failed with error:\n{e.stderr}")
        print("\nDebug: First few frames in directory:")
        for frame in frame_files[:5]:
            print(frame)
        raise RuntimeError(f"Failed to create video. FFmpeg error: {e.stderr}")
        
    finally:
        # Clean up
        if os.path.exists(list_file):
            os.remove(list_file)
        if os.path.exists(save_file):
            print(f"Cleaning up temporary directory: {tmp_dir}")
            shutil.rmtree(tmp_dir)
        else:
            print(f"Keeping temporary directory for debugging: {tmp_dir}")


# a class that allows to pull up the frams from a video and the indexes of the frames
# If load a video into that class, I want indexes 0-100, it will load the frames and return a numpy array 
def get_frames_from_idxs(cap, idxs, resize=None, flip=False, grayscale=True, video_file=None):
    """Helper function to load video segments.

    Parameters
    ----------
    cap : cv2.VideoCapture object
    idxs : array-like
        frame indices into video
    resize : tuple
        (width, height)
    flip : bool
        horizontal flip
    grayscale : bool
        True to save grayscale frames, False to use color
    video_file : str
        absolute path to video file, required if cap is None (forces use of imagio)

    Returns
    -------
    np.ndarray
        returned frames of shape (n_frames, n_channels, ypix, xpix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if cap:
            if fr == 0 or not is_contiguous:
                cap.set(1, i)
            ret, frame = cap.read()
        else:
            # more precise, much slower. required for chickadee data
            frame = iio.imread(video_file, index=i)
            ret = True
        if ret:
            if fr == 0:
                if resize is None:
                    height, width, _ = frame.shape
                else:
                    width, height = resize
                if grayscale:
                    frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
                else:
                    frames = np.zeros((n_frames, 3, height, width), dtype="uint8")
            if resize is not None:
                frame = cv2.resize(frame, resize)
            if flip:
                frame = cv2.flip(frame, 1)
            if grayscale:
                frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                frames[fr] = frame.transpose(2, 0, 1)
        else:
            print(
                'warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices')
            break
    return frames


def export_frames(
    video_file: Path,
    save_dir: Path,
    frame_idxs: np.ndarray,
    format: str = "png",
    n_digits: int = 8,
    context_frames: int = 0,
    reader: str = 'cv2',
) -> None:
    """Export frames from a video file to images.

    Parameters
    ----------
    video_file: absolute path to video file from which to select frames
    save_dir: absolute path to directory in which selected frames are saved
    frame_idxs: indices of frames to grab
    format: only "png" currently supported
    n_digits: number of digits in image names
    context_frames: number of frames on either side of selected frame to also save
    reader: backend video reader for frame extraction:
        'cv2': default, works for all datasets except chickadee
        'imageio': much slower but more precise

    """

    cap = cv2.VideoCapture(str(video_file))

    # expand frame_idxs to include context frames
    if context_frames > 0:
        context_vec = np.arange(-context_frames, context_frames + 1)
        frame_idxs = (frame_idxs[None, :] + context_vec[:, None]).flatten()
        frame_idxs.sort()
        frame_idxs = frame_idxs[frame_idxs >= 0]
        frame_idxs = frame_idxs[frame_idxs < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        frame_idxs = np.unique(frame_idxs)

    # load frames from video
    if reader == 'cv2':
        frames = get_frames_from_idxs(cap, frame_idxs)
    elif reader == 'imageio':
        frames = get_frames_from_idxs(None, frame_idxs, video_file=str(video_file))
    else:
        raise NotImplementedError

    cap.release()

    # save out frames
    save_dir.mkdir(parents=True, exist_ok=True)
    for frame, idx in zip(frames, frame_idxs):
        cv2.imwrite(
            filename=str(save_dir / f"img{str(idx).zfill(n_digits)}.{format}"),
            img=cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2BGR),
        )


def export_cropped_frames(
    save_dir_full: Path,
    save_dir_crop: Path,
    frame_idxs: np.ndarray,
    labels: np.ndarray,
    scales: np.ndarray,
    resize: tuple,  # (width, height)
    context_frames: int = 2,
    w3d: float = 1.0,
    n_digits: int = 8,
    format: str = "png",
    nan_centroid_idxs: Optional[np.ndarray] = None,
):
    """Save cropped frames given an input directory of frames.

    Parameters
    ----------
    save_dir_full: absolute path to directory where full-size frames are saved
    save_dir_crop: absolute path to directory where cropped frames will be saved
    frame_idxs: frame indices to crop and save
    labels: shape (n_frames, n_keypoints, 2), keypoint coordinates
    scales: shape (n_frames,), scale factors for centroids
    resize: tuple (width, height)
    context_frames: number of frames on either side of the selected frame to save
    w3d: scale multiplier for bounding box size
    n_digits: number of digits in output image filenames
    format: image format, only "png" currently supported
    nan_centroid_idxs: indices into `labels` to cycle through to compute centroids
        if all NaNs are encountered

    Returns
    -------
    np.ndarray : one row per frame, each row is [x, y, h, w]

    """

    save_dir_crop.mkdir(parents=True, exist_ok=True)
    bboxes = []

    nan_count = 0

    for frame_idx, center_idx in enumerate(frame_idxs):

        # compute centroid for crop using center frame
        centroid = np.nanmean(labels[frame_idx], axis=0)
        if all(np.isnan(centroid)):
            centroid = np.nanmean(labels[nan_centroid_idxs[nan_count]], axis=0)
            this_scale = w3d * scales[nan_centroid_idxs[nan_count]]
            nan_count += 1
        else:
            this_scale = w3d * scales[frame_idx]

        x_min = int(centroid[0] - this_scale)
        x_max = int(centroid[0] + this_scale)
        y_min = int(centroid[1] - this_scale)
        y_max = int(centroid[1] + this_scale)

        # Loop over context frames
        for context_idx in range(center_idx - context_frames, center_idx + context_frames + 1):
            frame_name = f"img{str(context_idx).zfill(n_digits)}.{format}"
            frame_path = save_dir_full / frame_name

            if frame_path.exists():
                frame = cv2.imread(str(frame_path))

                # Handle out-of-bounds cropping by padding the frame with black pixels
                padded_frame = np.zeros((
                    frame.shape[0] + abs(min(0, y_min)) + max(0, y_max - frame.shape[0]),
                    frame.shape[1] + abs(min(0, x_min)) + max(0, x_max - frame.shape[1]),
                    frame.shape[2]),
                    dtype=np.uint8,
                )

                # Insert the original frame into the padded frame
                y_start = abs(min(0, y_min))
                x_start = abs(min(0, x_min))
                padded_frame[y_start:y_start + frame.shape[0],
                x_start:x_start + frame.shape[1]] = frame

                # Adjust crop coordinates to the padded frame
                y_min_padded = y_min + y_start
                y_max_padded = y_max + y_start
                x_min_padded = x_min + x_start
                x_max_padded = x_max + x_start

                # Crop the padded frame
                frame_cropped = padded_frame[y_min_padded:y_max_padded, x_min_padded:x_max_padded]

                # Resize the cropped frame
                frame_resized = cv2.resize(frame_cropped, resize, interpolation=cv2.INTER_LINEAR)

                # Save cropped frame
                cropped_save_dir = save_dir_crop / frame_name
                cv2.imwrite(str(cropped_save_dir), frame_resized)

        bboxes.append([x_min, y_min, y_max - y_min, x_max - x_min])

    return np.array(bboxes)


def make_video_snippet(
    video_file: str | list,
    preds_file: Optional[str | list] = None,
    clip_length: int = 30,
    likelihood_thresh: float = 0.9,
    save_dir: Optional[str] = None,
) -> tuple:
    """Create a snippet from a larger video that contains the most movement.

    Parameters
    ----------
    video_file: absolute path to original video file
    preds_file: csv file containing pose estimates; if not None, "movement" is measured from these
        pose estimates; if None, "movement" is measured using the pixel values
    clip_length: length of the snippet in seconds
    likelihood_thresh: only measure movement using the preds_file for keypoints with a likelihood
        above this threshold (0-1)
    save_dir: output directory

    Returns
    -------
    - absolute path to video snippet
    - snippet start: frame index
    - snippet start: seconds

    """

    if isinstance(video_file, str):
        video_files = [video_file]
    else:
        video_files = video_file
    if preds_file and isinstance(preds_file, str):
        preds_files = [preds_file]
    else:
        preds_files = preds_file

    # how large is the clip window?
    video = cv2.VideoCapture(video_files[0])
    fps = video.get(cv2.CAP_PROP_FPS)
    win_len = int(fps * clip_length)
    n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()

    # manage paths
    if save_dir is None:
        if preds_file is None:
            # save videos with video file
            save_dir = os.path.dirname(video_files[0])
        else:
            # save videos with csv file
            save_dir = os.path.dirname(preds_files[0])

    os.makedirs(save_dir, exist_ok=True)

    me = []
    srcs = []
    dsts = []
    for v, video_file in enumerate(video_files):

        print(f'processing video {v+1}/{len(video_files)}')
        src = video_file
        new_video_name = os.path.basename(video_file).replace(
            ".mp4", ".short.mp4"
        ).replace(
            ".avi", ".short.mp4"
        )
        dst = os.path.join(save_dir, new_video_name)
        srcs.append(src)
        dsts.append(dst)

        # make a `clip_length` second video clip that contains the highest keypoint motion energy
        if win_len >= n_frames:
            # short video, no need to shorten further. just copy existing video
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
        else:
            # compute motion energy
            if preds_files is None:
                # motion energy averaged over pixels
                me_ = compute_video_motion_energy(video_file=video_file)
            else:
                # load pose predictions
                df = pd.read_csv(preds_files[v], header=[0, 1, 2], index_col=0)
                # motion energy averaged over predicted keypoints
                me_ = compute_motion_energy_from_predection_df(df, likelihood_thresh)
            me.append(me_[:, None])

    clip_start_idx = 0
    clip_start_sec = 0.0

    if len(me) > 0:
        # find window
        df_me = pd.DataFrame({"me": np.stack(me, axis=1).sum(axis=1)})
        df_me_win = df_me.rolling(window=win_len, center=False).mean()
        # rolling places results in right edge of window, need to subtract this
        clip_start_idx = df_me_win.me.argmax() - win_len
        # convert to seconds
        clip_start_sec = int(clip_start_idx / fps)
        # if all predictions are bad, make sure we still create a valid snippet video
        if np.isnan(clip_start_sec) or clip_start_sec < 0:
            clip_start_sec = 0

        for src, dst in zip(srcs, dsts):
            # make clip
            if not os.path.exists(dst):
                ffmpeg_cmd = f"ffmpeg -ss {clip_start_sec} -i {src} -t {clip_length} {dst}"
                subprocess.run(ffmpeg_cmd, shell=True)

    return dsts, int(clip_start_idx), float(clip_start_sec)


def compute_video_motion_energy(
    video_file: str,
    resize_dims: int = 32,
) -> np.ndarray:
    # read all frames, reshape, chop off unwanted portions of beginning/end
    cap = cv2.VideoCapture(video_file)
    frames = get_frames_from_idxs(
        cap=cap,
        idxs=np.arange(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        resize=(resize_dims, resize_dims),
    )
    cap.release()
    frame_count = frames.shape[0]
    frames = np.reshape(frames, (frame_count, -1)).astype('float16')

    # take temporal diffs
    me = np.diff(frames, axis=0, prepend=0)

    # take absolute values and sum over all pixels to get motion energy
    me = np.sum(np.abs(me), axis=1)

    return me


def compute_motion_energy_from_predection_df(
    df: pd.DataFrame,
    likelihood_thresh: float,
) -> np.ndarray:

    # Convert predictions to numpy array and reshape
    kps_and_conf = df.to_numpy().reshape(df.shape[0], -1, 3)
    kps = kps_and_conf[:, :, :2]
    conf = kps_and_conf[:, :, -1]
    # Duplicate likelihood scores for x and y coordinates
    conf2 = np.concatenate([conf[:, :, None], conf[:, :, None]], axis=2)

    # Apply likelihood threshold
    kps[conf2 < likelihood_thresh] = np.nan

    # Compute motion energy
    me = np.nanmean(np.linalg.norm(kps[1:] - kps[:-1], axis=2), axis=-1)
    me = np.concatenate([[0], me])
    return me
