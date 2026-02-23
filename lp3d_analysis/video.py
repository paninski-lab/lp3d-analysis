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
import matplotlib.pyplot as plt


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
    frame_idxs_save: None | np.ndarray = None,
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
    frame_idxs_save: indices to use for filename; if None, uses frame_idxs. In general this should
        not be used, but there are some datasets where frame count offsets lead to a mismatch that
        needs to be resolved this way
    format: only "png" currently supported
    n_digits: number of digits in image names
    context_frames: number of frames on either side of selected frame to also save
    reader: backend video reader for frame extraction:
        'cv2': default, works for all datasets except chickadee
        'imageio': much slower but more precise

    """

    def add_context(idxs, ctx, cap):
        if ctx <= 0:
            return idxs
        context_vec = np.arange(-ctx, ctx + 1)
        idxs = (idxs[None, :] + context_vec[:, None]).flatten()
        idxs.sort()
        idxs = idxs[idxs >= 0]
        idxs = idxs[idxs < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))]
        idxs = np.unique(idxs)
        return idxs

    cap = cv2.VideoCapture(str(video_file))

    # expand frame_idxs to include context frames
    frame_idxs = add_context(frame_idxs, context_frames, cap)

    # take care of mismatch between selected frames and save names
    if frame_idxs_save is None:
        frame_idxs_save = frame_idxs
    else:
        frame_idxs_save = add_context(frame_idxs_save, context_frames, cap)
        l1 = len(frame_idxs)
        l2 = len(frame_idxs_save)
        assert l1 == l2, f'frame_idxs (len={l1}) and frame_idxs_save (len={l2}) must be same length'

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
    for frame, idx, idx_save in zip(frames, frame_idxs, frame_idxs_save):
        cv2.imwrite(
            filename=str(save_dir / f"img{str(idx_save).zfill(n_digits)}.{format}"),
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
    idx_min: int = 0,
    idx_max: Optional[int] = None,
    cam_idxs: Optional[list] = None,
    save_dir: Optional[str] = None,
) -> tuple:
    """Create a snippet from a larger video(s) that contains the most movement.

    Parameters
    ----------
    video_file: absolute path to original video file
    preds_file: csv file containing pose estimates; if not None, "movement" is measured from these
        pose estimates; if None, "movement" is measured using the pixel values
    clip_length: length of the snippet in seconds
    likelihood_thresh: only measure movement using the preds_file for keypoints with a likelihood
        above this threshold (0-1)
    idx_min: minimum frame number to consider from original video
    idx_max: maximum frame number to consider from original video
    cam_idxs: subset of cameras to use for computing motion energy; snippets will be created for
        all cameras
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

        src = video_file
        new_video_name = os.path.basename(video_file).replace(
            ".mp4", ".short.mp4"
        ).replace(
            ".avi", ".short.mp4"
        )
        dst = os.path.join(save_dir, new_video_name)
        srcs.append(src)
        dsts.append(dst)

        if cam_idxs is not None and v not in cam_idxs:
            continue

        print(f'processing video {v+1}/{len(video_files)}')

        # make a `clip_length` second video clip that contains the highest keypoint motion energy
        if win_len >= n_frames:
            # short video, no need to shorten further. just copy existing video
            if not os.path.exists(dst):
                shutil.copyfile(src, dst)
        else:
            # compute motion energy
            if preds_files is None:
                # motion energy averaged over pixels
                me_ = compute_video_motion_energy(
                    video_file=video_file, idx_min=idx_min, idx_max=idx_max,
                )
            else:
                # load pose predictions
                df = pd.read_csv(preds_files[v], header=[0, 1, 2], index_col=0)
                # motion energy averaged over predicted keypoints
                me_ = compute_motion_energy_from_predection_df(df, likelihood_thresh)
            me.append(me_[idx_min:, None])

    clip_start_idx = 0
    clip_start_sec = 0.0

    if len(me) > 0:
        # find window
        df_me = pd.DataFrame({"me": np.hstack(me).sum(axis=1)})
        df_me_win = df_me.rolling(window=win_len, center=False).mean()
        # rolling places results in right edge of window, need to subtract this
        # also need to add back in minimum index
        clip_start_idx = df_me_win.me.argmax() - win_len + idx_min
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
    idx_min: int = 0,
    idx_max: Optional[int] = None,
) -> np.ndarray:
    # read frames, reshape
    cap = cv2.VideoCapture(video_file)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if idx_max is None:
        idxs = np.arange(idx_min, total_frames)
    else:
        idxs = np.arange(idx_min, min(idx_max, total_frames))

    frames = get_frames_from_idxs(
        cap=cap,
        idxs=idxs,
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


def create_video_grid(
    video_files: list,
    save_file: str,
    grid_layout: tuple,
    scale_width: int = 256,
    scale_height: int = 256,
) -> None:
    """Combine multiple videos into a grid layout.

    Args:
        video_files (list): list of input video file paths.
        save_file (str): path for the output video file.
        grid_layout (tuple): tuple (rows, cols) specifying the grid layout.
        scale_width (int): width to scale each video.
        scale_height (int): height to scale each video.

    """
    num_videos = len(video_files)
    rows, cols = grid_layout

    if num_videos > rows * cols:
        raise ValueError(f"Grid layout ({rows}x{cols}) cannot fit {num_videos} videos.")

    os.makedirs(os.path.dirname(save_file), exist_ok=True)

    # build the input arguments for ffmpeg
    input_args = []
    for path in video_files:
        input_args.extend(["-i", path])

    # scale and label each video
    filter_parts = []
    for i in range(num_videos):
        filter_parts.append(f"[{i}:v]scale={scale_width}:{scale_height}[v{i}]")

    # create xstack layout
    layout_parts = []
    for r in range(rows):
        for c in range(cols):
            idx = r * cols + c
            if idx < num_videos:
                layout_parts.append(f"{c * scale_width}_{r * scale_height}")

    layout = "|".join(layout_parts)
    filter_complex = "; ".join(
        filter_parts) + f"; {' '.join(f'[v{i}]' for i in range(num_videos))}xstack=inputs={num_videos}:layout={layout}[vout]"

    # build ffmpeg command
    cmd = [
        "ffmpeg",
        *input_args,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264",  # Use H.264 codec
        "-crf", "23",  # Adjust quality (lower is better quality)
        "-preset", "fast",  # Encoding speed
        save_file
    ]

    # run the command
    try:
        subprocess.run(cmd, check=True)
        print(f"Video successfully created at {save_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")



def make_labeled_video(
    save_file, caps, points, labels=None, likelihood_thresh=0.05,
    max_frames=None, idxs=None, skeleton=None, leg_names=None,
    colors={'rng=0': 'red', 'rng=1': 'red', 'rng=2': 'red',
            'singleview': 'blue', 'multiview': 'blue'},
    markers={'rng=0': 'o', 'rng=1': 'o', 'rng=2': 'o',
             'singleview': 's', 'multiview': '^'},
    colors_skeleton=None,linewidth_error =1.5, markersize=6, framerate=20, height=4, model_type=None,
    hide_low_likelihood=False, keypoints_to_plot=None, resize_dims=[256, 256],
):
    
    # Input validation
    if not caps:
        raise ValueError("No video captures provided")
    if not points:
        raise ValueError("No points data provided")
    if not save_file:
        raise ValueError("No save file path provided")

    # tried to add for the case where we have the filled and not filled markers 
    # Default marker styles if none provided
    if markers is None:
        markers = {model: 'o' for model in labels} if labels else {'default': 'o'}
    
        
    # Setup temporary directory
    tmp_dir = os.path.join(os.path.dirname(save_file), 'tmpZzZ')
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir)
    print(f"Created temporary directory at: {tmp_dir}")

    # Get view information
    view_names = list(caps.keys())
    n_views = len(view_names)
    
    # Pre-process frame dimensions and create scaled points
    view_dimensions = {}
    scaled_points = {}
    
    # # Calculate optimal grid layout
    n_cols = int(np.ceil(np.sqrt(n_views)))
    n_rows = int(np.ceil(n_views / n_cols))

    print("Pre-processing frame dimensions and scaling markers...")
    for view in view_names:
        # Load frame 0 to get dimensions so we will use it later on 
        frame = get_frames_from_idxs(caps[view], [0], resize=None)
        if frame is None or frame.shape[0] == 0:
            raise ValueError(f"Failed to read initial frame for view {view}")
            
        # Record original dimensions
        h, w = frame.shape[2:4]  # Get height and width from frame dimensions
        view_dimensions[view] = {'height': h, 'width': w}
        
        # Scale points for this view
        if view in points:
            scaled_points[view] = []
            for point_dict in points[view]:
                scaled_dict = {}
                for marker_name, marker_vals in point_dict.items():
                    # Create copy of marker values
                    scaled_vals = marker_vals.copy()
                    # Scale x and y coordinates
                    scaled_vals[:, 0] = marker_vals[:, 0] * resize_dims[0] / w
                    scaled_vals[:, 1] = marker_vals[:, 1] * resize_dims[1] / h
                    # Scale uncertainty values if they exist
                    if marker_vals.shape[1] > 3:
                        scaled_vals[:, 3] = marker_vals[:, 3] * resize_dims[0] / w
                        scaled_vals[:, 4] = marker_vals[:, 4] * resize_dims[1] / h
                  
                    scaled_dict[marker_name] = scaled_vals
                scaled_points[view].append(scaled_dict)
    
    print("Frame dimensions:")
    for view, dims in view_dimensions.items():
        print(f"View {view}: {dims['width']}x{dims['height']}")
    
    # Create figure with calculated layout
    h = height
    w = h * (resize_dims[0] / resize_dims[1])

    '''
    This is for the type of video for the shape of the plots
    '''
    # fig, axes = plt.subplots(2, 1, figsize=(w, 2 * h))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*w, n_rows*h))
    axes = axes.flatten()
    
    # Configure axes
    for idx in range(n_views, len(axes)):
        axes[idx].set_visible(False)
    for ax in axes[:n_views]:
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim([0, resize_dims[0]])
        ax.set_ylim([resize_dims[1], 0])
    plt.subplots_adjust(wspace=0, hspace=0, left=0, bottom=0, right=1, top=1)
    
    # Text properties remain the same
    txt_kwargs = {
        'fontsize': 6,
        'horizontalalignment': 'left',
        'verticalalignment': 'bottom',
        'fontname': 'monospace',
        'transform': axes[0].transAxes,
    }
    txt_fr_kwargs = {
        'fontsize': 6,
        'color': [1, 1, 1],
        'horizontalalignment': 'left',
        'verticalalignment': 'top',
        'fontname': 'monospace',
        'bbox': dict(facecolor='k', alpha=0.25, edgecolor='none'),
        'transform': axes[0].transAxes
    }
    
    # Main processing loop
    frames_saved = 0
    for i, idx in enumerate(idxs):
        if i % 100 == 0:
            print(f'Processing frame {i:03d}/{len(idxs):03d}')
            
        if max_frames is not None and i >= max_frames:
            break
            
        for idx_view, view in enumerate(view_names):
            ax = axes[idx_view]
            ax.clear()
            
            # Read and process frame
            frame = get_frames_from_idxs(caps[view], [idx], resize=None)
            if frame is None or frame.shape[0] == 0:
                print(f"Warning: Failed to read frame {idx} for view {view}")
                continue
                
            # Resize frame
            frame_resized = cv2.resize(frame[0, 0], (resize_dims[0], resize_dims[1]))
            ax.imshow(frame_resized, vmin=0, vmax=255, cmap='gray')
            
            # Add view label
            ax.text(0.02, 0.92, f'View {view}', color='white',
                   bbox=dict(facecolor='k', alpha=0.25, edgecolor='none'),
                   transform=ax.transAxes)

            # Plot markers and skeleton using scaled points
            if view in scaled_points:
                for p, point_dict in enumerate(scaled_points[view]):
                    label_name = labels[p] if labels is not None else f"Model {p}"
                    
                    # Plot markers
                    for marker_name, marker_vals in point_dict.items():
                        if keypoints_to_plot is not None and marker_name not in keypoints_to_plot:
                            continue
                            
                        if idx >= len(marker_vals):
                            print(f"Warning: Index {idx} out of bounds for marker {marker_name}")
                            continue
                            
                        if marker_vals[idx, 2] < likelihood_thresh and hide_low_likelihood:
                            continue

                        # marker_color = colors.get(label_name, 'gray')
                        marker_color = None
                        if leg_names is not None:
                            for leg_name in leg_names:
                                if leg_name in marker_name:
                                    marker_color = colors.get(leg_name)
                                    break
                        if marker_color is None:
                            marker_color = colors.get(label_name, 'gray')
                                
                        marker_style = markers.get(label_name, 'o')
                        
                        # Use pre-scaled coordinates
                        x, y = marker_vals[idx, 0], marker_vals[idx, 1]
                        
                        # Plot using exact marker style as provided
                        face_color = "none" if marker_style == 'o' else marker_color
                        ax.plot(
                            x, y,
                            marker_style,
                            markersize=markersize,
                            markerfacecolor=face_color,  # Always use the color
                            markeredgecolor=marker_color,
                            alpha=0.7 if marker_vals[idx, 2] < likelihood_thresh else 1.0
                        )
                        
                        # Plot uncertainty if available
                        if marker_vals.shape[1] > 3:
                            # Use pre-scaled error bars
                            xerr, yerr = marker_vals[idx, 3], marker_vals[idx, 4]
                            
                            # Horizontal error bar
                            ax.plot([x - xerr, x + xerr], [y, y],
                                  '-', color=marker_color, linewidth=linewidth_error)
                            # Vertical error bar
                            ax.plot([x, x], [y - yerr, y + yerr],
                                  '-', color=marker_color, linewidth=linewidth_error)    
                    
                    # Plot skeleton
                    if skeleton is not None:
                        for s in skeleton:
                            if (keypoints_to_plot is not None and 
                                (s[0] not in keypoints_to_plot or s[1] not in keypoints_to_plot)):
                                continue
                                
                            if (point_dict[s[0]][idx, 2] < likelihood_thresh or 
                                point_dict[s[1]][idx, 2] < likelihood_thresh):
                                continue
                            
                            # Use pre-scaled coordinates
                            x0, y0 = point_dict[s[0]][idx, 0], point_dict[s[0]][idx, 1]
                            x1, y1 = point_dict[s[1]][idx, 0], point_dict[s[1]][idx, 1]
                            
                            # Determine line color based on leg names
                            line_color = None
                            if leg_names is not None:
                                # Check for leg identifiers in both keypoint names
                                for leg_name in leg_names:
                                    if leg_name in s[0] or leg_name in s[1]:
                                        line_color = colors.get(leg_name)
                                        break
                            
                            # Fall back to model color if no leg color is found
                            if line_color is None:
                                line_color = colors.get(label_name, 'gray')
                                
                            ax.plot([x0, x1], [y0, y1], '-', color=line_color)

            if idx_view == 0 and labels is not None:
                for p, label_name in enumerate(labels):
                    label_color = colors.get(label_name, 'red') # will probably change this to gray because now we don't have the colors of each model 
                    marker_style = markers.get(label_name, 'o')
                    ax.text(0.04, 0.04 + p * 0.05, f'{label_name} ({marker_style})',
                           color=label_color, **txt_kwargs)
                    
                           
            # Add frame number to first view
            if idx_view == 0:
                ax.text(0.02, 0.98, f'frame {idx}', **txt_fr_kwargs)
                # Add model name below frame number
                model_txt_kwargs = txt_fr_kwargs.copy()
                model_txt_kwargs['verticalalignment'] = 'top'
                ax.text(0.02, 0.90, f'model: {model_type}', **model_txt_kwargs)

            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_xlim([0, resize_dims[0]])
            ax.set_ylim([resize_dims[1], 0])
        
        # Save frame
        save_path = os.path.join(tmp_dir, f'frame_{i:06d}.jpeg')
        # save_path = os.path.join(tmp_dir, f'frame_{i:06d}.pdf')
        try:
            plt.savefig(save_path, dpi=300)
            frames_saved += 1
            if i < 5:
                print(f"Saved frame: {save_path}")
        except Exception as e:
            print(f"Failed to save frame {i}: {str(e)}")
    
    print(f"Total frames saved: {frames_saved}")
    
    # Save video if frames were generated
    if frames_saved > 0:
        try:
            save_video(save_file, tmp_dir, framerate=framerate)
            print(f"Successfully saved video to: {save_file}")
        except Exception as e:
            raise RuntimeError(f"Failed to save video: {str(e)}")
    else:
        raise RuntimeError("No frames were saved successfully")
        
    #Cleanup - tpmzzz folder to delete
    try:
        shutil.rmtree(tmp_dir)
        print("Cleaned up temporary directory")
    except Exception as e:
        print(f"Warning: Failed to clean up temporary directory: {str(e)}")