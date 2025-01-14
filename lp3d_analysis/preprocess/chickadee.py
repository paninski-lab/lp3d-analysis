# from pathlib import Path
import os

from tqdm import tqdm
import cv2
import imageio.v3 as iio
import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd

import lp3d_analysis.preprocess.bundle_adjust as pySBA


def formatData(data3D: np.ndarray, nParts: int = 18) -> tuple:
    # From Selmaan
    nFrames = data3D.shape[0]
    data3D = np.reshape(data3D, (nFrames, nParts, 3))
    return (data3D, nFrames, nParts)


def projectData(matfile):
    # From Selmaan
    camParams = pySBA.convertParams(matfile['camParams'])
    (pt3d, nFrames, nParts) = formatData(matfile['data_3D'])
    sba = pySBA.PySBA(camParams, np.NaN, np.NaN, np.NaN, np.NaN) #points_2d[:, :2], camera_ind, point_2dind3d, points_2d[:, 2])
    nCams = camParams.shape[0]
    allLabels = np.full((nFrames, nCams, nParts, 2), np.NaN)
    allCamScales = np.full((nFrames,nCams), np.NaN)
    for nCam in range(nCams):
        rVec = camParams[nCam][:3].reshape((1,3))
        tVec = camParams[nCam][3:6]
        for nPart in range(nParts):
            allLabels[:, nCam, nPart, :] = sba.project(pt3d[:,nPart], np.tile(camParams[nCam],(nFrames,1)))
        pt3d_centroid = np.mean(pt3d,axis=1) # average over parts
        pt3d_centroid = sba.rotate(pt3d_centroid, np.tile(rVec,(nFrames,1))) # rotate to camera coordinates
        camDist = pt3d_centroid[:,2] + tVec[2] # get z-axis distance ie along optical axis
        camScale = camParams[nCam][6] / camDist # convert to focal length divided by distance
        allCamScales[:,nCam] = camScale

    return allLabels, allCamScales


def find_best_frame_indices(frame_arrays: list, video_files: list):
    """Find the most closely matching frames from synchronized video files.

    Parameters
    ----------
    frame_arrays (list of numpy arrays): Each element is a 4D array (height, width, 1, n_frames).
    video_files (list of str): List of six video file paths, in the order corresponding to frame_arrays.

    Returns
    -------
    list of int: List of best matching frame indices for each frame in theseImages.
    
    """

    n_cams = len(frame_arrays)
    assert len(video_files) == n_cams, "Mismatch between videos and theseImages"

    n_frames = frame_arrays[0].shape[3]
    print(frame_arrays[0].shape)

    # Initialize the error array: (n_total_frames, n_frames, n_cams)
    video_caps = [cv2.VideoCapture(file) for file in video_files]
    n_total_frame_count = int(video_caps[0].get(cv2.CAP_PROP_FRAME_COUNT))

    error_array = np.zeros((n_total_frame_count, n_frames, n_cams), dtype=np.uint32)

    # Read and process all video frames once
    for c, cap in enumerate(video_caps):
        print(f"Processing video {c + 1}/{n_cams}")
        frames_labeled = np.ascontiguousarray(frame_arrays[c][:, :, 0, :].transpose(2, 0, 1))
        # frames_labeled shape is now (n_frames, height, width)
        for idx in tqdm(range(n_total_frame_count), desc=f"Reading frames (Cam {c + 1})"):

            ret, frame = cap.read()
            if not ret:
                break
            frame_gray = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # Calculate pixel errors for all frames in frames_labeled
            # this is actually faster than the vectorized version
            for n in range(n_frames):
                # THIS APPROACH IS BAD!
                # 1. abs(a - b) when a, b are uint8 has over/underflow issues
                # 2. sum() is also uint8, will have overfloat issues
                # could convert arrays to float16 first but this is very inefficient
                # error = np.abs(frames_labeled[n, :, :] - frame_gray).sum()
                # THIS APPROACH IS GOOD!
                # handles under/overflow properly and is super efficient
                error = (
                        np.maximum(frames_labeled[n, :, :], frame_gray)
                        - np.minimum(frames_labeled[n, :, :], frame_gray)
                ).sum(dtype=np.uint32)
                error_array[idx, n, c] = error

    # Release all video captures
    for cap in video_caps:
        cap.release()

    # Find the best indices
    best_indices = []
    for n in range(n_frames):
        best_index = np.argmin(error_array[:, n, :].sum(dtype=np.uint64, axis=-1))
        best_indices.append(best_index)

    return best_indices, error_array


def plot_frame_matches(
    frame_arrays,
    video_files,
    best_indices,
    labels=None,
    scales=None,
    w3d=1,
    save_dir=None,
):
    """Plot the frame matches for visual inspection.

    Parameters
    ----------
    theseImages (list of numpy arrays): Each element is a 4D array (height, width, 1, n_frames).
    video_files (list of str): List of six video file paths, in the order corresponding to theseImages.
    best_indices (list of int): Best matching frame indices for each frame in theseImages.
    labels (numpy array, optional): Shape (n_frames, n_views, n_keypoints, 2), keypoint coordinates.
    scales (numpy array, optional): Shape (n_frames, n_views), scale factors for centroids.
    w3d (float, optional): Scale multiplier for centroids (default: 10).
    save_dir (str): location of saved plots

    """
    n_cams = len(frame_arrays)
    assert len(frame_arrays) == n_cams, f"Expecting exactly {n_cams} elements in frame_arrays."
    assert len(video_files) == n_cams, f"Expecting exactly {n_cams} video files."

    n_frames = frame_arrays[0].shape[3]
    cap_list = [cv2.VideoCapture(file) for file in video_files]  # Open video files

    for n, best_index in enumerate(best_indices):
        fig, axes = plt.subplots(3, n_cams, figsize=(3 * n_cams, 9))
        for c, (img, cap) in enumerate(zip(frame_arrays, cap_list)):
            # Plot array
            axes[0, c].imshow(img[:, :, 0, n], cmap='gray', vmin=0, vmax=255)
            axes[0, c].set_title(f"Array {c}")
            axes[0, c].axis('off')

            # Plot video frame
            # OPENCV NOT PRECISE ENOUGH!
            # cap.set(cv2.CAP_PROP_POS_FRAMES, best_index)
            # ret, frame = cap.read()
            # IMAGEIO WORKS!
            frame = iio.imread(video_files[c], index=best_index)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            axes[1, c].imshow(frame_gray, cmap='gray', vmin=0, vmax=255)
            axes[1, c].set_title(f"Video Frame {c}")
            axes[1, c].axis('off')

            # Plot difference
            diff = np.abs(img[:, :, 0, n].astype(np.float64) - frame_gray.astype(np.float64))
            axes[2, c].imshow(diff, cmap='gray', vmin=0, vmax=255)
            axes[2, c].set_title(f"Difference {c}")
            axes[2, c].axis('off')

            # Optional: Plot zoomed box around centroid
            if labels is not None and scales is not None:
                centroid = np.nanmean(labels[n, c], axis=0)
                if not any(np.isnan(centroid)):
                    scale = w3d * scales[n, c]
                    axes[0, c].set_xlim(centroid[0] - scale, centroid[0] + scale)
                    axes[0, c].set_ylim(centroid[1] + scale, centroid[1] - scale)
                    axes[1, c].set_xlim(centroid[0] - scale, centroid[0] + scale)
                    axes[1, c].set_ylim(centroid[1] + scale, centroid[1] - scale)
                    axes[2, c].set_xlim(centroid[0] - scale, centroid[0] + scale)
                    axes[2, c].set_ylim(centroid[1] + scale, centroid[1] - scale)

        plt.tight_layout()

        # Save or display the figure
        if save_dir:
            save_path = os.path.join(save_dir, f"img{best_index:06d}.png")
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(save_path)
            plt.close(fig)
        else:
            plt.show()

    for cap in cap_list:
        cap.release()
