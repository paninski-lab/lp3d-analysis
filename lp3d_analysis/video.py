"""Functions for video handling."""

import os
import shutil
import subprocess

import cv2
import numpy as np


def save_video(save_file, tmp_dir, framerate=20, frame_pattern='frame_%06d.jpeg'):
    """

    Parameters
    ----------
    save_file : str
        absolute path of filename (including extension)
    tmp_dir : str
        temporary directory that stores frames of video; this directory will be deleted
    framerate : float, optional
        framerate of final video
    frame_pattern : str, optional
        string pattern used for naming frames in tmp_dir

    """

    if os.path.exists(save_file):
        os.remove(save_file)

    # make mp4 from images using ffmpeg
    call_str = \
        'ffmpeg -r %f -i %s -c:v libx264 -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" %s' % (
            framerate, os.path.join(tmp_dir, 'frame_%06d.jpeg'), save_file)
    print(call_str)
    subprocess.run(['/bin/bash', '-c', call_str], check=True)

    # delete tmp directory
    shutil.rmtree(tmp_dir)


# a class that allows to pull up the frams from a video and the indexes of the frames
# If load a video into that class, I want indexes 0-100, it will load the frames and return a numpy array 
def get_frames_from_idxs(cap, idxs, resize=None, flip=False):
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

    Returns
    -------
    np.ndarray
        returned frames of shape shape (n_frames, n_channels, ypix, xpix)

    """
    is_contiguous = np.sum(np.diff(idxs)) == (len(idxs) - 1)
    n_frames = len(idxs)
    for fr, i in enumerate(idxs):
        if fr == 0 or not is_contiguous:
            cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            if fr == 0:
                if resize is None:
                    height, width, _ = frame.shape
                else:
                    width, height = resize
                frames = np.zeros((n_frames, 1, height, width), dtype='uint8')
            if resize is not None:
                frame = cv2.resize(frame, resize)
            if flip:
                frame = cv2.flip(frame, 1)
            frames[fr, 0, :, :] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            print(
                'warning! reached end of video; returning blank frames for remainder of ' +
                'requested indices')
            break
    return frames


# the errors are ensemble variance of the lighting 
# pink is low dimensional error - kind of a projection 


