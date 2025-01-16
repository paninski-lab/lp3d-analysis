"""Initial script for preprocessing the Anipose dataset.

Data from the Anipose paper is located at  https://doi.org/10.5061/dryad.nzs7h44s4.
This Lightning Pose dataset is constructed from the file "fly-anipose.zip".

NOTE: the Lightning Pose dataset does NOT contain hand-labeled frames!
Instead, it is composed of filtered Anipose predictions, so that it was possible to
extract labels across all cameras for a single instant in time, as well as extract
context frames.

In the following, "instance" refers to the pose/frames across all cameras at a given
point in time; "frame" or "2D pose" refers to a single camera view.

The Lightning Pose dataset was constructed following these steps:
For a subset of sessions in
"fly-anipose/fly-testing/{animal_id}/videos-raw-compressed":
1. select frames and labels
    a. remove any instance where average 3D reprojection error is >10 pixels
    b. on remaining instances run k-means on the 3D poses; keep 25 instances/session
    c. for the 25 instances, use the filtered 2D predictions (as opposed to orig preds)
    d. set any keypoint where 2D reprojection error >10 pixels to NaN
2. save out frames and labels
3. copy videos over to LP dataset; all videos in this dataset are very short, no need to shorten
4. verify the keypoint extraction by plotting the frames and labels (saved in labeled-data-check)

After running this script you will need to create video snippets for each labeled frame
(scripts/preprocess/video_snippets_eks.py)

"""

from pathlib import Path
import shutil

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

from lp3d_analysis import dataset_info
from lp3d_analysis.preprocess import plot_labeled_frames
from lp3d_analysis.video import export_frames, get_frames_from_idxs


def insert_after_second_space(original_string, to_insert):
    """Helper function to manipulate filenames."""
    # Split the string into parts by spaces, remove empty strings
    parts = [part for part in original_string.split(' ') if part]
    # Ensure there are enough parts to insert after the second space
    if len(parts) > 2:
        # Reconstruct the string with the inserted value
        return ' '.join(parts[:2] + [to_insert] + parts[2:])
    else:
        # If the string has fewer than two spaces, return it unchanged
        raise ValueError('String must contain at least two spaces to insert after second space.')


base_dir = Path('/media/mattw/multiview-data/_raw/fly-anipose/fly-anipose/fly-testing')
save_dir = Path('/media/mattw/multiview-data/fly-anipose')

data_info = dataset_info['fly-anipose']
camera_names = data_info['cam_names']
InD_animals = data_info['InD_animals']
OOD_animals = data_info['OOD_animals']
frames_per_video = 25

error_thresh_3d = 10.0  # pixels; cannot go lower than this without losing most frames
error_thresh_2d = 10.0  # pixels

# save out "labels"
labels_InD = {c: [] for c in camera_names}
labels_OOD = {c: [] for c in camera_names}

# loop over flies
for animal in (InD_animals + OOD_animals):

    animal_dir = base_dir / animal

    pose_2d_dir = animal_dir / 'pose-2d'
    pose_2d_filtered_dir = animal_dir / 'pose-2d-filtered'
    pose_2d_proj_dir = animal_dir / 'pose-2d-proj'
    pose_3d_dir = animal_dir / 'pose-3d'
    video_dir = animal_dir / 'videos-raw-compressed'

    videos = list(pose_3d_dir.glob('*.csv'))
    print(f'Found {len(videos)} videos for {animal}')

    errors_dict = {}

    # loop over videos using "pose-3d" directory
    for video in videos[:6]:

        video_name = video.stem
        print(video_name)

        # -----------------------------------------------------------
        # remove any frame where any 3D reprojection error is >10 pix
        # -----------------------------------------------------------
        # NOTE: THIS DOES NOT FILTER OUT ALL ERRORS
        # these errors are averages across all views so will not
        # highlight bad single view predictions
        df_3d = pd.read_csv(video)
        errors = df_3d.loc[:, df_3d.columns.str.contains('_error')].to_numpy()
        idxs_good = np.where(np.max(errors, axis=-1) < error_thresh_3d)[0]
        print(f'Found {len(idxs_good)} indices with low reprojection error')
        if len(idxs_good) == 0:
            continue

        # -----------------------------------------------------------
        # run k-means on remaining frames using 3D poses
        # -----------------------------------------------------------
        if len(idxs_good) <= frames_per_video:
            idxs_prototypes = idxs_good
        else:
            df_3d = pd.read_csv(video)
            mask_xyz = df_3d.columns.str.contains('_x') | df_3d.columns.str.contains(
                '_y') | df_3d.columns.str.contains('_z')
            poses = df_3d.loc[idxs_good, mask_xyz].to_numpy()
            np.random.seed(0)
            kmeans = KMeans(n_clusters=min(len(idxs_good), frames_per_video))
            kmeans.fit(poses)
            centers = kmeans.cluster_centers_
            # centers is initially shape (n_clusters, n_coords); reformat
            centers = centers.T[None, :]
            # find frame that is closest to each cluster center
            # poses is shape (n_frames, n_coords)
            # centers is shape (1, n_pcs, n_clusters)
            dists = np.linalg.norm(poses[:, :, None] - centers, axis=1)
            # dists is shape (n_frames, n_clusters)
            idxs_prototypes_ = np.argmin(dists, axis=0)
            # now index into good frames to get overall indices, add offset
            idxs_prototypes = np.sort(idxs_good[idxs_prototypes_])

        # -----------------------------------------------------------
        # select the predictions from pose-2d-filtered
        # -----------------------------------------------------------
        for camera in camera_names:

            video_name_w_cam = insert_after_second_space(video_name, camera)

            # load original predictions
            df_og = pd.read_hdf(pose_2d_dir / f'{video_name_w_cam}.h5')
            # predictions from multiple networks(?), only keep one
            df_og = df_og.loc[:, df_og.columns.get_level_values('coords').isin(['x', 'y', 'likelihood'])]
            # load reprojections
            df_rp = pd.read_hdf(pose_2d_proj_dir / f'{video_name_w_cam}.h5')
            assert df_og.shape == df_rp.shape
            # compute reprojection errors
            preds_og = df_og.to_numpy().reshape(df_og.shape[0], -1, 3)
            preds_rp = df_rp.to_numpy().reshape(df_rp.shape[0], -1, 3)
            error = np.sqrt(np.sum((preds_og[:, :, :2] - preds_rp[:, :, :2]) ** 2, axis=-1))
            # store in dataframe
            df_error = pd.DataFrame(
                np.repeat(error, repeats=3, axis=1),
                columns=df_og.columns,
                index=df_og.index,
            )
            # only keep the "good" indices
            df_error = df_error.loc[idxs_prototypes]
            errors_dict[camera] = df_error

            # load filtered predictions
            df_2d_filt = pd.read_hdf(pose_2d_filtered_dir / f'{video_name_w_cam}.h5')
            # only keep the "good" indices
            df_2d_filt = df_2d_filt.loc[idxs_prototypes]

            n_initial_keypoints = df_2d_filt.shape[0] * df_2d_filt.shape[1] / 3
            # set bad predictions (according to 2d filt dataframe) to NaN
            df_2d_filt[df_2d_filt == -1.0] = np.nan
            n_dropped = int(df_2d_filt.isna().sum().sum() / 3)
            # print(f'Setting {n_dropped} / {n_initial_keypoints} keypoints to NaN (filtered)')
            # set bad predictions (according to 2d reprojection error) to NaN
            df_2d_filt[df_error > error_thresh_2d] = np.nan
            n_dropped = int(df_2d_filt.isna().sum().sum() / 3)
            print(
                f'Setting {n_dropped} / {n_initial_keypoints} keypoints to NaN (filtered + reproj error)')
            # get rid of likelihoods since these are "ground truth"
            df_2d_filt = df_2d_filt.loc[:,
                         df_2d_filt.columns.get_level_values('coords').isin(['x', 'y'])]
            # replace scorer name with 'anipose'
            col_index = pd.MultiIndex.from_tuples(
                [('anipose', bp, coord) for (_, bp, coord) in df_2d_filt.columns],
                names=('scorer', 'bodyparts', 'coords')
            )
            df_2d_filt.columns = col_index
            # update row index
            df_2d_filt.index = [
                f'labeled-data/{video_name_w_cam.replace(" ", "_")}/img{str(i).zfill(8)}.png'
                for i in df_2d_filt.index
            ]
            # store predictions in csv file
            if animal in InD_animals:
                labels_InD[camera].append(df_2d_filt)
            else:
                labels_OOD[camera].append(df_2d_filt)

            # -----------------------------------------------------------
            # copy video
            # -----------------------------------------------------------
            src = animal_dir / 'videos-raw-compressed' / f'{video_name_w_cam}.mp4'
            vid_dir = 'videos' if animal in InD_animals else 'videos_new'
            dst = save_dir / vid_dir / f'{video_name_w_cam.replace(" ", "_")}.mp4'
            if not dst.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)

            # -----------------------------------------------------------
            # extract frames and their context
            # -----------------------------------------------------------
            frames_save_dir = save_dir / 'labeled-data' / dst.stem
            frames_save_dir.mkdir(parents=True, exist_ok=True)
            frame_idxs = np.array([int(path.split('/')[-1][3:-4]) for path in df_2d_filt.index])
            export_frames(
                video_file=dst,
                save_dir=frames_save_dir,
                frame_idxs=frame_idxs,
                format='png',
                n_digits=8,
                context_frames=2,
            )

# save out labeled data csv files
for camera in camera_names:
    if len(labels_InD[camera]) > 0:
        l_ind = pd.concat(labels_InD[camera])
        l_ind.to_csv(save_dir / f'CollectedData_{camera}.csv')
    if len(labels_OOD[camera]) > 0:
        l_ood = pd.concat(labels_OOD[camera])
        l_ood.to_csv(save_dir / f'CollectedData_{camera}_new.csv')

# ---------------------------------------------------------------------
# plot to verify
# ---------------------------------------------------------------------

# Load the CSV files into a dictionary
csv_files = {
    cam: pd.read_csv(save_dir / f'CollectedData_{cam}.csv', index_col=0, header=[0, 1, 2])
    for cam in camera_names
}

plot_labeled_frames(
    csv_files=csv_files,
    data_dir=str(save_dir),
    s=10.0,
    linewidth=1.5,
    txt_offset=10,
    height=6.5,
    skeleton=data_info['skeleton'],
    skeleton_colors=data_info['skeleton_colors'],
)
