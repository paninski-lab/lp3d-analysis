from pathlib import Path
import shutil

from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lp3d_analysis.video import export_frames, get_frames_from_idxs


def insert_after_second_space(original_string, to_insert):
    """Helper function to manipulate filenames."""
    # Split the string into parts by spaces, remove empty strings
    parts = [part for part in original_string.split(" ") if part]
    # Ensure there are enough parts to insert after the second space
    if len(parts) > 2:
        # Reconstruct the string with the inserted value
        return " ".join(parts[:2] + [to_insert] + parts[2:])
    else:
        # If the string has fewer than two spaces, return it unchanged
        raise ValueError("String must contain at least two spaces to insert after the second space.")


base_dir = Path('/media/mattw/multiview/raw/fly-anipose/fly-anipose/fly-testing')
save_dir = Path('/media/mattw/multiview/datasets/fly-anipose')

camera_names = ['A', 'B', 'C', 'D', 'E', 'F']
frames_per_video = 25
InD_animals = ['Fly 1_0', 'Fly 2_0', 'Fly 3_0']
OOD_animals = ['Fly 4_0', 'Fly 5_0']

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

            video_name_w_cam = insert_after_second_space(video_name, f'Cam-{camera}')

            # load original predictions
            df_og = pd.read_hdf(pose_2d_dir / f'{video_name_w_cam}.h5')
            # predictions from multiple networks(?), only keep one
            df_og = df_og.loc[:,
                    df_og.columns.get_level_values('coords').isin(['x', 'y', 'likelihood'])]
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
                format="png",
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
    cam: pd.read_csv(save_dir / f"CollectedData_{cam}.csv", index_col=0, header=[0, 1, 2])
    for cam in camera_names
}

# Define colors for body parts
colors = {
    'L1': 'darkred', 'R1': 'red',
    'L2': 'darkgreen', 'R2': 'limegreen',
    'L3': 'darkblue', 'R3': 'dodgerblue',
}

# Define the body part groups
body_part_groups = ['L1', 'L2', 'L3', 'R1', 'R2', 'R3']

# Loop through the rows
for index in csv_files['A'].index:  # All files have the same indices
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5))
    axes = axes.flatten()

    # Loop through cameras
    for i, cam in enumerate(camera_names):
        index_ = index.replace('Cam-A', f'Cam-{cam}')
        # Get the corresponding row
        row = csv_files[cam].loc[index_]

        # Extract x, y coordinates (alternating columns)
        x_coords = row.iloc[0::2]
        y_coords = row.iloc[1::2]

        # Get the original column names
        columns = csv_files[cam].columns[0::2]  # Get x-coordinates' column names

        # Load the corresponding image
        image_path = save_dir / index_  # index is the relative path to the image
        image = cv2.imread(str(image_path))

        # Check if the image loaded successfully
        if image is None:
            print(f"Error: Could not load image {image_path}")
            continue

        # Convert BGR to RGB for plotting
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Plot the image
        axes[i].imshow(image)
        # axes[i].scatter(x_coords, y_coords, color='red', s=10)  # Plot keypoints
        axes[i].axis('off')  # Turn off axes

        # Loop through body part groups to plot points and connect them
        for group in body_part_groups:
            # Filter columns for the current group
            group_columns = [col for col in columns if
                             group in col[1]]  # Adjust to match the group name in the header
            if not group_columns:
                continue

            # Get indices for the current group
            group_indices = [columns.get_loc(col) for col in group_columns]

            # Extract x and y for the current group
            group_x = x_coords.iloc[group_indices].values
            group_y = y_coords.iloc[group_indices].values

            # Plot connections
            axes[i].plot(group_x, group_y, color=colors[group], linewidth=1.5, alpha=0.8)

            # Plot points
            axes[i].scatter(group_x, group_y, color=colors[group], s=20)

        # Add the camera name in the top-left corner
        axes[i].text(
            10, 10, f'Cam-{cam}',  # Position: (5, 5) pixels from the top-left
            color='white', fontsize=12, weight='bold',
            ha='left', va='top',
            bbox=dict(facecolor='black', alpha=0.8)  # , boxstyle='round,pad=0.3')
        )

    # Remove all possible whitespace
    fig.subplots_adjust(left=0, right=1, top=0.98, bottom=0, wspace=0, hspace=0)

    # Add a suptitle with the row index
    fig.suptitle(index.replace('Cam-A', ''), fontsize=16)

    plt.tight_layout()

    save_file = save_dir / index.replace('labeled-data', 'labeled-data-check').replace('Cam-A', '')
    save_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_file, bbox_inches='tight', dpi=300)
    plt.close(fig)
