"""Initial script for preprocessing the rat7m dataset.

Data from the DANNCE paper is located at https://figshare.com/collections/Rat_7M/5295370/3.

An initial step to concatenate all video clips can be found in scripts/preprocess/rat7m_step0.py.

There are several main parts of this script:

After running this script there is still some manual curation that must take place.

For the uncropped frames:
1. split data into InD and OOD (scripts/preprocess/split_data.py)
2. create video snippets for each OOD frame (scripts/preprocess/video_snippets_eks.py)
3. create unlabeled videos for unsupervised losses (scripts/preprocess/video_snippets_unlabeled.py)

For the cropped frames, the InD/OOD split can be performed as above.
For the video snippets, it is best to train a detector model on the uncropped frames and then use
that model to create cropped videos (scripts/preprocess/video_snippets_unlabeled_cropped.py)

"""
from pathlib import Path

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

from lp3d_analysis import dataset_info
from lp3d_analysis.preprocess import plot_labeled_frames
from lp3d_analysis.preprocess.bundle_adjust import convertParams, project_3D_to_2D
from lp3d_analysis.video import get_frames_from_idxs, export_frames, export_cropped_frames


def skeleton_length_pass(poses, keypoint0, keypoint1, min_val, max_val,
                         keypoint_names):
    """
    Filter poses based on distance between two keypoints.

    Parameters:
    -----------
    poses : np.ndarray
        Array of shape [n_timepoints, n_keypoints * 3] containing 3D pose data
    keypoint0 : str
        Name of first keypoint
    keypoint1 : str
        Name of second keypoint
    min_val : float
        Minimum allowed distance between keypoints
    max_val : float
        Maximum allowed distance between keypoints
    keypoint_names : list
        List of keypoint names corresponding to indices in poses array

    Returns:
    --------
    np.ndarray
        Boolean array of shape [n_timepoints] indicating which poses pass the filter
    """

    # Get indices of the keypoints
    try:
        kp0_idx = keypoint_names.index(keypoint0)
        kp1_idx = keypoint_names.index(keypoint1)
    except ValueError as e:
        raise ValueError(
            f"Keypoint name not found in keypoint_names list: {e}")

    n_timepoints, n_coords = poses.shape
    n_keypoints = n_coords // 3

    # Validate keypoint indices
    if kp0_idx >= n_keypoints or kp1_idx >= n_keypoints:
        raise ValueError(
            f"Keypoint indices ({kp0_idx}, {kp1_idx}) exceed number of keypoints ({n_keypoints})")

    # Reshape poses to [n_timepoints, n_keypoints, 3] for easier indexing
    poses_reshaped = poses.reshape(n_timepoints, n_keypoints, 3)

    # Extract 3D positions for both keypoints
    kp0_pos = poses_reshaped[:, kp0_idx, :]  # [n_timepoints, 3]
    kp1_pos = poses_reshaped[:, kp1_idx, :]  # [n_timepoints, 3]

    # Calculate Euclidean distances
    distances = np.linalg.norm(kp1_pos - kp0_pos, axis=1)  # [n_timepoints]

    # Create boolean mask for poses that pass the filter
    pass_filter = (distances >= min_val) & (distances <= max_val)

    return pass_filter


# path info
base_dir = Path('/media/mattw/multiview-data/_raw/rat7m')
data_dir_proc = Path('/media/mattw/multiview-data/rat7m')

# set flags
select_idxs = False

save_frames_full = True  # save out full matched frames and context
save_frames_full_labeled = True  # quality control

save_frames_cropped = True  # save out cropped matched frames and context
save_frames_cropped_labeled = True  # quality control

# collect dataset info
data_info = dataset_info['rat7m']
camera_names = data_info['cam_names']
keypoints = data_info['keypoints_raw']
InD_animals = data_info['InD_animals']
OOD_animals = data_info['OOD_animals']
skeleton = data_info['skeleton_raw']
skeleton_colors = data_info['skeleton_colors_raw']
w3d = data_info['w3d']
crop_size = data_info['crop_size']

frames_per_session = 100

labels_file = 'labels.pqt'
bboxes_file = 'bboxes.csv'

# for diagnostics
skeleton_cross_body = np.array([
    [8, 9],  # HipL, HipR
    [10, 14],  # ElbowL, ElbowR
    [12, 13],  # ShoulderL, ShoulderR
    [16, 17],  # KneeL, KneeR
    [18, 19],  # ShinL, ShinR
])
skeleton_new = np.concatenate([skeleton, skeleton_cross_body], axis=0)

# loop over sessions
for animal in OOD_animals[:1]:  # (InD_animals + OOD_animals):

    print(f'Extracting poses for session {animal}')
    animal_dir = base_dir / animal

    #     if animal not in ['s1-d1']:
    #         continue

    # -----------------------------------------------------------
    # load data
    # -----------------------------------------------------------
    # 3d mocap points
    mocap = sio.loadmat(
        base_dir / 'mocap' / f'mocap-{animal}.mat')  # get cam params from this

    training_data = False
    try:
        mocap_tr = sio.loadmat(
            base_dir / 'rat7m_training' / f'{animal.replace("-", "_")}_rat7m_training.mat')
        points_3d = [mocap_tr['mocap'][0][0][kp] for kp in range(20)]
        training_data = True
    except:
        points_3d = [mocap['mocap'][0][0][kp] for kp in range(20)]

    points_3d = np.stack(points_3d, axis=1)

    # camera params
    camParams = []
    frame_mapping = []
    for c in range(len(camera_names)):
        # cam data not ordered as you would expect
        if animal not in ['s5-d1', 's5-d2']:
            if c in [0, 1, 5]:
                c_ = c
            elif c == 2:
                c_ = 4
            elif c == 3:
                c_ = 2
            elif c == 4:
                c_ = 3
        else:
            c_ = c
        i_mat = mocap['cameras'][0][0][c_][0][0][1]  # intrinsic matrix
        r_mat = mocap['cameras'][0][0][c_][0][0][2]  # rotation matrix
        t_vec = mocap['cameras'][0][0][c_][0][0][3]  # translation vector
        t_dtr = mocap['cameras'][0][0][c_][0][0][4]  # tangential distortion
        r_dtr = mocap['cameras'][0][0][c_][0][0][5]  # radial distortion
        #     print(f'intrinsic matrix shape: {i_mat.shape}')
        #     print(f'rotation matrix shape: {r_mat.shape}')
        #     print(f'translation vector shape: {t_vec.shape}')
        #     print(f'tangential distortion shape: {t_dtr.shape}')
        #     print(f'radial distortion shape: {r_dtr.shape}')
        camParams_ = {'K': i_mat, 'r': r_mat, 't': t_vec[0],
                      'RDistort': r_dtr[0]}
        camParams.append((camParams_,))

        if training_data:
            # training frames
            frame_mapping.append(mocap_tr['frames'][0][0][c].flatten())
        else:
            # full vids
            frame_mapping.append(mocap['cameras'][0][0][c_][0][0][0][0])

    # -----------------------------------------------------------
    # run k-means on 3D poses
    # -----------------------------------------------------------
    idxs_file = data_dir_proc / 'labeled-data' / f'{animal}_idxs.npy'
    if not idxs_file.exists() or select_idxs:
        poses = points_3d.reshape((points_3d.shape[0], -1))
        idxs_good = np.isnan(poses).sum(axis=1) == 0
        # filter out potentially problematic points using skeleton distances
        filters = [
            {'keypoint0': 'ElbowR', 'keypoint1': 'ElbowL', 'min_val': 40,
             'max_val': 60},
            {'keypoint0': 'KneeR', 'keypoint1': 'ShinR', 'min_val': 10,
             'max_val': 1000},
            {'keypoint0': 'KneeL', 'keypoint1': 'ShinL', 'min_val': 10,
             'max_val': 1000},
        ]
        if training_data:
            filters.append({'keypoint0': 'ShoulderL', 'keypoint1': 'ShoulderR',
                            'min_val': 30, 'max_val': 1000})
        for filt in filters:
            idxs_good_filt = skeleton_length_pass(
                poses=poses, keypoint_names=keypoints, **filt,
            )
            idxs_good = idxs_good & idxs_good_filt
        idxs_good = np.where(idxs_good)[0]
        poses = poses[idxs_good]
        if len(idxs_good) <= frames_per_session:
            idxs_prototypes = idxs_good
        else:
            def find_prototypes(seed):
                np.random.seed(seed)
                kmeans = KMeans(
                    n_clusters=min(len(idxs_good), frames_per_session))
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
                # now index into good frames to get overall indices
                idxs_prototypes_final = np.sort(idxs_good[idxs_prototypes_])
                return idxs_prototypes_final


            rng_seed = 0
            idxs_prototypes = find_prototypes(seed=rng_seed)
            while np.any(np.diff(frame_mapping[0][idxs_prototypes]) <= 2):
                rng_seed += 1
                print(
                    f'context frames overlapping with labeled frames; trying seed = {rng_seed}')
                idxs_prototypes = find_prototypes(seed=rng_seed)

        # don't want context frames overlapping with labeled frames when cropping later
        idxs_final_mocap = idxs_prototypes
        assert not np.any(np.diff(frame_mapping[0][idxs_final_mocap]) <= 2)

        #         limb_lengths, limb_names = analyze_limb_lengths(poses, skeleton_new, keypoints)
        #         continue

        idxs_file.parent.mkdir(exist_ok=True, parents=True)
        np.save(idxs_file, idxs_final_mocap)
    else:
        idxs_final_mocap = np.load(idxs_file)

    idxs_final_save = frame_mapping[0][idxs_final_mocap].flatten()

    # -----------------------------------------------------------
    # convert 3d poses to 2d poses
    # -----------------------------------------------------------
    camParams = convertParams(camParams)
    points_2d, scales = project_3D_to_2D(points_3d, camParams)

    # --------------------------------------------------
    # output frames and labels for full/cropped datasets
    # --------------------------------------------------
    if save_frames_full:
        print(f'EXPORTING {len(idxs_final_mocap)} FULL FRAMES')
        for c, camera in enumerate(camera_names):
            print(f'Processing cam {c + 1} / {len(camera_names)}')
            src_video_file = base_dir / 'videos' / f'{animal}_{camera}.mp4'
            # frame indices corresponding to mocap data
            idxs_final_frame = frame_mapping[c][idxs_final_mocap]
            # create dataframe
            data = points_2d[idxs_final_mocap, c]
            pdindex = pd.MultiIndex.from_product(
                [['mocap'], keypoints, ['x', 'y']],
                names=['scorer', 'bodyparts', 'coords'],
            )
            row_index = [
                f'labeled-data/{animal}_{camera}/img{str(i).zfill(8)}.png'
                for i in idxs_final_save
            ]
            df_2d = pd.DataFrame(
                data.reshape(data.shape[0], -1), columns=pdindex,
                index=row_index,
            )
            # extract frames and their context
            save_dir = data_dir_proc / 'labeled-data' / f'{animal}_{camera}'
            save_dir.mkdir(parents=True, exist_ok=True)
            # frame_idxs = np.array([int(path.split('/')[-1][3:-4]) for path in df_2d.index])
            export_frames(
                video_file=src_video_file,
                save_dir=save_dir,
                frame_idxs=idxs_final_frame,
                frame_idxs_save=idxs_final_save,  # force naming
                format='png',
                n_digits=8,
                context_frames=2,
                grayscale=False,
                #                 reader='imageio',
            )
            # save out labels
            df_2d.to_parquet(save_dir / labels_file)

    if save_frames_full_labeled:
        print(
            f'EXPORTING {len(idxs_final_mocap)} FULL FRAMES (LABELED CHECKS)')
        csv_files = {
            cam_name: pd.read_parquet(
                Path(
                    data_dir_proc) / 'labeled-data' / f'{animal}_{cam_name}' / labels_file
            )
            for cam_name in camera_names
        }
        s = 0.5
        linewidth = 0.5
        txt_offset = 50
        height = 6
        plot_labeled_frames(
            csv_files=csv_files, data_dir=data_dir_proc,
            skeleton=skeleton, skeleton_colors=skeleton_colors,
            s=s, linewidth=linewidth, txt_offset=txt_offset, height=height,
        )

    if save_frames_cropped:
        print(f'EXPORTING {len(idxs_final_mocap)} CROPPED FRAMES')
        idxs_final_frame_save = None
        for c, cam_name in enumerate(camera_names):
            print(f'Processing cam {c + 1} / {len(camera_names)}')
            save_dir_full = Path(
                data_dir_proc) / 'labeled-data' / f'{animal}_{cam_name}'
            save_dir_crop = Path(
                data_dir_proc) / 'labeled-data-cropped' / f'{animal}_{cam_name}'
            bboxes = export_cropped_frames(
                save_dir_full=save_dir_full,
                save_dir_crop=save_dir_crop,
                frame_idxs=idxs_final_save,
                labels=points_2d[idxs_final_mocap, c],
                scales=scales[idxs_final_mocap, c],
                w3d=w3d,
                resize=crop_size,
                context_frames=2,
                # nan_centroid_idxs=nan_centroid_idxs,
            )
            # transform labels
            labels_cropped = points_2d[idxs_final_mocap, c].copy()
            labels_cropped[:, :, 0] -= bboxes[:, 0, None]
            labels_cropped[:, :, 0] = labels_cropped[:, :, 0] / bboxes[
                :, 3, None] * crop_size[0]
            labels_cropped[:, :, 1] -= bboxes[:, 1, None]
            labels_cropped[:, :, 1] = labels_cropped[:, :, 1] / bboxes[
                :, 2, None] * crop_size[1]
            # save out labels
            df_index = [
                f'labeled-data-cropped/{animal}_{cam_name}/img{str(i).zfill(8)}.png'
                for i
                in idxs_final_save]
            this_df_crop = pd.DataFrame(
                labels_cropped.reshape(labels_cropped.shape[0], -1),
                columns=pdindex,
                index=df_index,
            )
            this_df_crop.to_parquet(save_dir_crop / labels_file)
            # save out bounding boxes
            bbox_df = pd.DataFrame(bboxes, columns=['x', 'y', 'h', 'w'],
                                   index=df_index)
            bbox_df.to_csv(save_dir_crop / bboxes_file)

    if save_frames_cropped_labeled:
        print(
            f'EXPORTING {len(idxs_final_mocap)} CROPPED FRAMES (LABELED CHECKS)')
        csv_files = {
            cam_name: pd.read_parquet(
                Path(
                    data_dir_proc) / 'labeled-data-cropped' / f'{animal}_{cam_name}' / labels_file
            )
            for cam_name in camera_names
        }
        s = 1
        linewidth = 1
        txt_offset = 5
        height = 8.5
        plot_labeled_frames(
            csv_files=csv_files, data_dir=data_dir_proc,
            skeleton=skeleton, skeleton_colors=skeleton_colors,
            s=s, linewidth=linewidth, txt_offset=txt_offset, height=height,
        )


def __remove_bad_frames():
    # from manual inspection
    bad_frames = [
        'labeled-data/s1-d1/img00018740.png',
        'labeled-data/s1-d1/img00019699.png',
        'labeled-data/s1-d1/img00043246.png',
        'labeled-data/s1-d1/img00043500.png',
        'labeled-data/s1-d1/img00046632.png',
        'labeled-data/s1-d1/img00047175.png',
        'labeled-data/s1-d1/img00047560.png',
        'labeled-data/s1-d1/img00051094.png',
        'labeled-data/s1-d1/img00051665.png',
        'labeled-data/s2-d1/img00024416.png',
        'labeled-data/s2-d1/img00031097.png',
        'labeled-data/s2-d1/img00041687.png',
        'labeled-data/s2-d2/img00024978.png',
        'labeled-data/s2-d2/img00050054.png',
        'labeled-data/s2-d2/img00052150.png',
        'labeled-data/s2-d2/img00052603.png',
        'labeled-data/s3-d1/img00050421.png',
        'labeled-data/s3-d1/img00050703.png',
        'labeled-data/s3-d1/img00052111.png',
        'labeled-data/s3-d1/img00053009.png',
        'labeled-data/s4-d1/img00013726.png',
        'labeled-data/s4-d1/img00013805.png',
        'labeled-data/s4-d1/img00014096.png',
        'labeled-data/s4-d1/img00015007.png',
        'labeled-data/s4-d1/img00015065.png',
        'labeled-data/s4-d1/img00015142.png',
        'labeled-data/s4-d1/img00015514.png',
        'labeled-data/s4-d1/img00018297.png',
        'labeled-data/s4-d1/img00018423.png',
        'labeled-data/s4-d1/img00018501.png',
        'labeled-data/s4-d1/img00019102.png',
        'labeled-data/s4-d1/img00019994.png',
        'labeled-data/s4-d1/img00020409.png',
        'labeled-data/s4-d1/img00020641.png',
        'labeled-data/s4-d1/img00020665.png',
        'labeled-data/s4-d1/img00020701.png',
        'labeled-data/s4-d1/img00020903.png',
        'labeled-data/s4-d1/img00024885.png',
        'labeled-data/s4-d1/img00028626.png',
        'labeled-data/s5-d1/img00047405.png',
        'labeled-data/s5-d1/img00096116.png',
        'labeled-data/s5-d1/img00103936.png',
        'labeled-data/s5-d1/img00106317.png',
        'labeled-data/s5-d1/img00108042.png',
        'labeled-data/s5-d1/img00109267.png',
        'labeled-data/s5-d1/img00113622.png',
        'labeled-data/s5-d1/img00115988.png',
        'labeled-data/s5-d1/img00121071.png',
        'labeled-data/s5-d1/img00131524.png',
        'labeled-data/s5-d1/img00152404.png',
        'labeled-data/s5-d1/img00154175.png',
        'labeled-data/s5-d1/img00166913.png',
        'labeled-data/s5-d1/img00167709.png',
        'labeled-data/s5-d1/img00176394.png',
        'labeled-data/s5-d1/img00192506.png',
        'labeled-data/s5-d2/img00011810.png',
        'labeled-data/s5-d2/img00027099.png',
        'labeled-data/s5-d2/img00071063.png',
        'labeled-data/s5-d2/img00080388.png',
        'labeled-data/s5-d2/img00090641.png',
        'labeled-data/s5-d2/img00102581.png',
        'labeled-data/s5-d2/img00171754.png',
        'labeled-data/s5-d2/img00174013.png',
        'labeled-data/s5-d2/img00178067.png',
        'labeled-data/s5-d2/img00198655.png',
        'labeled-data/s5-d2/img00209765.png',
        'labeled-data/s5-d2/img00210402.png',
    ]

    dataset = 'rat7m-crop'

    from pathlib import Path

    src_dir = Path('/media/mattw/multiview-data').joinpath(dataset)

    cam_names = dataset_info[dataset.replace('-crop', '')]['cam_names']

    # calibration files
    filename = 'calibrations.csv'
    for suff in ['', '_new']:
        filepath = src_dir.joinpath(filename.replace('.csv', f'{suff}.csv'))
        df = pd.read_csv(filepath, index_col=0, header=[0])
        df_new = df.drop(index=bad_frames,
                         errors='ignore')  # errors='ignore' won't fail if frames don't exist
        print(f'{filepath.name}: {df_new.shape[0]} rows')
        df_new.to_csv(filepath)
    print()

    # labels files
    filename = 'CollectedData.csv'
    for suff in ['', '_new']:
        for cam in cam_names:
            filepath = src_dir.joinpath(
                filename.replace('.csv', f'_{cam}{suff}.csv'))
            df = pd.read_csv(filepath, index_col=0, header=[0, 1, 2])
            bad_frames_cam = [
                '/'.join(parts[:1] + [parts[1] + f'_{cam}'] + parts[2:])
                for parts in [path.split('/') for path in bad_frames]
            ]
            df_new = df.drop(index=bad_frames_cam,
                             errors='ignore')  # errors='ignore' won't fail if frames don't exist
            print(f'{filepath.name}: {df_new.shape[0]} rows')
            df_new.to_csv(filepath)
    print()

    # bboxes files
    filename = 'bboxes.csv'
    for suff in ['', '_new']:
        for cam in cam_names:
            filepath = src_dir.joinpath(
                filename.replace('.csv', f'_{cam}{suff}.csv'))
            if filepath.exists():
                df = pd.read_csv(filepath, index_col=0, header=[0])
                bad_frames_cam = [
                    '/'.join(parts[:1] + [parts[1] + f'_{cam}'] + parts[2:])
                    for parts in [path.split('/') for path in bad_frames]
                ]
                df_new = df.drop(index=bad_frames_cam,
                                 errors='ignore')  # errors='ignore' won't fail if frames don't exist
                print(f'{filepath.name}: {df_new.shape[0]} rows')
                df_new.to_csv(filepath)
    print()
