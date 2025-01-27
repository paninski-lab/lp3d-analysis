"""Initial script for preprocessing the chickadee dataset.

The raw chickadee dataset was labeled in Label3D, which saves the 3D labels, camera parameters, and
frames. It does not, however, record the frame index for each labeled frame. Therefore, much of the
preprocessing work for this dataset involves scanning through the raw videos to find the matching
labeled frame.

There are several main parts of this script:
1. match labeled frames to video index
   (flag: overwrite_idxs)
2. verify this step by plotting the original frame, best matching frame, and their difference. The
   results will be stored in the raw data directory.
   (flag: plot_frames)
3. once the matching indices are verified, save out the matching frame and context frames. This
   step is quite time intensive because seeking in the avi files with opencv is not precise enough,
   so the imageio library is used. There is a lot of overhead to this operation, but it works.
   The frames will be saved in labeled-data
   Note 1: any frames where a match was not found is recorded in `bad_idxs`; these will be skipped
   when exporting frames.
   Note 2: some videos begin without a bird in the arena, and so I add some of these frames to the
   labeled dataset. These will be redundant for the full frames, but I take crops in different
   locations for the cropped dataset. This information is found in `add_blank_frames`
   (flag: save_frames_full)
4. verify this step by saving labeled frames in labeled-data-check to manually inspect
   (flag: save_frames_full_labeled)
5. export cropped labeled frames and their context; note the same bounding box is used for all
   frames in a given context window.
   (flag: save_frames_cropped)
6. verify this step by saving cropped labeled frames in labeled-data-cropped-check
   (flag: save_frames_cropped_labeled)

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
import os

import mat73
import numpy as np
import pandas as pd

from lp3d_analysis import dataset_info
from lp3d_analysis.preprocess import plot_labeled_frames
from lp3d_analysis.preprocess.chickadee import projectData, find_best_frame_indices, plot_frame_matches
from lp3d_analysis.video import get_frames_from_idxs, export_frames, export_cropped_frames


# path info
data_dir = '/media/mattw/multiview-data/_raw/chickadee/Label3D format frames'
video_dir = '/media/mattw/multiview-data/_raw/chickadee/Videos'
check_dir = '/media/mattw/multiview-data/_raw/chickadee/frame-matches'
data_dir_proc = '/media/mattw/multiview-data/chickadee'

# set flags
overwrite_idxs = False  # find indices of labeled frames; processing takes hours/video!
plot_matches = False  # quality control

save_frames_full = False            # save out full matched frames and context
save_frames_full_labeled = False    # quality control

save_frames_cropped = True          # save out cropped matched frames and context
save_frames_cropped_labeled = True  # quality control

# NOTES:
# - SLV151_200730_131948_Manual_relabel: need to rerun cropped frame sections with context=0 since there are
#   overlapping context frames; this messes up the bounding boxes
# - TRQ177_210630_132311: views 0, 3 have error in video metadata; you must:
#   - use view index 2 in find_best_frame_indices
#   - comment out the line frame_idxs = frame_idxs[frame_idxs < int(cap.get(cv2.CAP_PROP_FRAME_COUNT))] in export_frames

training_files = [
#     'CHC41_200705_105803_20200720_195918_Label3D_videos.mat',   # N labeled frames: 30, N video frames: 408849, estimated time: 5.688 hours
#     'CHC41_200709_110253_20200709_150243_Label3D_videos.mat',   # N labeled frames: 30, N video frames: 488825, estimated time: 5.688 hours
#     'CHC41_200709_110253_20200917_135958_Label3D_videos.mat',   # N labeled frames: 35, N video frames: 488825, estimated time: 6.636 hours
#     'CHC41_200709_110253_20200923_152544_Label3D_videos.mat',   # N labeled frames: 35, N video frames: 488825, estimated time: 6.636 hours -- DUPLICATE OF PREVIOUS
#     'CHC41_200723_110437_20200805_114218_Label3D_videos.mat',   # N labeled frames: 40, N video frames: 438697, estimated time: 7.584 hours
#     'EMR30_200714_104454_20200720_111555_Label3D_videos.mat',   # N labeled frames: 30, N video frames: 498958, estimated time: 5.688 hours
#     'EMR30_200720_110832_20200722_153703_Label3D_videos.mat',   # N labeled frames: 30, N video frames: 518805, estimated time: 5.688 hours
#     'IND88_201218_112623_20201223_152611_Label3D_videos.mat',   # N labeled frames: 92, N video frames: 586255, estimated time: 17.443 hours
#     'PRL43_200617_131904_20200630_170437_Label3D_videos.mat',   # N labeled frames: 25, N video frames: 200647, estimated time: 4.74 hours
#     'PRL43_200701_142147_20200709_092553_Label3D_videos.mat',   # N labeled frames: 30, N video frames: 415777, estimated time: 5.688 hours
#     'SLV151_200728_132004_20200729_185246_Label3D_videos.mat',  # N labeled frames: 30, N video frames: 429862, estimated time: 5.688 hours
#     'SLV151_200730_131948_20200807_122501_Label3D_videos.mat',  # N labeled frames: 60, N video frames: 474595, estimated time: 11.376 hours
#     'SLV151_200730_131948_Manual_relabel_videos.mat',           # N labeled frames: 30, N video frames: 474595, estimated time: 5.688 hours
#     'TRQ66_220728_132027_20220729_151716_Label3D_videos.mat',   # N labeled frames: 52, N video frames: 615118, estimated time: 9.859 hours
#     'TRQ177_200624_112234_20200702_154611_Label3D_videos.mat',  # N labeled frames: 30, N video frames: 252746, estimated time: 5.688 hours
#     'TRQ177_200702_135920_20200707_193640_Label3D_videos.mat',  # N labeled frames: 30, N video frames: 452459, estimated time: 5.688 hours
#     'TRQ177_200729_133912_20200811_151418_Label3D_videos.mat',  # N labeled frames: 60, N video frames: 486614, estimated time: 11.376 hours
#     'TRQ177_210630_132311_20210701_155901_Label3D_videos.mat',  # N labeled frames: 60, N video frames: 11765,  estimated time: 11.376 hours
]
training_files = [os.path.join(data_dir, f) for f in training_files]

# these indices have been inspected and are missing from the mat files,
# i.e. all NaNs even though bird is present
bad_idxs = {
    'CHC41_200705_105803_20200720_195918_Label3D_videos.mat': [169680],
    'CHC41_200709_110253_20200709_150243_Label3D_videos.mat': [81720],
    'CHC41_200709_110253_20200917_135958_Label3D_videos.mat': [],
    'CHC41_200723_110437_20200805_114218_Label3D_videos.mat': [48900, 53580, 72480, 77220, 110220, 124380, 143280, 148020, 152700, 162180, 166860, 176340, 181020, 185760, 190500, 199920],
    'EMR30_200714_104454_20200720_111555_Label3D_videos.mat': [102420, 112740, 117960, 154140, ],
    'EMR30_200720_110832_20200722_153703_Label3D_videos.mat': [],
    'IND88_201218_112623_20201223_152611_Label3D_videos.mat': [11617, 207362, 227933, 232763, 243413, 305750, 322460, 380575, 380993, 384169, 410739, 417796, 429510, 445185, 445472, 453217, 457106, 464846, 477194, 480949],
    'PRL43_200617_131904_20200630_170437_Label3D_videos.mat': [],
    'PRL43_200701_142147_20200709_092553_Label3D_videos.mat': [61020, 66180, 81720, 97267, 148980],
    'SLV151_200728_132004_20200729_185246_Label3D_videos.mat': [172740, 178980, 185160, 203820, 210000],
    'SLV151_200730_131948_20200807_122501_Label3D_videos.mat': [114252, 119254, 127652, 128238, 131705, 132967, 146105, 149985, 156163, 162476, 164451, 172709, 175953, 177956, 189944, 193317, 200630, 209977, 211073, 212460, 221110, 222890, 226657, 228348, 240591, 242413, 243109, 250538, 250666, 252358],
    'SLV151_200730_131948_Manual_relabel_videos.mat': [],
    'TRQ66_220728_132027_20220729_151716_Label3D_videos.mat': [23339, 55872, 58568, 63757, 235905, 237233, 237804, 238164, 239021, 240239, 241390, 241457, 242623, 245927, 246018, 247102, 247404, 249805, 254069, 254179, 614839],
    'TRQ177_200624_112234_20200702_154611_Label3D_videos.mat': [102420, 174840],  # 55860 hands+empty
    'TRQ177_200702_135920_20200707_193640_Label3D_videos.mat': [61020, 66180, 180000],
    'TRQ177_200729_133912_20200811_151418_Label3D_videos.mat': [118109, 119007, 120232, 121898, 128530, 144394, 168850, 171527, 185584, 186239, 189334, 191431, 199757, 206109, 206876, 219986, 225743, 226984, 231442, 232086, 234847, 237005, 237901, 239435, 241101, 241723, 255372, 265745, 266428, 267253, 278353],
    'TRQ177_210630_132311_20210701_155901_Label3D_videos.mat': [242220, 312720, 331920],
}

# these sessions do not have a bird in the initial frames so we'll add two blank frames from this section;
# the array corresponds to indices into the labeled frames that are used for extracting random crops
add_blank_frames = {
    'CHC41_200705_105803_20200720_195918_Label3D_videos.mat': np.array([2, 3]),
    'CHC41_200709_110253_20200709_150243_Label3D_videos.mat': np.array([2, 3]),
    'CHC41_200709_110253_20200917_135958_Label3D_videos.mat': None,  # none, use previous session
    'CHC41_200723_110437_20200805_114218_Label3D_videos.mat': np.array([2, 3]),
    'EMR30_200714_104454_20200720_111555_Label3D_videos.mat': np.array([2, 5]),
    'EMR30_200720_110832_20200722_153703_Label3D_videos.mat': np.array([2, 3]),
    'IND88_201218_112623_20201223_152611_Label3D_videos.mat': None,  # bird at beginning
    'PRL43_200617_131904_20200630_170437_Label3D_videos.mat': np.array([2, 8]),
    'PRL43_200701_142147_20200709_092553_Label3D_videos.mat': np.array([9, 11]),
    'SLV151_200728_132004_20200729_185246_Label3D_videos.mat': np.array([3, 21]),
    'SLV151_200730_131948_20200807_122501_Label3D_videos.mat': np.array([3, 7]),
    'SLV151_200730_131948_Manual_relabel_videos.mat': None,  # none, use previous session
    'TRQ66_220728_132027_20220729_151716_Label3D_videos.mat': None,  # bird at beginning
    'TRQ177_200624_112234_20200702_154611_Label3D_videos.mat': np.array([2, 26, 2]),  # includes good empty frame
    'TRQ177_200702_135920_20200707_193640_Label3D_videos.mat': np.array([2, 8]),
    'TRQ177_200729_133912_20200811_151418_Label3D_videos.mat': np.array([10, 11]),
    'TRQ177_210630_132311_20210701_155901_Label3D_videos.mat': None,  # bird at beginning
}

data_info = dataset_info['chickadee']

# set constants
w3d = data_info['w3d']
crop_size = data_info['crop_size']
cam_names = data_info['cam_names']
video_names = [c + '.avi' for c in cam_names]
keypoints = data_info['keypoints']
skeleton = data_info['skeleton']
skeleton_colors = data_info['skeleton_colors']

videos = os.listdir(video_dir)

# process data
for fn in training_files:

    print(fn)
    video_full = os.path.basename(fn)
    video = '_'.join(video_full.split('_')[:3])
    if video not in videos:
        print('no corresponding videos for this session')
        continue

    # --------------------------------------------------
    # load/collect data
    # --------------------------------------------------
    matfile = mat73.loadmat(fn)
    theseLabels, theseScales = projectData(matfile)
    theseImages = []
    for data in matfile['videos']:
        theseImages.append(data[0])

    # --------------------------------------------------
    # find best matching indices
    # --------------------------------------------------
    if video_full == 'CHC41_200709_110253_20200709_150243_Label3D_videos.mat':
        idxs_file = os.path.join(check_dir, video + '_0.npy')
        labels_file = 'labels_0.pqt'
        bboxes_file = 'bboxes_0.csv'
    elif video_full == 'CHC41_200709_110253_20200917_135958_Label3D_videos.mat':
        idxs_file = os.path.join(check_dir, video + '_1.npy')
        labels_file = 'labels_1.pqt'
        bboxes_file = 'bboxes_1.csv'
    elif video_full == 'CHC41_200709_110253_20200923_152544_Label3D_videos.mat':
        idxs_file = os.path.join(check_dir, video + '_2.npy')
        labels_file = 'labels_2.pqt'
        bboxes_file = 'bboxes_2.csv'
    elif video_full == 'SLV151_200730_131948_20200807_122501_Label3D_videos.mat':
        idxs_file = os.path.join(check_dir, video + '_0.npy')
        labels_file = 'labels_0.pqt'
        bboxes_file = 'bboxes_0.csv'
    elif video_full == 'SLV151_200730_131948_Manual_relabel_videos.mat':
        idxs_file = os.path.join(check_dir, video + '_1.npy')
        labels_file = 'labels_1.pqt'
        bboxes_file = 'bboxes_1.csv'
    else:
        idxs_file = os.path.join(check_dir, video + '.npy')
        labels_file = 'labels.pqt'
        bboxes_file = 'bboxes.csv'

    video_files = [os.path.join(video_dir, video, v) for v in video_names]

    if os.path.exists(idxs_file) and not overwrite_idxs:
        best_idxs = np.load(idxs_file)
        print(f'frames loaded from {idxs_file}')
    else:
        # only match a subset of vids for efficiency
        best_idxs, error_array = find_best_frame_indices([theseImages[0]], [video_files[0]])
        np.save(idxs_file, best_idxs)

    if plot_matches:
        plot_frame_matches(
            theseImages,
            video_files,
            best_idxs,
            labels=theseLabels,
            scales=theseScales,
            w3d=w3d,
            save_dir=os.path.join(check_dir, video),
        )

    # --------------------------------------------------
    # modify indices
    # --------------------------------------------------
    # remomve bad indices
    if video_full in bad_idxs.keys():
        bad_idxs_ = bad_idxs[video_full]
        idxs_to_keep = ~np.isin(best_idxs, bad_idxs_)
        best_idxs = best_idxs[idxs_to_keep]
        theseLabels = theseLabels[idxs_to_keep]
        theseScales = theseScales[idxs_to_keep]

    # add in blank frames
    if video_full in add_blank_frames.keys():
        nan_centroid_idxs = add_blank_frames[video_full]
        if nan_centroid_idxs is not None:
            best_idxs = np.insert(best_idxs, 0, 10)
            best_idxs = np.insert(best_idxs, 0, 5)
            _, n_cams, n_kps, n_coords = theseLabels.shape
            theseLabels = np.vstack([np.nan * np.zeros((2, n_cams, n_kps, n_coords)), theseLabels])
            theseScales = np.vstack([np.nan * np.zeros((2, n_cams)), theseScales])
    else:
        nan_centroid_idxs = None

    # --------------------------------------------------
    # output frames and labels for full/cropped datasets
    # --------------------------------------------------
    pdindex = pd.MultiIndex.from_product(
        [['selmaan'], keypoints, ['x', 'y']],
        names=['scorer', 'bodyparts', 'coords'],
    )

    if save_frames_full:
        print(f'EXPORTING {len(best_idxs)} FULL FRAMES')
        for c, cam_name in enumerate(cam_names):
            print(f'Processing cam {c + 1} / {len(cam_names)}')
            save_dir = Path(data_dir_proc) / 'labeled-data' / f'{video}_{cam_name}'
            export_frames(
                video_file=Path(video_dir) / video / (cam_name + '.avi'),
                save_dir=save_dir,
                frame_idxs=best_idxs,
                context_frames=2,
                reader='imageio',  # IMPORTANT!
            )
            # save out labels
            this_df_full = pd.DataFrame(
                theseLabels[:, c, :, :].reshape(theseLabels.shape[0], -1),
                columns=pdindex,
                index=[
                    f'labeled-data/{video}_{cam_name}/img{str(i).zfill(8)}.png' for i in best_idxs
                ],
            )
            this_df_full.to_parquet(save_dir / labels_file)

    if save_frames_full_labeled:
        print(f'EXPORTING {len(best_idxs)} FULL FRAMES (LABELED CHECKS)')
        csv_files = {
            cam_name: pd.read_parquet(
                Path(data_dir_proc) / 'labeled-data' / f'{video}_{cam_name}' / labels_file
            )
            for cam_name in cam_names
        }
        s = 0.5
        linewidth = 0.5
        txt_offset = 50
        height = 5.5
        plot_labeled_frames(
            csv_files=csv_files, data_dir=data_dir_proc, skeleton=skeleton,
            s=s, linewidth=linewidth, txt_offset=txt_offset, height=height,
        )

    if save_frames_cropped:
        print(f'EXPORTING {len(best_idxs)} CROPPED FRAMES')
        for c, cam_name in enumerate(cam_names):
            print(f'Processing cam {c + 1} / {len(cam_names)}')
            save_dir_full = Path(data_dir_proc) / 'labeled-data' / f'{video}_{cam_name}'
            save_dir_crop = Path(data_dir_proc) / 'labeled-data-cropped' / f'{video}_{cam_name}'
            bboxes = export_cropped_frames(
                save_dir_full=save_dir_full,
                save_dir_crop=save_dir_crop,
                frame_idxs=best_idxs,
                labels=theseLabels[:, c, ...],
                scales=theseScales[:, c],
                w3d=0.15,
                resize=crop_size,
                context_frames=2,
                nan_centroid_idxs=nan_centroid_idxs,
            )
            # transform labels
            labels_cropped = theseLabels[:, c, ...].copy()
            labels_cropped[:, :, 0] -= bboxes[:, 0, None]
            labels_cropped[:, :, 0] = labels_cropped[:, :, 0] / bboxes[:, 3, None] * crop_size[0]
            labels_cropped[:, :, 1] -= bboxes[:, 1, None]
            labels_cropped[:, :, 1] = labels_cropped[:, :, 1] / bboxes[:, 2, None] * crop_size[1]
            # save out labels
            df_index = [f'labeled-data-cropped/{video}_{cam_name}/img{str(i).zfill(8)}.png' for i
                        in best_idxs]
            this_df_crop = pd.DataFrame(
                labels_cropped.reshape(labels_cropped.shape[0], -1),
                columns=pdindex,
                index=df_index,
            )
            this_df_crop.to_parquet(save_dir_crop / labels_file)
            # save out bounding boxes
            bbox_df = pd.DataFrame(bboxes, columns=['x', 'y', 'h', 'w'], index=df_index)
            bbox_df.to_csv(save_dir_crop / bboxes_file)

    if save_frames_cropped_labeled:
        print(f'EXPORTING {len(best_idxs)} CROPPED FRAMES (LABELED CHECKS)')
        csv_files = {
            cam_name: pd.read_parquet(
                Path(data_dir_proc) / 'labeled-data-cropped' / f'{video}_{cam_name}' / labels_file
            )
            for cam_name in cam_names
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
