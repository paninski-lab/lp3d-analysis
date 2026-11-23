"""Collect useful dataset info here for standardized handling/plotting."""

import numpy as np


dataset_info = {
    'chickadee': {
        'InD_animals': [
            'CHC41_200705_105803',
            'CHC41_200709_110253',
            'CHC41_200723_110437',
            'EMR30_200714_104454',
            'EMR30_200720_110832',
            'IND88_201218_112623',
            'TRQ66_220728_132027',  # headstage
            'TRQ177_200624_112234',
            'TRQ177_200702_135920',
            'TRQ177_200729_133912',
            'TRQ177_210630_132311',  # headstage+wire
        ],
        'OOD_animals': [
            'PRL43_200617_131904',
            'PRL43_200701_142147',
            'SLV151_200728_132004',
            'SLV151_200730_131948',
        ],
        'cam_names': ['lBack', 'lFront', 'lTop', 'rBack', 'rFront', 'rTop'],
        'keypoints': [
            'topBeak', 'botBeak',
            'topHead', 'backHead',
            'centerChes', 'centerBack',
            'baseTail', 'tipTail',
            'leftEye', 'leftNeck','leftWing', 'leftAnkle', 'leftFoot',
            'rightEye', 'rightNeck', 'rightWing', 'rightAnkle', 'rightFoot',
        ],
        'skeleton': np.array([
            [ 0,  2],
            [ 8,  2],
            [13,  2],
            [ 2,  3],
            [ 5,  3],
            [ 5,  4],
            [ 5,  9],
            [ 9, 10],
            [ 5, 14],
            [14, 15],
            [ 6,  5],
            [ 6,  7],
            [ 6, 11],
            [11, 12],
            [ 6, 16],
            [16, 17],
        ]),
        'skeleton_colors': [
            'yellow',
            'darkred',
            'darkblue',
            'yellow',
            'yellow',
            'yellow',
            'darkgreen',
            'darkgreen',
            'limegreen',
            'limegreen',
            'yellow',
            'yellow',
            'darkred',
            'darkred',
            'red',
            'red',
        ],
        'w3d': 0.15,  # scaling factor for crops
        'crop_size': (320, 320),  # (width, height)
    },
    'fly-anipose': {
        'InD_animals': ['Fly 1_0', 'Fly 2_0', 'Fly 3_0'],
        'OOD_animals': ['Fly 4_0', 'Fly 5_0'],
        'cam_names': ['Cam-A', 'Cam-B', 'Cam-C', 'Cam-D', 'Cam-E', 'Cam-F'],
        'keypoints': [
            'L1A', 'L1B', 'L1C', 'L1D', 'L1E',
            'L2A', 'L2B', 'L2C', 'L2D', 'L2E',
            'L3A', 'L3B', 'L3C', 'L3D', 'L3E',
            'R1A', 'R1B', 'R1C', 'R1D', 'R1E',
            'R2A', 'R2B', 'R2C', 'R2D', 'R2E',
            'R3A', 'R3B', 'R3C', 'R3D', 'R3E',
        ],
        'skeleton': np.array([
            # L1
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 4],
            # L2
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 9],
            # L3
            [10, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            # R1
            [15, 16],
            [16, 17],
            [17, 18],
            [18, 19],
            # R2
            [20, 21],
            [21, 22],
            [22, 23],
            [23, 24],
            # R3
            [25, 26],
            [26, 27],
            [27, 28],
            [28, 29],
        ]),
        'skeleton_colors': [
            # L1
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            # L2
            'darkgreen',
            'darkgreen',
            'darkgreen',
            'darkgreen',
            # L3
            'darkblue',
            'darkblue',
            'darkblue',
            'darkblue',
            # R1
            'red',
            'red',
            'red',
            'red',
            # R2
            'limegreen',
            'limegreen',
            'limegreen',
            'limegreen',
            # R3
            'dodgerblue',
            'dodgerblue',
            'dodgerblue',
            'dodgerblue',
        ],
        'w3d': 1.0,
        'crop_size': None,
    },
    'rat7m': {
        # 'InD_animals': [
        #     's1-d1',
        #     's2-d1',
        #     's2-d2',
        #     's5-d1',
        #     's5-d2',
        # ],
        # 'OOD_animals': [
        #     's3-d1',
        #     's4-d1',
        # ],
        'InD_animals': [
            's1-d1',
            's2-d1',
            's2-d2',
            's3-d1',
        ],
        'OOD_animals': [
            's4-d1',
            's5-d1',
            's5-d2',
        ],
        'cam_names': ['camera1', 'camera2', 'camera3', 'camera4', 'camera5', 'camera6'],
        'keypoints_raw': [
            'HeadF', 'HeadB', 'HeadL',
            'SpineF', 'SpineM', 'SpineL',
            'Offset1', 'Offset2',
            'HipL', 'HipR',
            'ElbowL', 'ArmL', 'ShoulderL', 'ShoulderR', 'ElbowR', 'ArmR', 'KneeR', 'KneeL', 'ShinL', 'ShinR',
        ],
        'skeleton_raw': np.array([
            # head
            [ 0,  1],
            [ 1,  2],
            [ 2,  0],
            [ 0,  3],
            [ 1,  3],
            [ 2,  3],
            # torso
            [ 3,  4],
            [ 4,  5],
            # [ 3,  6],
            # [ 4,  6],
            # [ 4,  7],
            # [ 5,  7],
            # [ 6,  7],
            # l arm
            [ 3, 12],
            [12, 10],
            [10, 11],
            # r arm
            [ 3, 13],
            [13, 14],
            [14, 15],
            # l leg
            [ 5,  8],
            [ 8, 17],
            [17, 18],
            # r leg
            [ 5,  9],
            [ 9, 16],
            [16, 19],
        ]),
        'skeleton_colors_raw': [
            # head
            'blue',
            'blue',
            'blue',
            'blue',
            'blue',
            'blue',
            # torso
            'darkred',
            'darkred',
            # 'darkred',
            # 'darkred',
            # 'darkred',
            # 'darkred',
            # 'pink',
            # l arm
            'yellow',
            'yellow',
            'yellow',
            # r arm
            'limegreen',
            'limegreen',
            'limegreen',
            # l leg
            'dodgerblue',
            'dodgerblue',
            'dodgerblue',
            # r leg
            'darkblue',
            'darkblue',
            'darkblue',
        ],
        'keypoints': [
            # 'HeadF', 'HeadB', 'HeadL',
            'SpineF', 'SpineM', 'SpineL',
            # 'Offset1', 'Offset2',
            'HipL', 'HipR',
            'ElbowL', 'ArmL', 'ShoulderL', 'ShoulderR', 'ElbowR', 'ArmR', 'KneeR', 'KneeL', 'ShinL', 'ShinR',
        ],
        'skeleton': np.array([
            # torso
            [ 0,  1],
            [ 1,  2],
            # l arm
            [ 0,  7],
            [ 7,  5],
            [ 5,  6],
            # r arm
            [ 0,  8],
            [ 8,  9],
            [ 9, 10],
            # l leg
            [ 2,  3],
            [ 3, 12],
            [12, 13],
            # r leg
            [ 2,  4],
            [ 4, 11],
            [11, 14],
        ]),
        'skeleton_colors': [
            # torso
            'darkred',
            'darkred',
            # l arm
            'yellow',
            'yellow',
            'yellow',
            # r arm
            'limegreen',
            'limegreen',
            'limegreen',
            # l leg
            'dodgerblue',
            'dodgerblue',
            'dodgerblue',
            # r leg
            'darkblue',
            'darkblue',
            'darkblue',
        ],
        'w3d': 150,  # scaling factor for crops
        'crop_size': (320, 320),  # (width, height)
    },
    'rat-pup': {
        # single spliced session; no per-animal InD/OOD split available
        'InD_animals': [],
        'OOD_animals': [],
        'cam_names': ['Camera1', 'Camera2', 'Camera3', 'Camera4', 'Camera5', 'Camera6'],
        'keypoints': [
            'Snout',                              # 0
            'EarL', 'EarR',                        # 1, 2
            'SpineF', 'SpineM', 'SpineL',          # 3, 4, 5
            'TailBase',                            # 6
            'ShoulderL', 'ElbowL', 'WristL', 'HandL',  # 7, 8, 9, 10
            'ShoulderR', 'ElbowR', 'WristR', 'HandR',  # 11, 12, 13, 14
            'HipL', 'KneeL', 'AnkleL', 'FootL',    # 15, 16, 17, 18
            'HipR', 'KneeR', 'AnkleR', 'FootR',    # 19, 20, 21, 22
            'TailMid', 'TailTip',                  # 23, 24
        ],
        'skeleton': np.array([
            # head
            [ 0,  1],
            [ 0,  2],
            [ 1,  3],
            [ 2,  3],
            # torso
            [ 3,  4],
            [ 4,  5],
            [ 5,  6],
            # tail
            [ 6, 23],
            [23, 24],
            # l arm
            [ 3,  7],
            [ 7,  8],
            [ 8,  9],
            [ 9, 10],
            # r arm
            [ 3, 11],
            [11, 12],
            [12, 13],
            [13, 14],
            # l leg
            [ 5, 15],
            [15, 16],
            [16, 17],
            [17, 18],
            # r leg
            [ 5, 19],
            [19, 20],
            [20, 21],
            [21, 22],
        ]),
        'skeleton_colors': [
            # head
            'blue',
            'blue',
            'blue',
            'blue',
            # torso
            'darkred',
            'darkred',
            'darkred',
            # tail
            'darkgreen',
            'darkgreen',
            # l arm
            'yellow',
            'yellow',
            'yellow',
            'yellow',
            # r arm
            'limegreen',
            'limegreen',
            'limegreen',
            'limegreen',
            # l leg
            'dodgerblue',
            'dodgerblue',
            'dodgerblue',
            'dodgerblue',
            # r leg
            'darkblue',
            'darkblue',
            'darkblue',
            'darkblue',
        ],
        'w3d': None,  # scaling factor for crops; unused by rat-pup.py's ground-truth-driven cropping
        'crop_size': (320, 320),  # (width, height)
    },
    'human36m': {
        'InD_animals': [
            's_01',
            's_05',
            's_06',
            's_07',
            's_08',
        ],
        'OOD_animals': [
            's_09',
            's_11',
        ],
        'cam_names': ['ca_01', 'ca_02', 'ca_03', 'ca_04'],
        'keypoints': [
            'bottom_torso',
            'l_hip', 'l_knee', 'l_foot',
            'r_hip', 'r_knee', 'r_foot',
            'center_torso', 'upper_torso', 'neck_base', 'center_head',
            'r_shoulder', 'r_elbow', 'r_hand',
            'l_shoulder', 'l_elbow', 'l_hand',
        ],
        'skeleton': np.array([
            # pelvis
            [ 0,  1],
            [ 0,  4],
            # l leg
            [ 1,  2],
            [ 2,  3],
            # r leg
            [ 4,  5],
            [ 5,  6],
            # torso-head
            [ 0,  7],
            [ 7,  8],
            [ 8,  9],
            [ 9, 10],
            # r arm
            [ 8, 11],
            [11, 12],
            [12, 13],
            # l arm
            [ 8, 14],
            [14, 15],
            [15, 16],
        ]),
        'skeleton_colors': [
            # pelvis
            'pink',
            'pink',
            # l leg
            'dodgerblue',
            'dodgerblue',
            # r leg
            'darkblue',
            'darkblue',
            # torso-head
            'red',
            'red',
            'red',
            'red',
            # r arm
            'limegreen',
            'limegreen',
            'limegreen',
            # l arm
            'yellow',
            'yellow',
            'yellow',
        ],
        'w3d': 150,  # scaling factor for crops
        'crop_size': (240, 320),  # (width, height)
    },
    'one-mouse-princeton': {
        'InD_animals': [
            'CSDS-Day1-A_1-Defeat',
            'CSDS-Day1-A_3-Defeat',
            'CSDS-Day3-A_5-Defeat',
            'CSDS-Day3-A_7-Defeat',
            'CSDS-Day5-A_12-Defeat',
            'CSDS-Day7-A_1-Defeat',
            'CSDS-Day1-A_3-Defeat',
            'CSDS-Day9-A_5-Defeat',
            'CSDS-Day9-A_7-Defeat',
        ],
        'OOD_animals': [
            'CSDS-Day5-A_11-Defeat',
        ],
        'cam_names': ['Camera0', 'Camera1', 'Camera2', 'Camera3', 'Camera4'],
        'keypoints': [
            'Nose',             # 0
            'Ear_R',            # 1
            'Ear_L',            # 2
            'TTI',              # 3
            'TailTip',          # 4
            'Head',             # 5
            'Trunk',            # 6
            'Tail_0',           # 7
            'Tail_1',           # 8
            'Tail_2',           # 9
            'Shoulder_left',    # 10
            'Shoulder_right',   # 11
            'Haunch_left',      # 12
            'Haunch_right',     # 13
            'Neck',             # 14
        ],
        'skeleton': np.array([
            [ 3,  5],
            [ 3,  7],
            [ 3,  8],
            [ 3,  9],
            [ 3, 12],
            [ 3, 13],
            [ 3,  6],
            [ 3,  4],
            [ 5,  0],
            [ 5, 14],
            [ 5, 10],
            [ 5, 11],
            [ 5,  1],
            [ 5,  2],
        ]),
        'skeleton_colors': [
            'pink',
            'yellow',
            'yellow',
            'yellow',
            'dodgerblue',
            'dodgerblue',
            'dodgerblue',
            'yellow',
            'red',
            'red',
            'red',
            'red',
            'red',
            'red',
        ],
        'w3d': None,  # scaling factor for crops
        'crop_size': None,  # (width, height)
    },
    'r24m': {
        # multi-animal dataset (2 rats/frame); sessions restricted to calibration=4 viewset
        # (metadata.csv RecordingIndex 29-40) for a consistent 6-camera setup. RecordingIndex
        # 41,45,46,47,48 (all involving SR6m) were dropped entirely: SR6m has zero valid 3D
        # mocap reconstruction (animal2 markers are 100% NaN) in every session it appears in,
        # dataset-wide, not just in this viewset. OOD is instead built from held-out sessions
        # within the working set, chosen so SR1 and SR2 never appear in InD. Round-robin animal
        # identities still mean the remaining InD/OOD rats are not fully disjoint (SR3, SR4,
        # SR5, SR2r, SR4r, SR5r all appear only in InD here).
        'InD_animals': [
            '20210203_Recording_SR2red_SR3_social_wvid',    # RecordingIndex 33, SR3/SR2r
            '20210203_Recording_SR2red_SR5_social_wvid',    # RecordingIndex 34, SR2r/SR5
            '20210203_Recording_SR3_SR4red_social_wvid',    # RecordingIndex 35, SR3/SR4r
            '20210203_Recording_SR3_SR5red_social_wvid',    # RecordingIndex 36, SR3/SR5r
            '20210203_Recording_SR4_SR5red_social_wvid',    # RecordingIndex 37, SR4/SR5r
            '20210204_Recording_SR2red_SR4_social_wvid',    # RecordingIndex 40, SR2r/SR4
        ],
        'OOD_animals': [
            '20210203_Recording_SR1_SR2_social_wvid',       # RecordingIndex 29, SR1/SR2
            '20210203_Recording_SR1_SR3_social_wvid',       # RecordingIndex 30, SR1/SR3
            '20210203_Recording_SR1_SR4_social_wvid',       # RecordingIndex 31, SR4/SR1
            '20210203_Recording_SR1_SR5_social_wvid',       # RecordingIndex 32, SR1/SR5
        ],
        # Camera4 dropped: its raw calibration is ill-conditioned dataset-wide (implausible
        # principal point, fx/fy mismatch, outlier skew -- confirmed across every calibration
        # viewset in metadata.csv, worst in this one), and there's no independent 2D observation
        # source for Camera4 (our labels are projections of 3D points through each camera's own
        # calibration, so a bundle-adjustment-style refinement using only that data is circular).
        'cam_names': ['Camera1', 'Camera2', 'Camera3', 'Camera5', 'Camera6'],
        'keypoints': [
            'HeadF', 'HeadB', 'HeadL',
            'SpineF', 'SpineM', 'SpineL',
            'Offset1', 'Offset2',
            'HipL', 'HipR',
            'ShoulderL', 'ShoulderR',
        ],
        'skeleton': np.array([
            # head
            [0, 1],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 3],
            [2, 3],
            # torso
            [3, 4],
            [4, 5],
            # trunk offset markers
            [3, 6],
            [6, 4],
            [4, 7],
            [7, 5],
            # shoulders
            [3, 10],
            [3, 11],
            # hips
            [5, 8],
            [5, 9],
        ]),
        'skeleton_colors': [
            # head
            'blue',
            'blue',
            'blue',
            'blue',
            'blue',
            'blue',
            # torso
            'darkred',
            'darkred',
            # trunk offset markers
            'pink',
            'pink',
            'pink',
            'pink',
            # shoulders
            'yellow',
            'limegreen',
            # hips
            'dodgerblue',
            'darkblue',
        ],
        'w3d': None,  # scaling factor for crops; TBD once multi-animal cropping is designed
        'crop_size': (320, 320),  # (width, height)
    },
    '3d-pop': {
        # multi-animal dataset (2 or 5 pigeons/frame), 4 views. 10-pigeon sessions are not used, and
        # 2-bird sequences 39 and 50 are excluded (very high NaN rates in 2D labels). "Animals" are
        # sequences: InD = train sequences, OOD = test sequences (random 50/50 split, seed 0,
        # stratified by bird count; no sequence appears in both). See scripts/preprocess/3d-pop.py
        'InD_animals': [
            'Sequence9_n02_01072022',
            'Sequence17_n02_13072022',
            'Sequence21_n02_04072022',
            'Sequence23_n02_04072022',
            'Sequence28_n02_04072022',
            'Sequence32_n02_30062022',
            'Sequence37_n02_30062022',
            'Sequence41_n02_29062022',
            'Sequence46_n02_29062022',
            'Sequence55_n02_28062022',
            'Sequence24_n05_04072022',
            'Sequence29_n05_04072022',
            'Sequence42_n05_29062022',
            'Sequence47_n05_29062022',
            'Sequence53_n05_28062022',
        ],
        'OOD_animals': [
            'Sequence2_n02_01072022',
            'Sequence4_n02_01072022',
            'Sequence7_n02_01072022',
            'Sequence13_n02_13072022',
            'Sequence15_n02_13072022',
            'Sequence26_n02_04072022',
            'Sequence34_n02_30062022',
            'Sequence44_n02_29062022',
            'Sequence52_n02_28062022',
            'Sequence57_n02_28062022',
            'Sequence5_n05_01072022',
            'Sequence10_n05_01072022',
            'Sequence18_n05_13072022',
            'Sequence35_n05_30062022',
            'Sequence58_n05_28062022',
        ],
        'cam_names': ['Cam1', 'Cam2', 'Cam3', 'Cam4'],
        'keypoints': [
            'beak', 'nose', 'leftEye', 'rightEye',
            'leftShoulder', 'rightShoulder', 'topKeel', 'bottomKeel', 'tail',
        ],
        'skeleton': np.array([
            [0, 1],  # beak-nose
            [1, 2],  # nose-leftEye
            [1, 3],  # nose-rightEye
            [2, 3],  # leftEye-rightEye
            [4, 5],  # leftShoulder-rightShoulder
            [4, 6],  # leftShoulder-topKeel
            [5, 6],  # rightShoulder-topKeel
            [4, 8],  # leftShoulder-tail
            [5, 8],  # rightShoulder-tail
            [6, 7],  # topKeel-bottomKeel
            [7, 8],  # bottomKeel-tail
        ]),
        'skeleton_colors': [
            'blue', 'blue', 'blue', 'blue',  # head
            'yellow',  # shoulders
            'darkred', 'darkred', 'darkred', 'darkred',  # body
            'limegreen', 'limegreen',  # keel/tail
        ],
        'w3d': None,  # scaling factor for crops; TBD
        'crop_size': (320, 320),  # (width, height) of the cropped dataset, 3d-pop-crop
    },

    'ibl-mouse': {
        # 'InD_animals': [
        #     'CHC41_200705_105803',
        #     'CHC41_200709_110253',
        #     'CHC41_200723_110437',
        #     'EMR30_200714_104454',
        #     'EMR30_200720_110832',
        #     'IND88_201218_112623',
        #     'TRQ66_220728_132027',  # headstage
        #     'TRQ177_200624_112234',
        #     'TRQ177_200702_135920',
        #     'TRQ177_200729_133912',
        #     'TRQ177_210630_132311',  # headstage+wire
        # ],
        # 'OOD_animals': [
        #     'PRL43_200617_131904',
        #     'PRL43_200701_142147',
        #     'SLV151_200728_132004',
        #     'SLV151_200730_131948',
        # ],
        'cam_names': ['rightCamera','leftCamera'],
        'keypoints': [
            'pawL', 'pawR', 'nose',
        ],
        'skeleton': np.array([
            [ 0,  1],
            [ 1,  2],
            [ 2,  0],
        
        ]),
        'skeleton_colors': [
            'red',
            'blue',
            'blue',
        ],
        'w3d': 0.15,  # scaling factor for crops
        'crop_size': (256, 256),  # (width, height)
    },
    'human36m': {
        'cam_names': ["ca_01", "ca_02", "ca_03", "ca_04"],
        'keypoints': [
            "bottom_torso", "l_hip", "l_knee", "l_foot",
            "r_hip", "r_knee", "r_foot", "center_torso",
            "upper_torso", "neck_base", "center_head",
            "r_shoulder", "r_elbow", "r_hand",
            "l_shoulder", "l_elbow", "l_hand",
        ],
        'skeleton_names': np.array([
            ("bottom_torso", "l_hip"), ("bottom_torso", "r_hip"),
            ("l_hip", "l_knee"), ("l_knee", "l_foot"),
            ("r_hip", "r_knee"), ("r_knee", "r_foot"),
            ("bottom_torso", "center_torso"), ("center_torso", "upper_torso"),
            ("upper_torso", "neck_base"), ("neck_base", "center_head"),
            ("upper_torso", "r_shoulder"), ("r_shoulder", "r_elbow"), ("r_elbow", "r_hand"),
            ("upper_torso", "l_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_hand"),
        ]),
        'skeleton': np.array([
            [ 0,  1],
            [ 1,  2],
            [ 2,  0],
            [ 3,  4],
            [ 4,  5],
            [ 5,  3],
            [ 6,  7],
            [ 7,  8],
            [ 8,  6],
            [ 9, 10],
            [10, 11],
            [11,  9],
            [12, 13],
            [13, 14],
            [14, 12],
            [15, 16],
            [16, 17],
            [17, 15],
        
        ]),

        'skeleton_colors': [
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
        ],
        # 'w3d': 0.15,  # scaling factor for crops
        'crop_size': (240, 320),  # (width, height)
    },

    'two-mouse': {
        'InD_animals': [
            'CSDS-Day1-A_1-Defeat',
            'CSDS-Day1-A_3-Defeat',
            'CSDS-Day3-A_5-Defeat',
            'CSDS-Day3-A_7-Defeat',
            'CSDS-Day5-A_12-Defeat',
            'CSDS-Day7-A_1-Defeat',
            'CSDS-Day1-A_3-Defeat',
            'CSDS-Day9-A_5-Defeat',
            'CSDS-Day9-A_7-Defeat',
        ],
        'OOD_animals': [
            'CSDS-Day5-A_11-Defeat',
        ],
        'cam_names': ['Camera0', 'Camera1', 'Camera2', 'Camera3', 'Camera4'],
        'keypoints': [
            'Nose_black',             # 0
            'Ear_R_black',            # 1
            'Ear_L_black',            # 2
            'TTI_black',              # 3
            'Head_black',             # 4
            'Trunk_black',            # 5
            'Shoulder_left_black',    # 6
            'Shoulder_right_black',   # 7
            'Haunch_left_black',      # 8
            'Haunch_right_black',     # 9
            'Neck_black',             # 10
            'Nose_white',             # 11
            'Ear_R_white',            # 12
            'Ear_L_white',            # 13
            'TTI_white',              # 14
            'Head_white',             # 15
            'Trunk_white',            # 16
            'Shoulder_left_white',    # 17
            'Shoulder_right_white',   # 18
            'Haunch_left_white',      # 19
            'Haunch_right_white',     # 20
            'Neck_white',             # 21
        ],

        'skeleton': np.array([
            [ 3,  4],   # TTI_black -> Head_black
            [ 3,  8],   # TTI_black -> Haunch_left_black
            [ 3,  9],   # TTI_black -> Haunch_right_black
            [ 3,  5],   # TTI_black -> Trunk_black
            [ 4,  0],   # Head_black -> Nose_black
            [ 4, 10],   # Head_black -> Neck_black
            [ 4,  6],   # Head_black -> Shoulder_left_black
            [ 4,  7],   # Head_black -> Shoulder_right_black
            [ 4,  1],   # Head_black -> Ear_R_black
            [ 4,  2],   # Head_black -> Ear_L_black
            [14, 15],   # TTI_white -> Head_white
            [14, 19],   # TTI_white -> Haunch_left_white
            [14, 20],   # TTI_white -> Haunch_right_white
            [14, 16],   # TTI_white -> Trunk_white
            [15, 11],   # Head_white -> Nose_white
            [15, 21],   # Head_white -> Neck_white
            [15, 17],   # Head_white -> Shoulder_left_white
            [15, 18],   # Head_white -> Shoulder_right_white
            [15, 12],   # Head_white -> Ear_R_white
            [15, 13],   # Head_white -> Ear_L_white
        ]),
        'skeleton_colors': [
            # the black mouse 
            'pink',
            'dodgerblue',
            'dodgerblue',
            'dodgerblue',
            'red',
            'red',
            'red',
            'red',
            'red',
            'red',
            # the white mouse 
            'deeppink',
            'darkblue',
            'darkblue',
            'darkblue',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'darkred',
        ],
        'w3d': None,  # scaling factor for crops
        'crop_size': None,  # (width, height)
    },

    'cheese-3d': {
        'cam_names': [],  # to be filled in
        # Order matches CollectedData.csv bodyparts row (column order)
        'keypoints': [
            'ref(head-post)',      # 0
            'lowerlip',            # 1
            'upperlip(left)',      # 2
            'upperlip(right)',     # 3
            'nose(bottom)',        # 4
            'nose(tip)',           # 5
            'nose(top)',           # 6
            'eye(front)(left)',    # 7
            'eye(top)(left)',      # 8
            'eye(back)(left)',     # 9
            'eye(bottom)(left)',   # 10
            'ear(base)(left)',     # 11
            'ear(top)(left)',      # 12
            'ear(tip)(left)',      # 13
            'ear(bottom)(left)',   # 14
            'pad(top)(left)',      # 15
            'pad(side)(left)',     # 16
            'pad(center)',         # 17
            'eye(front)(right)',   # 18
            'eye(top)(right)',     # 19
            'eye(back)(right)',    # 20
            'eye(bottom)(right)',  # 21
            'ear(base)(right)',    # 22
            'ear(top)(right)',     # 23
            'ear(tip)(right)',     # 24
            'ear(bottom)(right)',  # 25
            'pad(top)(right)',     # 26
            'pad(side)(right)',    # 27
        ],
        'skeleton': np.array([
            # nose center (line: bottom -> tip -> top, bridging to pad center)
            [ 4,  5],  # nose(bottom) -> nose(tip)
            [ 5,  6],  # nose(tip) -> nose(top)
            [ 5, 17],  # nose(tip) -> pad(center)
            # left nose pad (closed triangle)
            [17, 15],  # pad(center) -> pad(top)(left)
            [15, 16],  # pad(top)(left) -> pad(side)(left)
            [16, 17],  # pad(side)(left) -> pad(center)
            # right nose pad (closed triangle)
            [17, 26],  # pad(center) -> pad(top)(right)
            [26, 27],  # pad(top)(right) -> pad(side)(right)
            [27, 17],  # pad(side)(right) -> pad(center)
            # mouth (closed triangle)
            [ 1,  2],  # lowerlip -> upperlip(left)
            [ 2,  3],  # upperlip(left) -> upperlip(right)
            [ 3,  1],  # upperlip(right) -> lowerlip
            # left eye (closed quadrilateral)
            [ 7,  8],  # eye(front)(left) -> eye(top)(left)
            [ 8,  9],  # eye(top)(left) -> eye(back)(left)
            [ 9, 10],  # eye(back)(left) -> eye(bottom)(left)
            [10,  7],  # eye(bottom)(left) -> eye(front)(left)
            # right eye (closed quadrilateral)
            [18, 19],  # eye(front)(right) -> eye(top)(right)
            [19, 20],  # eye(top)(right) -> eye(back)(right)
            [20, 21],  # eye(back)(right) -> eye(bottom)(right)
            [21, 18],  # eye(bottom)(right) -> eye(front)(right)
            # left ear (closed quadrilateral)
            [11, 12],  # ear(base)(left) -> ear(top)(left)
            [12, 13],  # ear(top)(left) -> ear(tip)(left)
            [13, 14],  # ear(tip)(left) -> ear(bottom)(left)
            [14, 11],  # ear(bottom)(left) -> ear(base)(left)
            # right ear (closed quadrilateral)
            [22, 23],  # ear(base)(right) -> ear(top)(right)
            [23, 24],  # ear(top)(right) -> ear(tip)(right)
            [24, 25],  # ear(tip)(right) -> ear(bottom)(right)
            [25, 22],  # ear(bottom)(right) -> ear(base)(right)
        ]),
        'skeleton_colors': [
            # nose center
            'dodgerblue',
            'dodgerblue',
            'dodgerblue',
            # left nose pad
            'limegreen',
            'limegreen',
            'limegreen',
            # right nose pad
            'darkgreen',
            'darkgreen',
            'darkgreen',
            # mouth
            'orange',
            'orange',
            'orange',
            # left eye
            'pink',
            'pink',
            'pink',
            'pink',
            # right eye
            'pink',
            'pink',
            'pink',
            'pink',
            # left ear
            'yellow',
            'yellow',
            'yellow',
            'yellow',
            # right ear
            'yellow',
            'yellow',
            'yellow',
            'yellow',
        ],
        'w3d': 0.15,  # scaling factor for crops
        'crop_size': (256, 256),  # (width, height)
    },

}