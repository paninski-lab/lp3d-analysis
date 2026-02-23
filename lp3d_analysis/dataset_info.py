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
        'InD_animals': [
            's1-d1',
            's2-d1',
            's2-d2',
            's5-d1',
            's5-d2',
        ],
        'OOD_animals': [
            's3-d1',
            's4-d1',
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
            [ 3,  5],
            [ 3, 12],
            [ 3, 13],
            [ 3,  6],
            [ 5,  0],
            [ 5, 14],
            [ 5, 10],
            [ 5, 11],
            [ 5,  1],
            [ 5,  2],
            [ 18,  20],
            [ 18, 27],
            [ 18, 28],
            [ 18,  21],
            [ 20,  15],
            [ 20, 29],
            [ 20, 25],
            [ 20, 26],
            [ 20,  16],
            [ 20,  17],

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

}