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
        'keypoints': [
            'HeadF', 'HeadB', 'HeadL',
            'SpineF', 'SpineM', 'SpineL',
            'Offset1', 'Offset2',
            'HipL', 'HipR',
            'ElbowL', 'ArmL', 'ShoulderL', 'ShoulderR', 'ElbowR', 'ArmR', 'KneeR', 'KneeL', 'ShinL', 'ShinR',
        ],
        'skeleton': np.array([
            # head
            [ 0,  1],
            [ 1,  2],
            [ 1,  0],
            [ 0,  3],
            [ 1,  3],
            [ 2,  3],
            # torso
            [ 3,  4],
            [ 4,  5],
            [ 3,  6],
            [ 4,  6],
            [ 4,  7],
            [ 5,  7],
            [ 6,  7],
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
            'darkred',
            'darkred',
            'darkred',
            'darkred',
            'pink',
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
}
