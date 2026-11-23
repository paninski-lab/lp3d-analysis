import numpy as np
import pandas as pd
from aniposelib.cameras import CameraGroup
from typing import Dict, Tuple, List



def compute_velocity(positions, dt=1.0):
    """Compute velocity via finite differences."""
    vel = np.zeros_like(positions)
    vel[1:] = np.diff(positions, axis=0) / dt
    return vel


def extract_features_sbea_style(points_3d, keypoints, dt=1.0/60.0):
    """
    Feature extraction for clustering (from reference notebook).
    """
    pawL_idx = keypoints.index('pawL')
    pawR_idx = keypoints.index('pawR')
    nose_idx = keypoints.index('nose')
    
    pawL_pos = points_3d[:, pawL_idx, :]
    pawR_pos = points_3d[:, pawR_idx, :]
    nose_pos = points_3d[:, nose_idx, :]

    # Distance between paws
    paw_dist = np.linalg.norm(pawL_pos - pawR_pos, axis=1, keepdims=True)
    
    # Velocities
    pawL_vel = compute_velocity(pawL_pos, dt)
    pawR_vel = compute_velocity(pawR_pos, dt)
    
    # Speeds (scalar)
    pawL_speed = np.linalg.norm(pawL_vel, axis=1, keepdims=True)
    pawR_speed = np.linalg.norm(pawR_vel, axis=1, keepdims=True)

    # Acceleration magnitudes
    pawL_acc = compute_velocity(pawL_vel, dt)
    pawR_acc = compute_velocity(pawR_vel, dt)
    pawL_acc_mag = np.linalg.norm(pawL_acc, axis=1, keepdims=True)
    pawR_acc_mag = np.linalg.norm(pawR_acc, axis=1, keepdims=True)
    
    features = np.concatenate([
        pawL_vel,       # 3: left paw velocity
        pawR_vel,       # 3: right paw velocity
        pawL_speed,     # 1: left paw speed
        pawR_speed,     # 1: right paw speed
        pawL_acc_mag,   # 1: left paw acceleration magnitude
        pawR_acc_mag,   # 1: right paw acceleration magnitude
        paw_dist,       # 1: distance between paws
    ], axis=1)  # total: 11 features
    
    feature_names = [
        'pawL_vx', 'pawL_vy', 'pawL_vz',
        'pawR_vx', 'pawR_vy', 'pawR_vz',
        'pawL_speed', 'pawR_speed',
        'pawL_acc_mag', 'pawR_acc_mag',
        'paw_dist',
    ]
    
    return features, feature_names


def extract_features_all_sessions(
    all_3d_data: Dict,
    keypoints: List[str],
    dt: float = 1.0/60.0,
) -> Tuple[Dict, Dict, List[str]]:
    """
    Extract features for all sessions and models.
    
    Returns:
        all_features: dict[model][session] -> (n_frames, n_features)
        session_labels: dict[model] -> list of session names per frame
        feature_names: list of feature names
    """
    all_features = {m: {} for m in all_3d_data.keys()}
    session_labels = {m: [] for m in all_3d_data.keys()}
    feature_names = None
    
    for model_name, sessions in all_3d_data.items():
        for session_name, points_3d in sessions.items():
            features, feature_names = extract_features_sbea_style(points_3d, keypoints, dt)
            all_features[model_name][session_name] = features
            session_labels[model_name].extend([session_name] * len(features))
    
    return all_features, session_labels, feature_names