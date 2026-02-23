
import pandas as pd
import numpy as np
import os
from math import ceil, sqrt
from typing import Dict, Tuple, List
from pathlib import Path
import sys

# Import CameraGroup from lightning-pose instead of aniposelib for faster triangulation
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "lightning-pose"))
from lightning_pose.data.cameras import CameraGroup



# ============================================================
# 2. LOAD 2D PREDICTIONS PER SESSION/VIEW/MODEL
# ============================================================
def load_2d_predictions(csv_path, keypoints):
    """Load 2D predictions from a CSV file.
    Returns dict with 'x', 'y', 'likelihood' arrays of shape (n_frames, n_keypoints).
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    
    x_vals = df.loc[:, df.columns.get_level_values(2) == 'x'].values
    y_vals = df.loc[:, df.columns.get_level_values(2) == 'y'].values
    likelihood = df.loc[:, df.columns.get_level_values(2) == 'likelihood'].values
    
    return {
        'x': x_vals,       # (n_frames, n_keypoints)
        'y': y_vals,
        'likelihood': likelihood,
        'n_frames': len(df),
    }


# ============================================================
# 3. TRIANGULATE 2D -> 3D
# ============================================================
def triangulate_session(session_data_by_view, camera_group, views, n_keypoints):
    """
    Triangulate 2D predictions from multiple views to get 3D coordinates.
    DEPRECATED: Use triangulate_session_vectorized for better performance.
    
    session_data_by_view: dict[view_name] -> {'x': ..., 'y': ..., 'likelihood': ...}
    camera_group: CameraGroup (from lightning_pose.data.cameras)
    
    Returns: np.ndarray of shape (n_frames, n_keypoints, 3)
    """
    n_frames = session_data_by_view[views[0]]['n_frames']
    points_3d = np.full((n_frames, n_keypoints, 3), np.nan)
    
    for frame_idx in range(n_frames):
        # Stack 2D points from all views: shape (n_views, n_keypoints, 2)
        pts_2d_all_views = []
        valid = True
        for view in views:
            vdata = session_data_by_view[view]
            x = vdata['x'][frame_idx]  # (n_keypoints,)
            y = vdata['y'][frame_idx]
            pts_2d = np.stack([x, y], axis=-1)  # (n_keypoints, 2)
            
            if np.any(np.isnan(pts_2d)) or np.any(np.isinf(pts_2d)):
                valid = False
                break
            pts_2d_all_views.append(pts_2d)
        
        if valid and len(pts_2d_all_views) >= 2:
            try:
                pts_2d_stacked = np.stack(pts_2d_all_views)  # (n_views, n_keypoints, 2)
                pts_3d = camera_group.triangulate(pts_2d_stacked, fast=True)
                points_3d[frame_idx] = pts_3d  # (n_keypoints, 3)
            except Exception as e:
                pass  # leave as NaN
    
    return points_3d


def triangulate_session_vectorized(session_data_by_view, camera_group, views, n_keypoints):
    """
    Vectorized triangulation - process all frames at once using lightning-pose's triangulate_fast.
    
    This is significantly faster than the frame-by-frame approach as it:
    1. Stacks all 2D points across frames
    2. Triangulates all points in a single call using cv2.triangulatePoints
    3. Uses nanmedian across camera pairs for robustness
    
    session_data_by_view: dict[view_name] -> {'x': ..., 'y': ..., 'likelihood': ...}
    camera_group: CameraGroup (from lightning_pose.data.cameras)
    views: list of view names
    n_keypoints: number of keypoints
    
    Returns: np.ndarray of shape (n_frames, n_keypoints, 3)
    """
    n_frames = session_data_by_view[views[0]]['n_frames']
    
    # Stack all 2D points: (n_views, n_frames, n_keypoints, 2)
    pts_2d_all = []
    for view in views:
        vdata = session_data_by_view[view]
        pts_2d = np.stack([vdata['x'], vdata['y']], axis=-1)  # (n_frames, n_keypoints, 2)
        pts_2d_all.append(pts_2d)
    
    pts_2d_stacked = np.stack(pts_2d_all)  # (n_views, n_frames, n_keypoints, 2)
    
    # Create per-frame validity mask (same logic as original - frame invalid if ANY keypoint has NaN/inf)
    valid_mask = np.ones(n_frames, dtype=bool)
    for view_pts in pts_2d_stacked:
        valid_mask &= ~np.any(np.isnan(view_pts), axis=(1, 2))
        valid_mask &= ~np.any(np.isinf(view_pts), axis=(1, 2))
    
    # Initialize output with NaN
    points_3d = np.full((n_frames, n_keypoints, 3), np.nan)
    
    n_valid = np.sum(valid_mask)
    if n_valid == 0:
        return points_3d
    
    # Extract valid frames only
    valid_pts = pts_2d_stacked[:, valid_mask, :, :]  # (n_views, n_valid_frames, n_keypoints, 2)
    
    # Reshape for triangulate_fast: (n_views, n_valid_frames * n_keypoints, 2)
    n_views = len(views)
    valid_pts_flat = valid_pts.reshape(n_views, -1, 2)
    
    # Triangulate all points at once using lightning-pose's fast method
    try:
        pts_3d_flat = camera_group.triangulate_fast(valid_pts_flat, undistort=True)  # (n_valid * n_keypoints, 3)
        
        # Reshape back: (n_valid_frames, n_keypoints, 3)
        pts_3d_valid = pts_3d_flat.reshape(n_valid, n_keypoints, 3)
        
        # Assign back to output
        points_3d[valid_mask] = pts_3d_valid
    except Exception as e:
        print(f"  Warning: vectorized triangulation failed: {e}")
        # Fall back to NaN output
    
    return points_3d

def normalize_3d_coordinates_per_session(points_3d: np.ndarray) -> np.ndarray:
    """
    Normalize 3D coordinates to [-1, 1] per session using min-max normalization.
    
    Args:
        points_3d: (n_frames, n_keypoints, 3) array
        
    Returns:
        Normalized array of same shape
    """
    normalized = np.copy(points_3d)
    
    for dim in range(3):  # x, y, z
        vals = points_3d[:, :, dim].flatten()
        valid_vals = vals[~np.isnan(vals)]
        
        if len(valid_vals) == 0:
            continue
            
        min_val = np.min(valid_vals)
        max_val = np.max(valid_vals)
        
        if max_val - min_val > 1e-8:
            normalized[:, :, dim] = 2 * (points_3d[:, :, dim] - min_val) / (max_val - min_val) - 1
        else:
            normalized[:, :, dim] = 0  # constant value -> center at 0
    
    return normalized

def load_and_triangulate_all_sessions(
    all_file_paths: Dict,
    models_of_interest: List[str],
    views: List[str],
    keypoints: List[str],
    camera_params_dir: str,
    normalize: bool = True,
    max_frames_per_session: int = None,
    sessions_of_interest: List[str] = None,
    use_vectorized: bool = True,
) -> Tuple[Dict, Dict, Dict]:
    """
    Load 2D predictions and triangulate to 3D for all sessions and models.
    
    Args:
        all_file_paths: Dict mapping session -> view -> model -> csv_path
        models_of_interest: List of model names to process
        views: List of camera view names
        keypoints: List of keypoint names
        camera_params_dir: Path to directory containing camera calibration files
        normalize: Whether to normalize 3D coordinates per session
        max_frames_per_session: Optional limit on frames per session
        sessions_of_interest: If provided, only process these sessions
        use_vectorized: If True, use fast vectorized triangulation (default: True)
    
    Returns:
        all_3d_data: dict[model][session] -> (n_frames, n_keypoints, 3)
        all_2d_data: dict[model][session][view] -> {'x', 'y', 'likelihood'}
        session_frame_counts: dict[session] -> n_frames
    """
    
    all_3d_data = {m: {} for m in models_of_interest}
    all_2d_data = {m: {} for m in models_of_interest}
    session_frame_counts = {}
    
    # Select triangulation function
    triangulate_fn = triangulate_session_vectorized if use_vectorized else triangulate_session
    if use_vectorized:
        print("Using vectorized triangulation (lightning-pose triangulate_fast)")
    
    # Filter sessions if specified
    sessions_to_process = all_file_paths.keys()
    if sessions_of_interest is not None:
        sessions_to_process = [s for s in sessions_to_process if s in sessions_of_interest]
        print(f"Processing only {len(sessions_to_process)} sessions of interest: {sessions_to_process}")
    
    for session_name in sessions_to_process:
        session_data = all_file_paths[session_name]
        print(f"\n=== Processing session: {session_name} ===")
        
        # Load camera calibration
        base_session_name = session_name.replace('.short', '')
        camera_params_file = Path(camera_params_dir) / f"{base_session_name}.toml"
        
        if not camera_params_file.exists():
            print(f"  Skipping: no calibration file")
            continue
            
        try:
            camera_group = CameraGroup.load(str(camera_params_file))
        except Exception as e:
            print(f"  Skipping: failed to load calibration: {e}")
            continue
        
        for model_name in models_of_interest:
            # Check all views exist for this model
            has_all_views = all(
                model_name in session_data.get(view, {})
                for view in views
            )
            if not has_all_views:
                print(f"  Skipping {model_name}: missing views")
                continue
            
            # Load 2D predictions for each view
            view_data = {}
            for view in views:
                csv_path = session_data[view][model_name]
                view_data[view] = load_2d_predictions(csv_path, keypoints)
            
            # Triangulate to 3D (vectorized or frame-by-frame)
            points_3d = triangulate_fn(view_data, camera_group, views, len(keypoints))
            
            # Optionally limit frames (apply to both 3D and 2D data)
            if max_frames_per_session is not None:
                points_3d = points_3d[:max_frames_per_session]
                # Also truncate 2D data to match
                for view in views:
                    view_data[view] = {
                        'x': view_data[view]['x'][:max_frames_per_session],
                        'y': view_data[view]['y'][:max_frames_per_session],
                        'likelihood': view_data[view]['likelihood'][:max_frames_per_session],
                        'n_frames': min(view_data[view]['n_frames'], max_frames_per_session),
                    }
            
            # Store 2D data (after truncation if applicable)
            all_2d_data[model_name][session_name] = view_data
            
            # Normalize per session
            if normalize:
                points_3d = normalize_3d_coordinates_per_session(points_3d)
            
            all_3d_data[model_name][session_name] = points_3d
            
            n_valid = np.sum(~np.isnan(points_3d[:, :, 0]))
            n_total = points_3d.shape[0] * points_3d.shape[1]
            print(f"  {model_name}: {n_valid}/{n_total} valid 3D points")
            
            if session_name not in session_frame_counts:
                session_frame_counts[session_name] = points_3d.shape[0]
    
    return all_3d_data, all_2d_data, session_frame_counts



# ============================================================
# POOL FEATURES ACROSS ALL SESSIONS AND ALL MODELS (NO REFERENCE)
# ============================================================
def pool_features_all_models_all_sessions(
    all_features: Dict,
    models_of_interest: List[str],
    max_frames_per_session: int = None,
    subsample_factor: int = 1,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], Dict]:
    """
    Pool features across ALL sessions AND ALL models together.
    
    Returns:
        pooled_features: (n_total_frames, n_features) - all models concatenated
        valid_mask: boolean mask for non-NaN rows
        session_labels: session name for each frame
        model_labels: model name for each frame
        model_frame_counts: dict[model] -> number of frames
    """
    pooled_features = []
    session_labels = []
    model_labels = []
    model_frame_counts = {}
    
    for model_name in models_of_interest:
        if model_name not in all_features:
            continue
            
        model_features = []
        
        for session_name, features in all_features[model_name].items():
            if max_frames_per_session is not None:
                features = features[:max_frames_per_session]
            
            # Subsample
            features = features[::subsample_factor]
            
            model_features.append(features)
            session_labels.extend([session_name] * len(features))
            model_labels.extend([model_name] * len(features))
        
        if model_features:
            model_concat = np.concatenate(model_features, axis=0)
            pooled_features.append(model_concat)
            model_frame_counts[model_name] = len(model_concat)
            print(f"{model_name}: {len(model_concat)} frames")
    
    pooled = np.concatenate(pooled_features, axis=0)
    
    # Remove NaN rows
    valid_mask = ~np.any(np.isnan(pooled), axis=1)
    
    print(f"\nTotal pooled: {pooled.shape[0]} frames x {pooled.shape[1]} features")
    print(f"Valid (non-NaN): {np.sum(valid_mask)} frames")
    
    return pooled, valid_mask, session_labels, model_labels, model_frame_counts


def split_labels_by_model(
    labels: np.ndarray,
    model_labels_clean: List[str],
    models_of_interest: List[str],
) -> Dict[str, np.ndarray]:
    """
    Split cluster labels back to each model after pooled clustering.
    """
    labels_by_model = {}
    model_labels_arr = np.array(model_labels_clean)
    
    for model_name in models_of_interest:
        mask = model_labels_arr == model_name
        labels_by_model[model_name] = labels[mask]
    
    return labels_by_model


def split_embeddings_by_model(
    embedding: np.ndarray,
    model_labels_clean: List[str],
    models_of_interest: List[str],
) -> Dict[str, np.ndarray]:
    """
    Split embeddings back to each model.
    """
    embeddings_by_model = {}
    model_labels_arr = np.array(model_labels_clean)
    
    for model_name in models_of_interest:
        mask = model_labels_arr == model_name
        embeddings_by_model[model_name] = embedding[mask]
    
    return embeddings_by_model






def _load_camera_parameters_for_organize(camera_params_dir, dataset_name, session_name):
    """Load camera calibration parameters for data organization."""
    if not os.path.exists(camera_params_dir):
        print(f"Camera parameters directory not found: {camera_params_dir}")
        return None
    
    # Remove .short from session name for calibration files if present
    base_session_name = session_name.replace('.short', '')
    camera_params_file = Path(camera_params_dir) / f"{base_session_name}.toml"
    
    if not camera_params_file.exists():
        print(f"Camera parameters file not found: {camera_params_file}")
        return None
    
    try:
        camera_group = CameraGroup.load(str(camera_params_file))
        print(f"Loaded camera parameters for session: {base_session_name}")
        return camera_group
    except Exception as e:
        print(f"Failed to load camera parameters for session {base_session_name}: {e}")
        return None


def compute_reprojection_error_vs_original(
    points_3d: np.ndarray,
    original_2d_by_view: Dict,
    camera_group,
    views: List[str],
    keypoints: List[str],
) -> np.ndarray:
    """
    Compute reprojection error (from reference notebook).
    
    Returns: (T, K, n_views) array of per-keypoint, per-view errors in pixels
    """
    T, K, _ = points_3d.shape
    n_views = len(views)
    
    reproj_errors = np.full((T, K, n_views), np.nan)
    
    for v_idx, view in enumerate(views):
        cam_names = [c.name for c in camera_group.cameras]
        if view not in cam_names:
            continue
        cam_idx = cam_names.index(view)
        camera = camera_group.cameras[cam_idx]
        
        # Reproject 3D -> 2D
        pts_3d_flat = points_3d.reshape(-1, 3)
        pts_2d_reproj = camera.project(pts_3d_flat)
        pts_2d_reproj = pts_2d_reproj.reshape(T, K, 2)
        
        # Get original 2D predictions
        orig_x = original_2d_by_view[view]['x']
        orig_y = original_2d_by_view[view]['y']
        
        # Compute Euclidean distance
        dx = pts_2d_reproj[:, :, 0] - orig_x
        dy = pts_2d_reproj[:, :, 1] - orig_y
        error = np.sqrt(dx**2 + dy**2)
        
        reproj_errors[:, :, v_idx] = error
    
    return reproj_errors


def compute_reprojection_errors_all_sessions(
    all_3d_data: Dict,
    all_2d_data: Dict,
    camera_params_dir: str,
    views: List[str],
    keypoints: List[str],
) -> Dict:
    """
    Compute reprojection errors for all sessions and models.
    
    Returns:
        reproj_errors: dict[model][session] -> (T, K, n_views) array
    """
    reproj_errors = {m: {} for m in all_3d_data.keys()}
    
    for model_name in all_3d_data.keys():
        for session_name in all_3d_data[model_name].keys():
            if session_name not in all_2d_data[model_name]:
                continue
            
            # Load camera calibration
            base_session_name = session_name.replace('.short', '')
            camera_params_file = Path(camera_params_dir) / f"{base_session_name}.toml"
            
            if not camera_params_file.exists():
                continue
                
            camera_group = CameraGroup.load(str(camera_params_file))
            
            # Get 3D points (need unnormalized for reprojection!)
            # Note: We need to reload without normalization for accurate reprojection
            points_3d = all_3d_data[model_name][session_name]
            original_2d = all_2d_data[model_name][session_name]
            
            errors = compute_reprojection_error_vs_original(
                points_3d, original_2d, camera_group, views, keypoints
            )
            
            reproj_errors[model_name][session_name] = errors
    
    return reproj_errors


def analyze_reprojection_by_cluster_pooled(
    reproj_errors: Dict,
    cluster_labels: np.ndarray,
    session_labels: List[str],
    model_labels: List[str],
    valid_mask: np.ndarray,
    keypoints: List[str],
    models_of_interest: List[str],
    model_short_names: Dict[str, str],
    keypoints_of_interest: List[str] = ['pawL', 'pawR'],
    subsample_factor: int = 1,
) -> pd.DataFrame:
    """
    Compute median reprojection error per model × keypoint × cluster.
    Now also computes SEM across sessions.
    """
    from scipy import stats
    
    results = []
    
    kp_indices = {kp: keypoints.index(kp) for kp in keypoints_of_interest}
    n_clusters = len(np.unique(cluster_labels))
    
    model_labels_arr = np.array(model_labels)
    session_labels_arr = np.array(session_labels)
    
    for model_name in models_of_interest:
        if model_name not in reproj_errors:
            print(f"  Warning: No reprojection errors for {model_name}")
            continue
        
        model_mask = model_labels_arr == model_name
        model_cluster_labels = cluster_labels[model_mask]
        model_session_labels = session_labels_arr[model_mask]
        
        unique_sessions = list(dict.fromkeys(model_session_labels))
        
        # Store per-session errors for SEM calculation
        session_errors_dict = {}  # {session_name: (errors_array, cluster_labels_array)}
        
        for session_name in unique_sessions:
            if session_name not in reproj_errors[model_name]:
                continue
            
            session_errors = reproj_errors[model_name][session_name]
            session_errors_subsampled = session_errors[::subsample_factor]
            
            session_mask = model_session_labels == session_name
            session_cluster_labels = model_cluster_labels[session_mask]
            
            n_session_frames = len(session_cluster_labels)
            
            if n_session_frames > 0 and len(session_errors_subsampled) >= n_session_frames:
                session_errors_dict[session_name] = (
                    session_errors_subsampled[:n_session_frames],
                    session_cluster_labels
                )
        
        if not session_errors_dict:
            print(f"  Warning: Could not align errors for {model_name}")
            continue
        
        # Compute per-cluster, per-keypoint statistics
        for kp_name, kp_idx in kp_indices.items():
            for cluster_id in range(n_clusters):
                # Collect median error from each session for this cluster
                session_medians = []
                all_errors_pooled = []
                
                for session_name, (sess_errors, sess_clusters) in session_errors_dict.items():
                    cluster_mask = sess_clusters == cluster_id
                    n_frames_cluster = np.sum(cluster_mask)
                    
                    if n_frames_cluster == 0:
                        continue
                    
                    kp_errors = sess_errors[cluster_mask, kp_idx, :]
                    kp_errors_mean_views = np.nanmean(kp_errors, axis=1)
                    
                    # Store session median
                    session_median = np.nanmedian(kp_errors_mean_views)
                    if not np.isnan(session_median):
                        session_medians.append(session_median)
                    
                    # Also pool all errors for overall median
                    all_errors_pooled.extend(kp_errors_mean_views[~np.isnan(kp_errors_mean_views)])
                
                if len(all_errors_pooled) == 0:
                    continue
                
                # Overall median (pooled across sessions)
                median_error = np.nanmedian(all_errors_pooled)
                q25 = np.nanpercentile(all_errors_pooled, 25)
                q75 = np.nanpercentile(all_errors_pooled, 75)
                
                # SEM across sessions
                n_sessions = len(session_medians)
                if n_sessions > 1:
                    sem_error = stats.sem(session_medians, nan_policy='omit')
                    mean_of_session_medians = np.mean(session_medians)
                else:
                    sem_error = np.nan
                    mean_of_session_medians = session_medians[0] if session_medians else np.nan
                
                results.append({
                    'model': model_short_names.get(model_name, model_name),
                    'model_full': model_name,
                    'keypoint': kp_name,
                    'cluster': cluster_id,
                    'median_error': median_error,
                    'mean_session_median': mean_of_session_medians,
                    'sem_error': sem_error,
                    'n_sessions': n_sessions,
                    'q25_error': q25,
                    'q75_error': q75,
                    'n_frames': len(all_errors_pooled),
                })
    
    df = pd.DataFrame(results)
    return df

def identify_paw_moving_clusters(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    threshold_percentile: float = 75,
) -> List[int]:
    """
    Identify clusters with high paw movement based on velocity/speed features.
    
    Returns:
        List of cluster IDs considered "paw-moving" states
    """
    pawL_speed_idx = feature_names.index('pawL_speed')
    pawR_speed_idx = feature_names.index('pawR_speed')
    
    # Compute overall speed threshold
    all_pawL_speed = features[:, pawL_speed_idx]
    all_pawR_speed = features[:, pawR_speed_idx]
    
    threshold_L = np.nanpercentile(all_pawL_speed, threshold_percentile)
    threshold_R = np.nanpercentile(all_pawR_speed, threshold_percentile)
    
    paw_moving_clusters = []
    n_clusters = len(np.unique(labels))
    
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_pawL_speed = np.nanmean(features[mask, pawL_speed_idx])
        cluster_pawR_speed = np.nanmean(features[mask, pawR_speed_idx])
        
        if cluster_pawL_speed > threshold_L or cluster_pawR_speed > threshold_R:
            paw_moving_clusters.append(cluster_id)
    
    return paw_moving_clusters
