"""
Distillation Pipeline for Multi-View Pose Estimation

This module provides functionality to create pseudo-labeled datasets by:
1. Loading EKS (Ensemble Kalman Smoother) predictions from trained models
2. Filtering out already manually labeled frames
3. Selecting diverse frames using variance-based filtering and 3D clustering
4. Generating CSV files and frame extraction information
5. Optionally extracting video frames
"""

import json
import logging
import os
import subprocess
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from lightning_pose.utils import io as io_utils
from lightning_pose.data.cameras import CameraGroup

from lp3d_analysis.utils import (
    extract_session_info, parse_labeled_frames, extract_sequence_name, 
    parse_session_name, remap_keypoints_to_original_space, create_image_path
)
from lp3d_analysis.io import load_camera_parameters, load_existing_data_with_split
        
logger = logging.getLogger(__name__)


@dataclass
class DistillationConfig:
    """Configuration for the distillation pipeline."""
    dataset_name: str
    base_data_dir: str = "/teamspace/studios/data"
    pseudo_labeled_base_dir: str = "/teamspace/studios/this_studio/pseudo_labeled_dataset"
    
    # Input paths (auto-constructed from dataset_name)
    eks_results_dir: Optional[str] = None
    camera_params_dir: Optional[str] = None
    existing_data_dir: Optional[str] = None
    video_base_dir: Optional[str] = None
    output_dir: Optional[str] = None
    
    # Configurations
    view_names: List[str] = None
    frames_per_video: int = 400
    n_clusters: int = 4000
    n_random_frames: int = 4000
    train_frames: Optional[int] = None
    min_frames_per_video: int = 60
    use_posterior_variance: bool = True
    use_random_selection: bool = False
    
    # Data splitting
    train_probability: float = 0.95
    val_probability: float = 0.05
    random_seed: int = 42
    torch_seed: int = 0
    
    # Frame extraction
    extract_frames: bool = True
    video_extensions: List[str] = None
    copy_existing_data: bool = True
    copy_calibrations: bool = True
    
    def __post_init__(self):
        if self.video_extensions is None:
            self.video_extensions = ['.mp4', '.avi', '.mov']
        self._construct_paths()
    
    def _construct_paths(self):
        """Construct paths based on dataset_name."""
        # Set default paths if not provided
        paths = {
            'existing_data_dir': os.path.join(self.base_data_dir, self.dataset_name),
            'camera_params_dir': os.path.join(self.base_data_dir, self.dataset_name, "calibrations"),
            'output_dir': os.path.join(self.pseudo_labeled_base_dir, self.dataset_name)
        }
        
        # Apply defaults for None values
        for attr, default_path in paths.items():
            if getattr(self, attr) is None:
                setattr(self, attr, default_path)
        
        # Handle video_base_dir specially
        if self.video_base_dir is None and self.eks_results_dir:
            eks_results_path = Path(self.eks_results_dir)
            self.video_base_dir = os.path.join(self.existing_data_dir, eks_results_path.name)
        
        # Validate required paths
        if self.eks_results_dir is None:
            raise ValueError("eks_results_dir is required")

class DistillationPipeline:
    """Complete distillation pipeline in a single class."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.views = config.view_names
    
    def load_eks_data(self) -> Dict:
        """Load EKS data from CSV files."""
        all_eks_data = {}
        
        for filename in os.listdir(self.config.eks_results_dir):
            if not filename.endswith('.csv') or '_uncropped' in filename:
                continue
                
            session_name, view_name = extract_session_info(filename, self.views)
            
            if session_name not in all_eks_data:
                all_eks_data[session_name] = {}
            
            csv_path = os.path.join(self.config.eks_results_dir, filename)
            all_eks_data[session_name][view_name] = self._process_csv_file(csv_path, session_name, view_name)
        
        logger.info(f"Found {len(all_eks_data)} sessions with EKS data")
        return all_eks_data
    
    def _process_csv_file(self, csv_path: str, session_name: str, view_name: str) -> Dict:
        """Process a single CSV file."""
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
        df = io_utils.fix_empty_first_row(df)
        
        # Load bbox and uncropped data if available
        bbox_data = self._load_bbox_data(session_name, view_name)
        uncropped_df = self._load_uncropped_data(csv_path)
        
        # Extract keypoint names
        keypoint_names = [col[1] for col in df.columns if col[2] in ['x', 'y']]
        keypoint_names = list(dict.fromkeys(keypoint_names))
        
        # Process frames
        smoothed_predictions = []
        for array_idx in range(len(df)):
            frame_data = df.iloc[array_idx]
            actual_frame_number = df.index[array_idx]
            
            frame_bbox = bbox_data.loc[actual_frame_number] if bbox_data is not None and actual_frame_number in bbox_data.index else None
            uncropped_frame = uncropped_df.loc[actual_frame_number] if uncropped_df is not None and actual_frame_number in uncropped_df.index else None
            
            smoothed_predictions.append(self._extract_frame_data(
                frame_data, keypoint_names, actual_frame_number, array_idx, df.columns, frame_bbox, uncropped_frame
            ))
        
        return {
            'video_path': csv_path,
            'frame_count': len(df),
            'keypoint_names': keypoint_names,
            'smoothed_predictions': smoothed_predictions,
            'has_bbox_data': bbox_data is not None,
            'has_uncropped_data': uncropped_df is not None
        }
    
    def _load_bbox_data(self, session_name: str, view_name: str) -> Optional[pd.DataFrame]:
        """Load bbox data if available."""
        base_session = session_name.replace('.short', '')
        bbox_path = Path(self.config.video_base_dir) / f"{base_session}_{view_name}.short_bbox.csv"
        return pd.read_csv(bbox_path, index_col=0) if bbox_path.exists() else None
    
    def _load_uncropped_data(self, csv_path: str) -> Optional[pd.DataFrame]:
        """Load uncropped data if available."""
        uncropped_path = csv_path.replace('.csv', '_uncropped.csv')
        if Path(uncropped_path).exists():
            df = pd.read_csv(uncropped_path, header=[0, 1, 2], index_col=0)
            return io_utils.fix_empty_first_row(df)
        return None
    
    def _extract_frame_data(self, frame_data: pd.Series, keypoint_names: List[str], 
                           actual_frame_number: int, array_idx: int, 
                           column_structure: pd.MultiIndex, frame_bbox_data: pd.Series = None,
                           uncropped_frame_data: pd.Series = None) -> Dict:
        """Extract keypoint data for a single frame."""
        scorer = column_structure.levels[0][0]
        
        # Extract keypoints and variances
        keypoints, variances = [], {'x': [], 'y': [], 'post_x': [], 'post_y': []}
        for kp_name in keypoint_names:
            keypoints.append([float(frame_data[(scorer, kp_name, 'x')]), float(frame_data[(scorer, kp_name, 'y')])])
            variances['x'].append(float(frame_data[(scorer, kp_name, 'x_ens_var')]))
            variances['y'].append(float(frame_data[(scorer, kp_name, 'y_ens_var')]))
            try:
                variances['post_x'].append(float(frame_data[(scorer, kp_name, 'x_posterior_var')]))
                variances['post_y'].append(float(frame_data[(scorer, kp_name, 'y_posterior_var')]))
            except KeyError:
                pass
        
        # Handle uncropped data
        uncropped_keypoints, uncropped_variances = None, None
        if uncropped_frame_data is not None:
            try:
                uncropped_keypoints, uncropped_variances = [], {'x': [], 'y': [], 'post_x': [], 'post_y': []}
                for kp_name in keypoint_names:
                    uncropped_keypoints.append([float(uncropped_frame_data[(scorer, kp_name, 'x')]), float(uncropped_frame_data[(scorer, kp_name, 'y')])])
                    uncropped_variances['x'].append(float(uncropped_frame_data[(scorer, kp_name, 'x_ens_var')]))
                    uncropped_variances['y'].append(float(uncropped_frame_data[(scorer, kp_name, 'y_ens_var')]))
                    try:
                        uncropped_variances['post_x'].append(float(uncropped_frame_data[(scorer, kp_name, 'x_posterior_var')]))
                        uncropped_variances['post_y'].append(float(uncropped_frame_data[(scorer, kp_name, 'y_posterior_var')]))
                    except KeyError:
                        pass
            except KeyError:
                uncropped_keypoints, uncropped_variances = keypoints, variances
        
        # Build result
        result = {
            'frame_idx': actual_frame_number,
            'array_idx': array_idx,
            'keypoints_2d': np.array(keypoints),
            'variances_x': np.array(variances['x']),
            'variances_y': np.array(variances['y']),
            'variances_post_x': np.array(variances['post_x']) if variances['post_x'] else None,
            'variances_post_y': np.array(variances['post_y']) if variances['post_y'] else None,
            'keypoints_2d_uncropped': np.array(uncropped_keypoints) if uncropped_keypoints else None,
            'variances_x_uncropped': np.array(uncropped_variances['x']) if uncropped_variances else None,
            'variances_y_uncropped': np.array(uncropped_variances['y']) if uncropped_variances else None,
            'variances_post_x_uncropped': np.array(uncropped_variances['post_x']) if uncropped_variances and uncropped_variances['post_x'] else None,
            'variances_post_y_uncropped': np.array(uncropped_variances['post_y']) if uncropped_variances and uncropped_variances['post_y'] else None,
            'bbox_data': {col: float(val) if pd.notna(val) else None for col, val in frame_bbox_data.items()} if frame_bbox_data is not None else None
        }
        
        return result
    
    def load_existing_data(self) -> Optional[Dict]:
        """Load existing data with train/val split."""
        return load_existing_data_with_split(
            self.config.existing_data_dir, self.views,
            self.config.train_probability, self.config.val_probability,
            self.config.torch_seed, self.config.train_frames
        )


    def add_camera_data_and_triangulate(self, eks_data: Dict) -> Dict:
        """Add camera parameters and perform triangulation or PCA-based 3D reconstruction."""
        for session_name in eks_data:
            logger.info(f"Processing cameras for session: {session_name}")
            
            camera_group = load_camera_parameters(self.config.camera_params_dir, session_name)
            
            if camera_group is not None:
                logger.info(f"Camera parameters found for {session_name}, using triangulation")
                eks_data[session_name]['camera_group'] = camera_group
                eks_data[session_name]['keypoints_3d'] = self._perform_triangulation(
                    eks_data[session_name], camera_group
                )
                eks_data[session_name]['reconstruction_method'] = 'triangulation'
            else:
                logger.info(f"No camera parameters found for {session_name}, using PCA-based 3D reconstruction")
                eks_data[session_name]['camera_group'] = None
                eks_data[session_name]['keypoints_3d'] = self._perform_pca_3d_reconstruction(
                    eks_data[session_name]
                )
                eks_data[session_name]['reconstruction_method'] = 'pca'
        
        return eks_data
    
    def _perform_pca_3d_reconstruction(self, session_data: Dict) -> List[np.ndarray]:
        """Perform PCA-based 3D reconstruction when camera parameters are not available."""
        first_view = next(view for view in self.views if view in session_data)
        num_frames = len(session_data[first_view]['smoothed_predictions'])
        num_keypoints = len(session_data[first_view]['keypoint_names'])
        
        keypoints_3d_all_frames = []
        
        for frame_idx in range(num_frames):
            keypoints_2d_all_views = []
            
            for view_name in self.views:
                if (view_name in session_data and
                    'smoothed_predictions' in session_data[view_name] and
                    frame_idx < len(session_data[view_name]['smoothed_predictions'])):
                    
                    frame_data = session_data[view_name]['smoothed_predictions'][frame_idx]
                    bbox_data = frame_data.get('bbox_data', None)
                    
                    # Use uncropped keypoints if available
                    if (bbox_data is not None and 
                        session_data[view_name].get('has_bbox_data', False) and
                        session_data[view_name].get('has_uncropped_data', False) and
                        frame_data.get('keypoints_2d_uncropped') is not None):
                        keypoints_2d = frame_data['keypoints_2d_uncropped']
                    else:
                        keypoints_2d = frame_data['keypoints_2d']
                        if bbox_data is not None and session_data[view_name].get('has_bbox_data', False):
                            keypoints_2d = remap_keypoints_to_original_space(keypoints_2d, bbox_data)
                    
                    if not (np.any(np.isnan(keypoints_2d)) or np.any(np.isinf(keypoints_2d))):
                        keypoints_2d_all_views.append(keypoints_2d)
            
            # Perform PCA if we have enough views
            if len(keypoints_2d_all_views) >= 2:
                try:
                    keypoints_2d_stacked = np.stack(keypoints_2d_all_views, axis=1)
                    keypoints_2d_flat = keypoints_2d_stacked.reshape(num_keypoints, -1)
                    
                    n_components = min(3, keypoints_2d_flat.shape[1])
                    pca = PCA(n_components=n_components)
                    keypoints_3d = pca.fit_transform(keypoints_2d_flat)
                    
                    # Pad with zeros if needed
                    if keypoints_3d.shape[1] < 3:
                        padding = np.zeros((keypoints_3d.shape[0], 3 - keypoints_3d.shape[1]))
                        keypoints_3d = np.hstack([keypoints_3d, padding])
                    
                    keypoints_3d_all_frames.append(keypoints_3d)
                except Exception as e:
                    logger.warning(f"PCA reconstruction failed for frame {frame_idx}: {e}")
                    keypoints_3d_all_frames.append(np.full((num_keypoints, 3), np.nan))
            else:
                keypoints_3d_all_frames.append(np.full((num_keypoints, 3), np.nan))
        
        return keypoints_3d_all_frames
    
    def _perform_triangulation(self, session_data: Dict, camera_group) -> List[np.ndarray]:
        """Perform triangulation to get 3D keypoints from 2D predictions."""
        first_view = next(view for view in self.views if view in session_data)
        num_frames = len(session_data[first_view]['smoothed_predictions'])
        num_keypoints = len(session_data[first_view]['keypoint_names'])
        
        keypoints_3d_all_frames = []
        
        for frame_idx in range(num_frames):
            keypoints_2d_all_views = []
            
            for view_name in self.views:
                if (view_name in session_data and
                    'smoothed_predictions' in session_data[view_name] and
                    frame_idx < len(session_data[view_name]['smoothed_predictions'])):
                    
                    frame_data = session_data[view_name]['smoothed_predictions'][frame_idx]
                    bbox_data = frame_data.get('bbox_data', None)
                    
                    # Use uncropped keypoints if available
                    if (bbox_data is not None and 
                        session_data[view_name].get('has_bbox_data', False) and
                        session_data[view_name].get('has_uncropped_data', False) and
                        frame_data.get('keypoints_2d_uncropped') is not None):
                        keypoints_2d = frame_data['keypoints_2d_uncropped']
                    else:
                        keypoints_2d = frame_data['keypoints_2d']
                        if bbox_data is not None and session_data[view_name].get('has_bbox_data', False):
                            keypoints_2d = remap_keypoints_to_original_space(keypoints_2d, bbox_data)
                    
                    if not (np.any(np.isnan(keypoints_2d)) or np.any(np.isinf(keypoints_2d))):
                        keypoints_2d_all_views.append(keypoints_2d)
            
            # Perform triangulation if we have enough views
            if len(keypoints_2d_all_views) >= 2:
                try:
                    keypoints_2d_stacked = np.stack(keypoints_2d_all_views)
                    keypoints_3d = camera_group.triangulate_fast(keypoints_2d_stacked, undistort=True)
                    keypoints_3d_all_frames.append(keypoints_3d)
                except Exception as e:
                    logger.warning(f"Triangulation failed for frame {frame_idx}: {e}")
                    keypoints_3d_all_frames.append(np.full((num_keypoints, 3), np.nan))
            else:
                keypoints_3d_all_frames.append(np.full((num_keypoints, 3), np.nan))
        
        return keypoints_3d_all_frames
    
    def filter_out_labeled_frames(self, eks_data: Dict, existing_data: Optional[Dict]) -> Dict:
        """Filter out frames that are already manually labeled."""
        if not existing_data:
            return eks_data
        
        reference_view = self.views[0]
        labeled_frames_by_session = parse_labeled_frames(
            existing_data.get(reference_view, {}).get('image_paths', []), self.views
        )
        
        filtered_data = {}
        total_excluded = 0
        
        for session_name, session_data in eks_data.items():
            labeled_frame_numbers = labeled_frames_by_session.get(session_name, set())
            
            if not labeled_frame_numbers:
                filtered_data[session_name] = session_data
                continue
            
            filtered_session_data = self._filter_session_data(session_data, labeled_frame_numbers)
            filtered_data[session_name] = filtered_session_data
            
            available_view = next((v for v in self.views if v in session_data and v in filtered_session_data), None)
            if available_view:
                original_count = len(session_data[available_view]['smoothed_predictions'])
                filtered_count = len(filtered_session_data[available_view]['smoothed_predictions'])
                total_excluded += original_count - filtered_count
        
        logger.info(f"Excluded {total_excluded} already labeled frames")
        return filtered_data
    
    
    def _filter_session_data(self, session_data: Dict, labeled_frame_numbers: Set[int]) -> Dict:
        """Filter session data to exclude labeled frames."""
        filtered_data = {}
        
        for view_name, view_data in session_data.items():
            if view_name in self.views and 'smoothed_predictions' in view_data:
                frames_to_keep = [pred for pred in view_data['smoothed_predictions'] if pred['frame_idx'] not in labeled_frame_numbers]
                filtered_view_data = view_data.copy()
                filtered_view_data['smoothed_predictions'] = frames_to_keep
                filtered_view_data['frame_count'] = len(frames_to_keep)
                filtered_data[view_name] = filtered_view_data
        
        # Filter 3D keypoints and copy metadata
        if 'keypoints_3d' in session_data:
            reference_view = next((v for v in self.views if v in session_data and 'smoothed_predictions' in session_data[v]), None)
            if reference_view:
                filtered_data['keypoints_3d'] = [kp for i, kp in enumerate(session_data['keypoints_3d']) if session_data[reference_view]['smoothed_predictions'][i]['frame_idx'] not in labeled_frame_numbers]
            else:
                filtered_data['keypoints_3d'] = session_data['keypoints_3d']
        
        for key in ['camera_group', 'camera_params', 'reconstruction_method']:
            if key in session_data:
                filtered_data[key] = session_data[key]
        
        return filtered_data
    
    def filter_by_variance(self, eks_data: Dict) -> Dict:
        """Filter frames by variance to select diverse poses."""
        logger.info("Filtering frames by variance...")
        return self._filter_frames(eks_data, method='variance')
    
    def filter_by_random(self, eks_data: Dict) -> Dict:
        """Filter frames by random selection."""
        logger.info("Filtering frames by random selection...")
        return self._filter_frames(eks_data, method='random')
    
    def _filter_frames(self, eks_data: Dict, method: str) -> Dict:
        """Filter frames using specified method (variance or random)."""
        filtered_data = {}
        
        for session_name, session_data in eks_data.items():
            available_view = next((v for v in self.views if v in session_data and 'smoothed_predictions' in session_data[v]), None)
            if not available_view:
                continue
            
            total_frames = len(session_data[available_view]['smoothed_predictions'])
            num_frames = max(self.config.min_frames_per_video, min(self.config.frames_per_video, total_frames))
            
            if method == 'variance':
                selected_indices = self._select_frames_by_variance(session_data, total_frames, num_frames)
            else:  # random
                selected_indices = self._select_frames_randomly(total_frames, num_frames)
            
            filtered_data[session_name] = self._filter_session_by_indices(session_data, selected_indices)
        
        return filtered_data
    
    def _select_frames_by_variance(self, session_data: Dict, total_frames: int, num_frames: int) -> List[int]:
        """Select frames with lowest variance."""
        frame_scores = []
        
        for array_idx in range(total_frames):
            max_variances = []
            for view_name in self.views:
                if view_name in session_data:
                    view_frame = session_data[view_name]['smoothed_predictions'][array_idx]
                    var_key = 'variances_post' if (self.config.use_posterior_variance and view_frame['variances_post_x'] is not None) else 'variances'
                    variances = np.concatenate([view_frame[f'{var_key}_x'], view_frame[f'{var_key}_y']])
                    max_variances.append(np.max(variances))
            
            frame_scores.append({
                'array_idx': array_idx,
                'max_variance': np.max(max_variances)
            })
        
        # Select best frames (lowest variance)
        frame_scores.sort(key=lambda x: x['max_variance'])
        return [f['array_idx'] for f in frame_scores[:num_frames]]
    
    def _select_frames_randomly(self, total_frames: int, num_frames: int) -> List[int]:
        """Select frames randomly."""
        import random
        random.seed(self.config.random_seed)
        selected_indices = random.sample(range(total_frames), num_frames)
        selected_indices.sort()  # Keep chronological order
        return selected_indices
    
    def _filter_session_by_indices(self, session_data: Dict, indices: List[int]) -> Dict:
        """Filter session data by selected indices."""
        filtered_data = {}
        
        # Filter view data
        for view_name in self.views:
            if view_name in session_data:
                view_data = session_data[view_name].copy()
                filtered_predictions = []
                for idx in indices:
                    if idx < len(view_data['smoothed_predictions']):
                        frame_data = view_data['smoothed_predictions'][idx].copy()
                        frame_data['original_frame_number'] = frame_data['frame_idx']
                        filtered_predictions.append(frame_data)
                view_data.update({'smoothed_predictions': filtered_predictions, 'frame_count': len(filtered_predictions)})
                filtered_data[view_name] = view_data
        
        # Filter 3D keypoints
        if 'keypoints_3d' in session_data:
            reference_view = next((v for v in self.views if v in session_data and 'smoothed_predictions' in session_data[v]), None)
            if reference_view:
                filtered_data['keypoints_3d'] = [{'keypoints_3d': session_data['keypoints_3d'][idx], 'original_frame_number': session_data[reference_view]['smoothed_predictions'][idx]['frame_idx'], 'array_idx': idx} for idx in indices if idx < len(session_data['keypoints_3d'])]
            else:
                filtered_data['keypoints_3d'] = [{'keypoints_3d': session_data['keypoints_3d'][idx], 'original_frame_number': None, 'array_idx': idx} for idx in indices if idx < len(session_data['keypoints_3d'])]
        
        # Copy metadata and frame numbers
        for key in ['camera_group', 'camera_params', 'reconstruction_method']:
            if key in session_data:
                filtered_data[key] = session_data[key]
        
        reference_view = next((v for v in self.views if v in session_data and 'smoothed_predictions' in session_data[v]), None)
        selected_frame_numbers = [session_data[reference_view]['smoothed_predictions'][i]['frame_idx'] for i in indices if i < len(session_data[reference_view]['smoothed_predictions'])] if reference_view else [None] * len(indices)
        filtered_data.update({'selected_array_indices': indices, 'selected_frame_numbers': selected_frame_numbers})
        
        return filtered_data
    
    def create_pseudo_labeled_dataset(self, filtered_data: Dict) -> Dict:
        """Create pseudo-labeled dataset using 3D clustering."""
        logger.info(f"Creating pseudo-labeled dataset with {self.config.n_clusters} clusters...")
        return self._create_pseudo_dataset(filtered_data, method='clustering')
    
    def create_random_pseudo_labeled_dataset(self, filtered_data: Dict) -> Dict:
        """Create pseudo-labeled dataset using random selection."""
        logger.info(f"Creating random pseudo-labeled dataset with {self.config.n_random_frames} frames...")
        return self._create_pseudo_dataset(filtered_data, method='random')
    
    def _create_pseudo_dataset(self, filtered_data: Dict, method: str) -> Dict:
        """Create pseudo-labeled dataset using specified method."""
        # Log reconstruction methods used
        reconstruction_methods = {name: data.get('reconstruction_method', 'unknown') for name, data in filtered_data.items()}
        
        # Collect valid 3D keypoints
        all_keypoints, frame_info = self._collect_3d_keypoints(filtered_data)
        logger.info(f"Collected {len(all_keypoints)} valid 3D poses for {method}")
        
        if method == 'clustering':
            selected_frames, cluster_centers, cluster_labels = self._select_frames_by_clustering(all_keypoints, frame_info)
        else:  # random
            selected_frames, cluster_centers, cluster_labels = self._select_frames_randomly_from_all(all_keypoints, frame_info)
        
        logger.info(f"Selected {len(selected_frames)} frames using {method}")
        
        return {
            'selected_frames': selected_frames,
            'cluster_centers': cluster_centers,
            'all_3d_keypoints': all_keypoints,
            'cluster_labels': cluster_labels,
            'frame_info': frame_info,
            'reconstruction_methods': reconstruction_methods
        }
    
    def _select_frames_by_clustering(self, all_keypoints: np.ndarray, frame_info: List) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """Select frames using K-means clustering."""
        kmeans = KMeans(n_clusters=self.config.n_clusters, random_state=self.config.random_seed, n_init=10)
        cluster_labels = kmeans.fit_predict(all_keypoints)
        
        selected_frames = {}
        for cluster_idx in range(self.config.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_idx)[0]
            if len(cluster_indices) == 0:
                continue
                
            cluster_center = kmeans.cluster_centers_[cluster_idx]
            cluster_frames = all_keypoints[cluster_indices]
            distances = np.linalg.norm(cluster_frames - cluster_center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            
            session_name, filtered_idx, original_frame = frame_info[closest_idx]
            selected_frames[cluster_idx] = {
                'session_name': session_name.replace('.short', ''),
                'original_session_name': session_name,
                'filtered_frame_idx': filtered_idx,
                'original_frame_number': original_frame,
                'cluster_center': cluster_center,
                'num_frames_in_cluster': len(cluster_indices)
            }
        
        return selected_frames, kmeans.cluster_centers_, cluster_labels
    
    def _select_frames_randomly_from_all(self, all_keypoints: np.ndarray, frame_info: List) -> Tuple[Dict, np.ndarray, List]:
        """Select frames randomly from all available frames."""
        import random
        random.seed(self.config.random_seed)
        
        if len(all_keypoints) <= self.config.n_random_frames:
            selected_indices = list(range(len(all_keypoints)))
        else:
            selected_indices = random.sample(range(len(all_keypoints)), self.config.n_random_frames)
            selected_indices.sort()
        
        selected_frames = {}
        for i, idx in enumerate(selected_indices):
            session_name, filtered_idx, original_frame = frame_info[idx]
            selected_frames[i] = {
                'session_name': session_name.replace('.short', ''),
                'original_session_name': session_name,
                'filtered_frame_idx': filtered_idx,
                'original_frame_number': original_frame,
                'cluster_center': all_keypoints[idx],
                'num_frames_in_cluster': 1
            }
        
        return selected_frames, all_keypoints[selected_indices], selected_indices
    
    def _collect_3d_keypoints(self, filtered_data: Dict):
        """Collect valid 3D keypoints from all sessions."""
        all_keypoints = []
        frame_info = []
        
        for session_name, session_data in filtered_data.items():
            if 'keypoints_3d' not in session_data:
                continue
                
            for filtered_idx, kp_data in enumerate(session_data['keypoints_3d']):
                keypoints_3d = kp_data['keypoints_3d']
                if not np.any(np.isnan(keypoints_3d)):
                    all_keypoints.append(keypoints_3d.flatten())
                    frame_info.append((session_name, filtered_idx, kp_data['original_frame_number']))
        
        return np.array(all_keypoints), frame_info
    


    def generate_outputs(self, filtered_data: Dict, pseudo_data: Dict, existing_data: Optional[Dict]) -> Dict:
        """Generate all output files."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Generating output files...")
        
        # Get train/val indices from existing_data (already filtered)
        train_indices = existing_data.get(self.views[0], {}).get('train_indices', []) if existing_data else []
        val_indices = existing_data.get(self.views[0], {}).get('val_indices', []) if existing_data else []
        
        # Generate CSV files
        extraction_info = self._generate_csv_files(filtered_data, pseudo_data, existing_data, output_dir)
        
        # Generate bbox files for each view
        self._generate_bbox_files(filtered_data, pseudo_data, existing_data, output_dir)
        
        # Generate calibrations file
        self._generate_calibrations_file(pseudo_data, existing_data, output_dir)
        
        # Save extraction info
        with open(output_dir / "image_extraction_info.json", 'w') as f:
            json.dump(extraction_info, f, indent=2)
        
        self._print_summary(extraction_info)
        
        return {
            'extraction_info': extraction_info,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'test_indices': []
        }
    
    def _generate_bbox_files(self, filtered_data: Dict, pseudo_data: Dict, 
                            existing_data: Optional[Dict], output_dir: Path):
        """Generate bbox CSV files for each view."""
        logger.info("Generating bbox files...")
        
        for view_name in self.views:
            bbox_data_list = []
            
            # Load existing bbox data
            bbox_data_list.extend(self._load_existing_bbox_data(view_name, existing_data))
            
            # Collect bbox data from pseudo-labeled frames
            bbox_data_list.extend(self._collect_pseudo_bbox_data(view_name, filtered_data, pseudo_data))
            
            # Save bbox CSV if we have data
            if bbox_data_list:
                bbox_df = pd.DataFrame(bbox_data_list)
                bbox_df.set_index('image_path', inplace=True)
                bbox_df = bbox_df[~bbox_df.index.duplicated(keep='first')]
                
                bbox_file = output_dir / f"bboxes_{view_name}.csv"
                bbox_df.to_csv(bbox_file)
                logger.info(f"Saved {len(bbox_df)} total bbox entries to {bbox_file.name}")
            else:
                logger.info(f"No bbox data available for {view_name}")
    
    def _load_existing_bbox_data(self, view_name: str, existing_data: Optional[Dict]) -> List[Dict]:
        """Load existing bbox data for a view."""
        bbox_data_list = []
        if existing_data and self.config.existing_data_dir:
            existing_bbox_file = Path(self.config.existing_data_dir) / f"bboxes_{view_name}.csv"
            if existing_bbox_file.exists():
                existing_bbox_df = pd.read_csv(existing_bbox_file, index_col=0)
                if view_name in existing_data:
                    used_image_paths = set(existing_data[view_name]['image_paths'])
                    for img_path in existing_bbox_df.index:
                        if img_path in used_image_paths:
                            bbox_row = existing_bbox_df.loc[img_path].to_dict()
                            bbox_row['image_path'] = img_path
                            bbox_data_list.append(bbox_row)
        return bbox_data_list
    
    def _collect_pseudo_bbox_data(self, view_name: str, filtered_data: Dict, pseudo_data: Dict) -> List[Dict]:
        """Collect bbox data from pseudo-labeled frames."""
        bbox_data_list = []
        selected_frames = pseudo_data.get('selected_frames', {})
        
        for cluster_id in sorted(selected_frames.keys()):
            frame_info = selected_frames[cluster_id]
            session_name = frame_info['session_name']
            original_session_name = frame_info.get('original_session_name', session_name)
            filtered_idx = frame_info['filtered_frame_idx']
            original_frame = frame_info['original_frame_number']
            
            if (original_session_name in filtered_data and 
                view_name in filtered_data[original_session_name] and
                filtered_idx < len(filtered_data[original_session_name][view_name]['smoothed_predictions'])):
                
                frame_data = filtered_data[original_session_name][view_name]['smoothed_predictions'][filtered_idx]
                bbox_data = frame_data.get('bbox_data')
                
                if bbox_data is not None:
                    img_path = create_image_path(session_name, view_name, original_frame, self.views)
                    bbox_row = bbox_data.copy()
                    bbox_row['image_path'] = img_path
                    bbox_data_list.append(bbox_row)
        
        return bbox_data_list
    
    
    def _generate_csv_files(self, filtered_data: Dict, pseudo_data: Dict, 
                           existing_data: Optional[Dict], output_dir: Path) -> Dict:
        """Generate CSV files for each view."""
        # Create extraction info (existing_data is already filtered)
        extraction_info = self._create_extraction_info(existing_data, pseudo_data)
        
        # Generate CSV for each view
        for view_name in self.views:
            logger.info(f"Processing {view_name}...")
            combined_df = self._create_view_dataframe(view_name, existing_data, pseudo_data, filtered_data)
            
            if not combined_df.empty:
                csv_path = output_dir / f"CollectedData_{view_name}.csv"
                combined_df.to_csv(csv_path)
                logger.info(f"Saved {len(combined_df)} frames to {csv_path.name}")
        
        return extraction_info
    
    def _create_extraction_info(self, existing_data: Optional[Dict], pseudo_data: Dict) -> Dict:
        """Create extraction info for both existing and pseudo-labeled frames."""
        extraction_info = {}
        
        # Add existing frames
        extraction_info.update(self._extract_existing_frames_info(existing_data))
        
        # Add frames from _new.csv files
        extraction_info.update(self._extract_new_csv_frames_info())
        
        # Add pseudo-labeled frames
        extraction_info.update(self._extract_pseudo_frames_info(pseudo_data))
        
        return extraction_info
    
    def _extract_existing_frames_info(self, existing_data: Optional[Dict]) -> Dict:
        """Extract info for existing frames."""
        extraction_info = {}
        if existing_data:
            filtered_df = existing_data.get(self.views[0], {}).get('data', pd.DataFrame())
            if not filtered_df.empty:
                for idx, (img_path, _) in enumerate(filtered_df.iterrows()):
                    frame_info = self._parse_frame_info_from_path(img_path)
                    if frame_info:
                        extraction_info[f"existing_{idx}"] = {**frame_info, 'data_type': 'existing', 'original_img_path': img_path}
        return extraction_info
    
    def _extract_new_csv_frames_info(self) -> Dict:
        """Extract info for frames from _new.csv files."""
        extraction_info = {}
        source_data_dir = Path(self.config.existing_data_dir)
        new_csv_count = 0
        
        for view_name in self.views:
            new_csv_file = source_data_dir / f"CollectedData_{view_name}_new.csv"
            if new_csv_file.exists():
                try:
                    df = pd.read_csv(new_csv_file, header=[0, 1, 2], index_col=0)
                    df = io_utils.fix_empty_first_row(df)
                    for img_path in df.index:
                        frame_info = self._parse_frame_info_from_path(img_path)
                        if frame_info:
                            extraction_info[f"new_csv_{new_csv_count}"] = {**frame_info, 'data_type': 'new_csv'}
                            new_csv_count += 1
                except Exception as e:
                    logger.error(f"Error processing {new_csv_file.name}: {e}")
        
        return extraction_info
    
    def _extract_pseudo_frames_info(self, pseudo_data: Dict) -> Dict:
        """Extract info for pseudo-labeled frames."""
        extraction_info = {}
        for cluster_idx, frame_info in pseudo_data['selected_frames'].items():
            original_frame = frame_info.get('original_frame_number', frame_info.get('original_frame_idx'))
            if original_frame is not None:
                extraction_info[cluster_idx] = {
                    'session_name': frame_info['session_name'],
                    'original_frame_number': int(original_frame),
                    'img_filename': f"img{original_frame:08d}.png",
                    'video_frame_number': int(original_frame),
                    'data_type': 'pseudo_labeled'
                }
        return extraction_info
    
    def _parse_frame_info_from_path(self, img_path: str) -> Optional[Dict]:
        """Parse frame information from image path."""
        import re
        img_filename = img_path.split('/')[-1]
        frame_match = re.search(r'img(\d+)\.png', img_filename)
        
        if frame_match:
            frame_number = int(frame_match.group(1))
            session_dir = img_path.split('/')[-2]
            session_name = extract_sequence_name(session_dir, self.views)
            return {
                'session_name': session_name,
                'original_frame_number': frame_number,
                'img_filename': img_filename,
                'video_frame_number': frame_number
            }
        return None
    
    def _create_view_dataframe(self, view_name: str, existing_data: Optional[Dict], 
                              pseudo_data: Dict, filtered_data: Dict) -> pd.DataFrame:
        """Create combined dataframe for a specific view."""
        frames_list = []
        
        # Add existing data
        if existing_data:
            existing_df = existing_data.get(view_name, {}).get('data', pd.DataFrame())
            if not existing_df.empty:
                existing_flat = existing_df.copy()
                existing_flat.columns = [f"{col[1]}_{col[2]}" if isinstance(col, tuple) else str(col) 
                                       for col in existing_df.columns]
                existing_flat.index = [img_path.replace('.short', '') for img_path in existing_flat.index]
                frames_list.append(existing_flat)
        
        # Add pseudo-labeled data
        pseudo_rows = self._create_pseudo_rows(view_name, pseudo_data, filtered_data)
        if pseudo_rows:
            frames_list.append(pd.DataFrame(pseudo_rows))
        
        if not frames_list:
            return pd.DataFrame()
        
        # Combine and format
        combined_df = pd.concat(frames_list, axis=0)
        combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
        
        # Create multi-level columns
        first_view_data = next((existing_data[v]['data'] for v in self.views if v in existing_data and not existing_data[v]['data'].empty), None)
        
        if first_view_data is not None:
            # Extract keypoint names in original order from existing data
            bodyparts = []
            seen_bodyparts = set()
            for col in first_view_data.columns:
                if isinstance(col, tuple) and len(col) == 3:
                    bp_name = col[1]
                    if bp_name not in seen_bodyparts and col[2] in ['x', 'y']:
                        bodyparts.append(bp_name)
                        seen_bodyparts.add(bp_name)
        else:
            # Fallback to sorted order if no existing data
            bodyparts = sorted(set(col.split('_')[0] for col in combined_df.columns if col.endswith(('_x', '_y'))))
        
        columns = pd.MultiIndex.from_tuples(
            [('anipose', bp, coord) for bp in bodyparts for coord in ['x', 'y']],
            names=['scorer', 'bodyparts', 'coords']
        )
        
        df_columns = [f'{bp}_{coord}' for bp in bodyparts for coord in ['x', 'y']]
        combined_df = combined_df.reindex(columns=df_columns, fill_value=np.nan)
        combined_df.columns = columns
        
        return combined_df


    def _create_pseudo_rows(self, view_name: str, pseudo_data: Dict, filtered_data: Dict) -> List:
        """Create pseudo-labeled rows for a specific view."""
        pseudo_rows = []
        
        for cluster_idx, frame_info in pseudo_data['selected_frames'].items():
            session_name = frame_info['session_name']
            original_session_name = frame_info.get('original_session_name', session_name)
            filtered_idx = frame_info['filtered_frame_idx']
            original_frame = frame_info.get('original_frame_number', frame_info.get('original_frame_idx'))
            
            if original_frame is None:
                continue
                
            if view_name not in filtered_data.get(original_session_name, {}):
                continue
            
            view_data = filtered_data[original_session_name][view_name]
            frame_data = view_data['smoothed_predictions'][filtered_idx]
            
            # Use cropped keypoints for final output
            keypoints_2d = frame_data['keypoints_2d']
            
            # Create image path
            img_path = create_image_path(session_name, view_name, original_frame, self.views)
            
            # Create row data using cropped keypoints
            row_data = {}
            for i, kp in enumerate(view_data['keypoint_names']):
                row_data[f'{kp}_x'] = keypoints_2d[i][0]
                row_data[f'{kp}_y'] = keypoints_2d[i][1]
            
            pseudo_rows.append(pd.Series(row_data, name=img_path))
        
        return pseudo_rows
    
    def _generate_calibrations_file(self, pseudo_data: Dict, existing_data: Optional[Dict], output_dir: Path):
        """Generate calibrations.csv file only if calibration files exist."""
        logger.info("Generating calibrations.csv...")
        
        # Check if any calibration files exist
        calib_dir = output_dir / "calibrations"
        if not calib_dir.exists() or not list(calib_dir.glob("*.toml")):
            logger.info("No calibration files found - skipping calibrations.csv generation")
            return
        
        mappings = []
        
        # Add existing calibrations
        if existing_data:
            existing_df = existing_data.get(self.views[0], {}).get('data', pd.DataFrame())
            if not existing_df.empty:
                for img_path in existing_df.index:
                    session_dir = img_path.split('/')[-2]
                    session_name = extract_sequence_name(session_dir, self.views)
                    
                    # Only add if calibration file exists
                    calib_file = calib_dir / f"{session_name}.toml"
                    if calib_file.exists():
                        mappings.append({
                            '': img_path, 
                            'file': f"calibrations/{session_name}.toml"
                        })
        
        # Add pseudo-labeled calibrations
        for frame_info in pseudo_data['selected_frames'].values():
            original_frame = frame_info.get('original_frame_number', frame_info.get('original_frame_idx'))
            if original_frame is not None:
                session_name = frame_info['session_name']
                
                # Only add if calibration file exists
                calib_file = calib_dir / f"{session_name}.toml"
                if calib_file.exists():
                    img_path = create_image_path(session_name, self.views[0], original_frame, self.views)
                    mappings.append({
                        '': img_path,
                        'file': f"calibrations/{session_name}.toml"
                    })
        
        # Create DataFrame and save only if we have mappings
        if mappings:
            df = pd.DataFrame(mappings)
            df = df.drop_duplicates(subset=[''], keep='first')
            
            calibrations_path = output_dir / "calibrations.csv"
            df.to_csv(calibrations_path, index=False)
            
            logger.info(f"Generated calibrations.csv with {len(df)} mappings")
        else:
            logger.info("No valid calibration mappings found - skipping calibrations.csv generation")
    
    def _print_summary(self, extraction_info: Dict):
        """Print summary of extraction info."""
        existing_count = sum(1 for v in extraction_info.values() if v.get('data_type') == 'existing')
        new_csv_count = sum(1 for v in extraction_info.values() if v.get('data_type') == 'new_csv')
        pseudo_count = sum(1 for v in extraction_info.values() if v.get('data_type') == 'pseudo_labeled')
        
        logger.info(f"Extraction info contains:")
        logger.info(f"  Existing frames: {existing_count}")
        logger.info(f"  _new.csv frames: {new_csv_count}")
        logger.info(f"  Pseudo-labeled frames: {pseudo_count}")
        logger.info(f"  Total frames to extract: {len(extraction_info)}")


    def copy_existing_data(self) -> Dict:
        """Copy existing labeled data (_new.csv files) and their corresponding frames."""
        if not self.config.copy_existing_data:
            logger.info("Skipping existing data copy (disabled in config)")
            return {}
        
        logger.info("Copying existing labeled data...")
        
        source_data_dir = Path(self.config.existing_data_dir)
        output_dir = Path(self.config.output_dir)
        
        copied_files = {}
        failed_files = []
        
        # Copy OOD data (_new.csv files) to output directory
        for view_name in self.views:
            source_file = source_data_dir / f"CollectedData_{view_name}_new.csv"
            if source_file.exists():
                dest_file = output_dir / f"CollectedData_{view_name}_new.csv"
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    if dest_file.exists():
                        dest_file.unlink()
                    shutil.copy2(source_file, dest_file)
                    copied_files[f"CollectedData_{view_name}_new.csv"] = str(dest_file)
                    logger.info(f"Copied OOD data: {source_file.name}")
                except Exception as e:
                    logger.error(f"Failed to copy {source_file.name}: {e}")
                    failed_files.append(source_file.name)
            
            bbox_files = source_data_dir / f"bboxes_{view_name}_new.csv"
            if bbox_files.exists():
                dest_bbox_file = output_dir / f"bboxes_{view_name}_new.csv"
                dest_bbox_file.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    if dest_bbox_file.exists():
                        dest_bbox_file.unlink()
                    shutil.copy2(bbox_files, dest_bbox_file)
                    copied_files[f"bboxes_{view_name}_new.csv"] = str(dest_bbox_file)
                    logger.info(f"Copied OOD bbox data: {bbox_files.name}")
                except Exception as e:
                    logger.error(f"Failed to copy {bbox_files.name}: {e}")
                    failed_files.append(bbox_files.name)
        
        # Copy calibrations_new.csv if it exists
        source_calib_file = source_data_dir / "calibrations_new.csv"
        if source_calib_file.exists():
            dest_calib_file = output_dir / "calibrations_new.csv"
            dest_calib_file.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                if dest_calib_file.exists():
                    dest_calib_file.unlink()
                shutil.copy2(source_calib_file, dest_calib_file)
                copied_files["calibrations_new.csv"] = str(dest_calib_file)
                logger.info(f"Copied OOD calibrations: calibrations_new.csv")
            except Exception as e:
                logger.error(f"Failed to copy calibrations_new.csv: {e}")
                failed_files.append("calibrations_new.csv")
        
        # Copy corresponding frames for the OOD data
        self._copy_existing_frames()
        
        if failed_files:
            logger.error(f"Failed to copy {len(failed_files)} files: {failed_files}")
            raise RuntimeError(f"Failed to copy existing data files: {failed_files}")
        
        logger.info(f"Successfully copied {len(copied_files)} existing data files")
        return copied_files
    
    def _copy_existing_frames(self):
        """Copy frames corresponding to the OOD data (_new.csv files)."""
        source_data_dir = Path(self.config.existing_data_dir)
        output_dir = Path(self.config.output_dir)
        
        copied_count = 0
        failed_count = 0
        
        # Copy frames from each view's _new.csv file
        for view_name in self.views:
            new_csv_file = source_data_dir / f"CollectedData_{view_name}_new.csv"
            if not new_csv_file.exists():
                continue
            
            try:
                df = pd.read_csv(new_csv_file, header=[0, 1, 2], index_col=0)
                df = io_utils.fix_empty_first_row(df)
                
                logger.info(f"Found {len(df)} frames in {new_csv_file.name} to copy")
                
                for img_path in df.index:
                    source_img_path = source_data_dir / "labeled-data" / img_path
                    dest_img_path = output_dir / "labeled-data" / img_path
                    
                    if source_img_path.exists():
                        try:
                            dest_img_path.parent.mkdir(parents=True, exist_ok=True)
                            if dest_img_path.exists():
                                dest_img_path.unlink()
                            shutil.copy2(source_img_path, dest_img_path)
                            copied_count += 1
                        except Exception as e:
                            logger.warning(f"Failed to copy frame {img_path}: {e}")
                            failed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {new_csv_file.name}: {e}")
                failed_count += 1
        
        logger.info(f"Frame copying completed: {copied_count} successful, {failed_count} failed")
    
    def copy_calibrations(self) -> Dict:
        """Copy calibration files from source to output directory."""
        if not self.config.copy_calibrations:
            logger.info("Skipping calibrations copy (disabled in config)")
            return {}
        
        logger.info("Copying calibration files...")
        
        source_calib_dir = Path(self.config.camera_params_dir)
        output_calib_dir = Path(self.config.output_dir) / "calibrations"
        
        copied_files = {}
        failed_files = []
        
        if source_calib_dir.exists():
            output_calib_dir.mkdir(parents=True, exist_ok=True)
            
            calib_files = list(source_calib_dir.glob("*.toml"))
            logger.info(f"Found {len(calib_files)} calibration files to copy")
            
            for calib_file in calib_files:
                dest_file = output_calib_dir / calib_file.name
                try:
                    if dest_file.exists():
                        dest_file.unlink()
                    shutil.copy2(calib_file, dest_file)
                    copied_files[calib_file.name] = str(dest_file)
                    logger.info(f"Copied calibration: {calib_file.name}")
                except Exception as e:
                    logger.error(f"Failed to copy {calib_file.name}: {e}")
                    failed_files.append(calib_file.name)
        else:
            logger.warning(f"Calibration directory not found: {source_calib_dir}")
        
        if failed_files:
            logger.error(f"Failed to copy {len(failed_files)} calibration files: {failed_files}")
            raise RuntimeError(f"Failed to copy calibration files: {failed_files}")
        
        logger.info(f"Successfully copied {len(copied_files)} calibration files")
        return copied_files


    def extract_frames(self, extraction_info: Dict) -> Dict:
        """Extract video frames based on extraction info."""
        if not self.config.extract_frames or not self.config.video_base_dir:
            logger.info("Frame extraction skipped (not enabled or no video directory)")
            return {}
        
        logger.info(f"Extracting {len(extraction_info)} frames...")
        
        # Group frames by session and data type
        session_frames = {}
        for info in extraction_info.values():
            session_name = info['session_name']
            frame_number = info['video_frame_number']
            img_filename = info['img_filename']
            data_type = info['data_type']
            
            if session_name not in session_frames:
                session_frames[session_name] = {'frames': [], 'data_types': []}
            session_frames[session_name]['frames'].append((frame_number, img_filename))
            session_frames[session_name]['data_types'].append(data_type)
        
        # Extract frames
        start_time = time.time()
        all_results = {}
        total_successful = 0
        total_frames = 0
        
        for session_name, session_data in session_frames.items():
            frame_list = session_data['frames']
            data_types = session_data['data_types']
            session_results = self._extract_frames_for_session(session_name, frame_list, data_types)
            all_results[session_name] = session_results
            
            session_successful = sum(session_results.values())
            session_total = len(frame_list) * len(self.views)
            total_successful += session_successful
            total_frames += session_total
        
        total_time = time.time() - start_time
        success_rate = total_successful / total_frames if total_frames > 0 else 0
        
        logger.info(f"Completed: {total_successful}/{total_frames} ({success_rate:.1%}) in {total_time:.1f}s")
        
        # Save results
        results_data = {
            'total_frames': total_frames,
            'successful_frames': total_successful,
            'success_rate': success_rate,
            'processing_time': total_time
        }
        
        with open(Path(self.config.output_dir) / "frame_extraction_results.json", 'w') as f:
            json.dump(results_data, f, indent=2)
        
        return results_data
    
    def _extract_frames_for_session(self, session_name: str, frame_list: List[Tuple[int, str]], data_types: List[str]) -> Dict[str, int]:
        """Extract frames for a session across all views."""
        logger.info(f"Processing session: {session_name}")
        
        results = {}
        for view_name in self.views:
            success_count = 0
            
            # Separate frames by data type
            existing_frames = [(f, img) for i, (f, img) in enumerate(frame_list) if data_types[i] != 'pseudo_labeled']
            pseudo_frames = [(f, img) for i, (f, img) in enumerate(frame_list) if data_types[i] == 'pseudo_labeled']
            
            # Copy existing frames
            if existing_frames:
                success_count += self._copy_frames_from_labeled_data(session_name, view_name, existing_frames)
            
            # Extract pseudo frames
            if pseudo_frames:
                success_count += self._extract_frames_from_video(session_name, view_name, pseudo_frames)
            
            results[view_name] = success_count
            logger.info(f"  {view_name}: {success_count}/{len(frame_list)} frames")
        
        return results
    
    def _copy_frames_from_labeled_data(self, session_name: str, view_name: str, frame_list: List[Tuple[int, str]]) -> int:
        """Copy frames from existing labeled-data directory using dynamic pattern matching."""
        success_count = 0
        
        # Generate all possible source directory patterns dynamically
        clean_session = session_name.replace('.short', '')
        source_patterns = self._generate_video_patterns(clean_session, view_name)
        
        source_dir = None
        for pattern in source_patterns:
            potential_source = Path(self.config.existing_data_dir) / "labeled-data" / pattern
            if potential_source.exists():
                source_dir = potential_source
                break
        
        if not source_dir:
            return 0
        
        output_dir = self._get_output_dir(session_name, view_name)
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}")
            return 0
        
        for frame_number, img_filename in frame_list:
            source_path = source_dir / img_filename
            output_path = output_dir / img_filename
            
            if source_path.exists():
                try:
                    if output_path.exists():
                        output_path.unlink()
                    shutil.copy2(source_path, output_path)
                    success_count += 1
                except Exception as e:
                    logger.warning(f"Failed to copy {img_filename}: {e}")
        
        return success_count
    
    def _extract_frames_from_video(self, session_name: str, view_name: str, frame_list: List[Tuple[int, str]]) -> int:
        """Extract frames from video (fallback method)."""
        video_path = self._find_video_for_view(session_name, view_name)
        if not video_path:
            logger.warning(f"No video found for {session_name}_{view_name}, skipping frame extraction")
            return 0
        
        output_dir = self._get_output_dir(session_name, view_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        success_count = 0
        
        for frame_number, img_filename in frame_list:
            output_path = output_dir / img_filename
            if self._extract_single_frame(video_path, frame_number, output_path):
                success_count += 1
        
        return success_count
    
    def _find_video_for_view(self, session_name: str, view_name: str) -> str:
        """Find video file for a specific view using dynamic pattern generation."""
        clean_session = session_name.replace('.short', '')
        
        # Generate all possible video filename patterns dynamically
        patterns = self._generate_video_patterns(clean_session, view_name)
        
        # Try main directory first
        for pattern in patterns:
            for ext in self.config.video_extensions:
                video_path = Path(self.config.video_base_dir) / f"{pattern}{ext}"
                if video_path.exists():
                    return str(video_path)
        
        # Try alternative directories
        for video_dir in ["videos-full_new", "videos_new"]:
            video_dir_path = Path(self.config.video_base_dir).parent / video_dir
            if video_dir_path.exists():
                for pattern in patterns:
                    for ext in self.config.video_extensions:
                        video_path = video_dir_path / f"{pattern}{ext}"
                        if video_path.exists():
                            return str(video_path)
        
        return ""
    
    def _generate_video_patterns(self, session_name: str, view_name: str) -> List[str]:
        """Generate all possible video filename patterns dynamically."""
        patterns = []
        
        # Split session name into parts
        parts = session_name.split('_')
        
        # Generate patterns by inserting view_name at different positions
        for i in range(len(parts) + 1):
            # Insert view_name at position i
            new_parts = parts[:i] + [view_name] + parts[i:]
            pattern = '_'.join(new_parts)
            patterns.append(pattern)
            patterns.append(f"{pattern}.short")
        
        # Also try replacing any existing view names with the target view
        for i, part in enumerate(parts):
            if part in self.views:
                new_parts = parts.copy()
                new_parts[i] = view_name
                pattern = '_'.join(new_parts)
                patterns.append(pattern)
                patterns.append(f"{pattern}.short")
        
        # Try appending view_name to the end
        patterns.append(f"{session_name}_{view_name}")
        patterns.append(f"{session_name}_{view_name}.short")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_patterns = []
        for pattern in patterns:
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)
        
        return unique_patterns
    
    def _get_output_dir(self, session_name: str, view_name: str) -> Path:
        """Get output directory for a session and view."""
        clean_session = session_name.replace('.short', '')
        dir_name = f"{clean_session}_{view_name}"
        return Path(self.config.output_dir) / "labeled-data" / dir_name
    
    def _extract_single_frame(self, video_path: str, frame_number: int, output_path: Path) -> bool:
        """Extract a single frame using FFmpeg."""
        if output_path.exists():
            return True
        
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cmd = ['ffmpeg', '-i', video_path, '-vf', f'select=eq(n\\,{frame_number})', '-vframes', '1', '-y', str(output_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0 and output_path.exists()
        except Exception as e:
            logger.warning(f"FFmpeg extraction failed for frame {frame_number}: {e}")
            return False
    

def run_distillation_pipeline(config: DistillationConfig) -> Dict:
    """Run the complete distillation pipeline."""
    logger.info("Starting distillation pipeline...")
    
    # Initialize pipeline
    pipeline = DistillationPipeline(config)
    
    # Step 0: Copy existing data and calibrations
    logger.info("Step 0: Copying existing data and calibrations...")
    copied_existing_data = pipeline.copy_existing_data()
    copied_calibrations = pipeline.copy_calibrations()
    
    # Step 1: Load data
    logger.info("Step 1: Loading data...")
    eks_data = pipeline.load_eks_data()
    
    # Load existing data
    existing_data_dict = pipeline.load_existing_data()
    if existing_data_dict:
        existing_data = existing_data_dict['filtered']
        original_existing_data = existing_data_dict['original']
    else:
        existing_data = None
        original_existing_data = None
    
    # Step 2: Add camera data and triangulate
    logger.info("Step 2: Adding camera data and performing 3D reconstruction...")
    eks_data_with_3d = pipeline.add_camera_data_and_triangulate(eks_data)
    
    # Step 3: Filter out labeled frames
    logger.info("Step 3: Filtering out labeled frames...")
    filtered_data = pipeline.filter_out_labeled_frames(eks_data_with_3d, original_existing_data)
    
    # Step 4: Filter frames
    if config.use_random_selection:
        logger.info("Step 4: Filtering by random selection...")
        filtered_frames_data = pipeline.filter_by_random(filtered_data)
    else:
        logger.info("Step 4: Filtering by variance...")
        filtered_frames_data = pipeline.filter_by_variance(filtered_data)
    
    # Step 5: Create pseudo-labeled dataset
    logger.info("Step 5: Creating pseudo-labeled dataset...")
    if config.use_random_selection:
        pseudo_data = pipeline.create_random_pseudo_labeled_dataset(filtered_frames_data)
    else:
        pseudo_data = pipeline.create_pseudo_labeled_dataset(filtered_frames_data)
    
    # Step 6: Generate outputs
    logger.info("Step 6: Generating outputs...")
    outputs = pipeline.generate_outputs(filtered_frames_data, pseudo_data, existing_data)
    
    # Step 7: Extract frames
    logger.info("Step 7: Extracting frames...")
    extraction_results = pipeline.extract_frames(outputs['extraction_info'])
    
    logger.info("Distillation pipeline completed successfully!")
    
    return {
        'eks_data': eks_data_with_3d,
        'filtered_data': filtered_frames_data,
        'pseudo_data': pseudo_data,
        'outputs': outputs,
        'extraction_results': extraction_results,
        'copied_existing_data': copied_existing_data,
        'copied_calibrations': copied_calibrations
    }