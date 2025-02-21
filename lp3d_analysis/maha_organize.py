
import os
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from eks.stats import compute_mahalanobis

def organize_data_for_mahalanobis(observed_data: np.ndarray, data: Dict[str, Dict], 
                                 views: List[str], key: str,
                                 ensemble_methods: List[str]) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Reorganizes pose estimation data for Mahalanobis distance computation using variances from ensemble methods.
    
    Args:
        observed_data: numpy array of shape (N, C, K) for observations
        data: Dictionary containing data and variances
        views: List of view names
        key: Current prediction key being processed
        ensemble_methods: List of ensemble method names to get variances from
    """
    N, C, K = observed_data.shape
    num_keypoints = K // 2
    
    assert K % 2 == 0, f"Number of coordinates ({K}) should be even"
    
    # Find the appropriate ensemble method to use for variances
    ensemble_method = None
    for method in ensemble_methods:
        if method in data[views[0]]['x_ens_var']:
            ensemble_method = method
            break
    
    if ensemble_method is None:
        raise ValueError("No valid ensemble method found for variance calculation")
    
    keypoint_data = {}
    for kp in range(num_keypoints):
        # Extract x,y coordinates for this keypoint across all views
        xy_indices = slice(2*kp, 2*kp + 2)
        kp_data = observed_data[:, :, xy_indices]  # Shape: (N, C, 2)
        
        # Create variance array using the ensemble method variances
        kp_vars = np.zeros((N, C, 2))
        for v_idx, view in enumerate(views):
            # Access variances from the ensemble method
            kp_vars[:, v_idx, 0] = data[view]['x_ens_var'][ensemble_method][:, kp]
            kp_vars[:, v_idx, 1] = data[view]['y_ens_var'][ensemble_method][:, kp]
        
        # Reshape to (N, 2C)
        xy = kp_data.reshape(N, -1)
        v = kp_vars.reshape(N, -1)
        
        keypoint_data[kp] = {
            'xy': xy,  # Shape: (N, 2C)
            'v': v     # Shape: (N, 2C)
        }
    
    return keypoint_data

def compute_mahalanobis_all_keypoints(keypoint_data, n_latent=3, likelihoods=None, likelihood_threshold=0.9, loading_matrix = None, mean = None):
    """
    Computes Mahalanobis distance for all keypoints.
    
    Args:
        keypoint_data: Dictionary from organize_data_for_mahalanobis
        Other args: Same as compute_mahalanobis function
        
    Returns:
        dict: Dictionary with Mahalanobis distances for each keypoint
    """
    results = {}
    
    for kp, data in keypoint_data.items():
        print(f"Computing Mahalanobis for keypoint {kp}")
        # Get data for this keypoint
        xy = data['xy']
        v = data['v']
        
        # Compute Mahalanobis distance
        kp_likelihoods = likelihoods[:, kp] if likelihoods is not None else None
        # I need to get the square root of the variance

        results[kp] = compute_mahalanobis(
            x=xy,
            v=v,
            n_latent=n_latent,
            likelihoods=kp_likelihoods,
            likelihood_threshold=likelihood_threshold,
            loading_matrix=loading_matrix,
            mean=mean
        )
        for view_idx in results[kp]['mahalanobis'].keys():
            results[kp]['mahalanobis'][view_idx] = np.sqrt(results[kp]['mahalanobis'][view_idx])
            print(results[kp]['mahalanobis'][view_idx].shape)
        
    
    return results


def process_mahalanobis(
    data: Dict[str, Dict], 
    views: List[str], 
    keypoint_names: List[str], 
    ensemble_methods: List[str],
    loading_matrix: Optional[np.ndarray] = None,
    mean: Optional[np.ndarray] = None,
) -> Dict[str, Dict]:

    """
    Process Mahalanobis distances using ensemble method variances.
    
    Args:
        data: Dictionary containing all data and variances
        views: List of view names
        keypoint_names: List of keypoint names
        ensemble_methods: List of ensemble method names
        loading_matrix (np.ndarray): shape (2C x n_latent) optional
        mean (np.ndarray): shape (2C,) optional
    """
    n_frames = next(iter(data[views[0]]['x'].values())).shape[0]
    n_keypoints = len(keypoint_names)
    
    # Initialize result structures
    for view in views:
        for result_type in ['mahalanobis', 'reconstruct_x', 'reconstruct_y',
                           'posterior_variance_x', 'posterior_variance_y']:
            data[view][result_type] = {}
    
    # Get all valid prediction keys (including both seeds and ensemble methods)
    valid_keys = [k for k in data[views[0]]['x'].keys() 
                 if k not in ['ensemble_variance']]
    
    for key in valid_keys:
        # Create observed data array
        observed_data = np.zeros((n_frames, len(views), 2*n_keypoints))
        
        # Fill observed data array
        for kp_idx in range(n_keypoints):
            for v_idx, view in enumerate(views):
                observed_data[:, v_idx, 2*kp_idx] = data[view]['x'][key][:, kp_idx]
                observed_data[:, v_idx, 2*kp_idx + 1] = data[view]['y'][key][:, kp_idx]
        
        # Process Mahalanobis distances using ensemble method variances
        keypoint_data = organize_data_for_mahalanobis(
            observed_data=observed_data,
            data=data,
            views=views,
            key=key,
            ensemble_methods=ensemble_methods
        )
        
        results = compute_mahalanobis_all_keypoints(keypoint_data,loading_matrix=loading_matrix, mean=mean)
        
        # Store results
        for view_idx, view in enumerate(views):
            for result_type in ['mahalanobis', 'reconstruct_x', 'reconstruct_y',
                              'posterior_variance_x', 'posterior_variance_y']:
                data[view][result_type][key] = {}
                
            for kp_idx, kp_name in enumerate(keypoint_names):
                data[view]['mahalanobis'][key][kp_name] = results[kp_idx]['mahalanobis'][view_idx]
                data[view]['reconstruct_x'][key][kp_name] = results[kp_idx]['reconstructed'][:, 2*view_idx]
                data[view]['reconstruct_y'][key][kp_name] = results[kp_idx]['reconstructed'][:, 2*view_idx + 1]
                data[view]['posterior_variance_x'][key][kp_name] = results[kp_idx]['posterior_variance'][view_idx][:, 0, 0]
                data[view]['posterior_variance_y'][key][kp_name] = results[kp_idx]['posterior_variance'][view_idx][:, 1, 1]
    
    return data