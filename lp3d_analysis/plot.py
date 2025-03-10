
import pandas as pd
import numpy as np
import os 
import re
import matplotlib.pyplot as plt


plot_colors = {
    'eks_singleview': 'blue',
    'eks_multiview': 'green',
    'eks_multiview_varinf': '#FF1493',
    'rngs': 'red',
    'ensemble_median': 'purple',
    'ensemble_mean': 'orange',
    'ensemble_variance': 'brown',
    'labels': '#FFD700', #(Bright Gold)
    'pca_reprojection': 'pink',
    'eks_multiview_smooth':  '#FFD700',
    'eks_multiview_varinf_smooth': 'brown', 
    
}


def find_prediction_files(directory, view_name):

    """
    Find files in a directory that match a specific view name pattern.
    
    Args:
        directory (str): Path to the directory to search in
        view_name (str): Name of the view to search for (e.g., "Cam-A")
    
    Returns:
        list: List of full file paths that match the criteria
        
    The function:
    1. Only processes files (not directories)
    2. Looks for the view_name using regex pattern that ensures exact matches
    3. Excludes files containing 'pixel_error'
    4. Returns full file paths of matches
    """

    matching_files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        # First check if it's a file (not a directory)
        if os.path.isfile(filepath):
            # Use regex to find the view name in the filename
            pattern = rf"(?<![A-Z]){re.escape(view_name)}(?![A-Z])"
            if re.search(pattern, filename) and 'pixel_error' not in filename:
                matching_files.append(filepath)
    return matching_files

def generate_paths_with_models_and_ensembles(
    dataset_name, 
    views, 
    results_dir,
    model_type,
    video_dir,
    seed_dirs, 
    ensemble_seed, 
    ensemble_methods, 
    n_hand_labels, 
    base_path="/teamspace/studios",
    data_dir="data",
    studio_dir="this_studio",
    output_dir="outputs"
):
    """
    Generate paths for ground truth and prediction data with models and ensembles.
    
    Args:
        dataset_name (str): Name of the dataset (e.g., 'fly-anipose')
        views (list): List of view names (e.g., ['A', 'B', 'C'])
        seed_dirs (list): List of seed directory names (e.g., ['0', '1', '2'])
        ensemble_seed (str): Seed for ensemble models
        ensemble_methods (list): List of ensemble method names
        n_hand_labels (str): Number of hand labels used
        camera_prefix (str): Prefix for camera views (e.g., "Cam-")
        base_path (str): Base path for all data (default: "/teamspace/studios")
        data_dir (str): Directory containing ground truth data (default: "data")
        studio_dir (str): Directory for studio-specific data (default: "this_studio")
        output_dir (str): Directory for model outputs (default: "outputs")
        
    Returns:
        tuple: (ground_truth_csvs, file_paths)
    """
    # Generate ground truth paths
    # ground_truth_csvs = {
    #     view: os.path.join(base_path, data_dir, dataset_name, f'CollectedData_{view}_new.csv') 
    #     for view in views
    # }

    In_Dist_paths = {
        view: os.path.join(base_path, data_dir, dataset_name, f'CollectedData_{view}.csv') 
        for view in views
    }

    Out_Dist_paths = {
        view: os.path.join(base_path, data_dir, dataset_name, f'CollectedData_{view}_new.csv') 
        for view in views
    }


    # Generate prediction paths
    file_paths = {}
    for view in views:
        view_data = {}
        
        # Add individual model predictions
        for seed in seed_dirs:
            base_dir = os.path.join(
                base_path, 
                studio_dir, 
                output_dir,
                dataset_name,
                results_dir,
                f"{model_type}_{n_hand_labels}_{seed}",
                video_dir
            )
            print(base_dir)
            matching_files = find_prediction_files(base_dir, view)
            if matching_files:
                view_data[seed] = matching_files[0]
        
        # Add ensemble predictions
        if ensemble_methods is not None and ensemble_seed is not None:
            for method in ensemble_methods:
                base_dir = os.path.join(
                    base_path,
                    studio_dir,
                    output_dir,
                    dataset_name,
                    results_dir,
                    f"{model_type}_{n_hand_labels}_{ensemble_seed}",
                    method,
                    video_dir
                )
                matching_files = find_prediction_files(base_dir, view)
                if matching_files:
                    view_data[method] = matching_files[0]
        
        file_paths[view] = view_data
    
    return In_Dist_paths, Out_Dist_paths, file_paths




def organize_data_structure(file_paths, views, seed_dirs, ensemble_methods, include_pixel_error= False, n_frames=None):
    
    # Determine frame indices
    if isinstance(n_frames, tuple):
        start_frame, end_frame = n_frames
    else:
        start_frame, end_frame = 0, n_frames  # Default: first `n_frames` frames
    
    organized_data = {}
    
    for view_idx, view in enumerate(views):

        first_csv = next(iter(file_paths[view].values()))
        first_df = pd.read_csv(first_csv, header=[0, 1, 2], index_col=0)
        keypoints = list(dict.fromkeys(first_df.columns.get_level_values(1)))
        print(f"Keypoints for {view} view: {keypoints}")
        # remove keypoints if stare with unamed
        keypoints = [keypoint for keypoint in keypoints if not keypoint.startswith('Unnamed')]
        print(f"Keypoints for {view} view: {keypoints}")
        
        organized_data[view] = {
            'x': {},
            'y': {},
            'likelihood': {},
            'x_ens_var': {},
            'y_ens_var': {},
            'x_posterior_var': {},
            'y_posterior_var': {},
            'pixel_error': {},
            'mahalanobis': {},
            'keypoints': keypoints

        }
        
        # Process each seed model
        for seed in seed_dirs:
            df = pd.read_csv(file_paths[view][seed], header=[0, 1, 2], index_col=0)
            df= df.iloc[start_frame:end_frame] # select the spesific range of frames
            key = f'rng_{seed}'
            organized_data[view]['x'][key] = df.loc[:, df.columns.get_level_values(2) == 'x'].values
            organized_data[view]['y'][key] = df.loc[:, df.columns.get_level_values(2) == 'y'].values
            
            # Add pixel error if requested
            if include_pixel_error:
                # Construct pixel error file path by replacing .csv with _pixel_error.csv
                pixel_error_path = file_paths[view][seed].replace('.csv', '_pixel_error.csv')
                try:
                    df_error = pd.read_csv(pixel_error_path, header=[0], index_col=0)
                    df_error = df_error.iloc[start_frame:end_frame]
                    if 'set' in df_error.columns:
                        df_error = df_error.drop(columns=['set'])
                    organized_data[view]['pixel_error'][key] = df_error.values
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    print(f"Pixel error file not found or empty for {key} in {view} view")
            
        # Process ensemble methods    
        for method in ensemble_methods:
            df = pd.read_csv(file_paths[view][method], header=[0, 1, 2], index_col=0)
            df = df.iloc[start_frame:end_frame] # select the spesific range of frames
            organized_data[view]['x'][method] = df.loc[:, df.columns.get_level_values(2) == 'x'].values
            organized_data[view]['y'][method] = df.loc[:, df.columns.get_level_values(2) == 'y'].values
            organized_data[view]['likelihood'][method] = df.loc[:, df.columns.get_level_values(2) == 'likelihood'].values
            # now add the x_posterior_var and y_posterior_var only if this method has this information
            # Add variances only if columns contain data
            level_2_cols = df.columns.get_level_values(2)
            
            # Ensemble variance
            if any(col == 'x_ens_var' for col in level_2_cols):
                organized_data[view]['x_ens_var'][method] = df.loc[:, df.columns.get_level_values(2) == 'x_ens_var'].values
                organized_data[view]['y_ens_var'][method] = df.loc[:, df.columns.get_level_values(2) == 'y_ens_var'].values
            
            # Posterior variance
            if any(col == 'x_posterior_var' for col in level_2_cols):
                organized_data[view]['x_posterior_var'][method] = df.loc[:, df.columns.get_level_values(2) == 'x_posterior_var'].values
                organized_data[view]['y_posterior_var'][method] = df.loc[:, df.columns.get_level_values(2) == 'y_posterior_var'].values
            

                # Add pixel error if requested
            if include_pixel_error:
                pixel_error_path = file_paths[view][method].replace('.csv', '_pixel_error.csv')
                try:
                    df_error = pd.read_csv(pixel_error_path, header=[0], index_col=0)
                    df_error = df_error.iloc[start_frame:end_frame]
                    if 'set' in df_error.columns:
                        df_error = df_error.drop(columns=['set'])
                    organized_data[view]['pixel_error'][method] = df_error.values
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    print(f"Pixel error file not found or empty for {method} in {view} view")

    
        # # Add ensemble variance for the current view
        # frames = end_frame - start_frame
        # n_keypoints = ensemble_variances.shape[2]  # Number of keypoints
        
        # if ensemble_variances is not None:
        #     organized_data[view]['x']['ensemble_variance'] = ensemble_variances[start_frame:end_frame, view_idx, :, 0]  # x variances
        #     organized_data[view]['y']['ensemble_variance'] = ensemble_variances[start_frame:end_frame, view_idx, :, 1]  # y variances
            
    return organized_data 