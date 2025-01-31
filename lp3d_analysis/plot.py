
import pandas as pd
import numpy as np
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
    
}


def organize_data_structure(file_paths, views, seed_dirs, ensemble_methods, ensemble_variances, include_pixel_error= False, n_frames=None):
    
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
            if 'eks' in method:
                organized_data[view]['x_ens_var'][method] = df.loc[:, df.columns.get_level_values(2) == 'x_ens_var'].values
                organized_data[view]['y_ens_var'][method] = df.loc[:, df.columns.get_level_values(2) == 'y_ens_var'].values
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
        
        if ensemble_variances is not None:
            organized_data[view]['x']['ensemble_variance'] = ensemble_variances[start_frame:end_frame, view_idx, :, 0]  # x variances
            organized_data[view]['y']['ensemble_variance'] = ensemble_variances[start_frame:end_frame, view_idx, :, 1]  # y variances
            
    return organized_data