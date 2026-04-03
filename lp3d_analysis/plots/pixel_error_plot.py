import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr

from lp3d_analysis.utils import io_utils
from lp3d_analysis.plot import plot_colors 



def compute_ensemble_stddev(
    df_ground_truth,
    df_preds,
    keypoint_ensemble_list,
    scorer_name='standard_scorer'
):
    """
    Parameters
    ----------
    df_ground_truth : List[pd.DataFrame]
        ground truth predictions
    df_preds : List[pd.DataFrame]
        model predictions
    keypoint_ensemble_list : List[str]
        keypoints to include in the analysis

    Returns
    -------
    np.ndarray
        shape (n_frames, n_keypoints)
    """
    # Initial check for NaNs in df_preds
    for i, df in enumerate(df_preds):
        if df.isna().any().any():
            print(f"Warning: NaN values detected in initial DataFrame {i}.")
            nan_indices = df[df.isna().any(axis=1)].index
            nan_columns = df.columns[df.isna().any()]
            print(f"NaN values found at indices: {nan_indices} in columns: {nan_columns}")

    preds = []
    cols_order = None
    for i, df in enumerate(df_preds):
        assert np.all(
            df.index == df_ground_truth.index
        ), f"Index mismatch between ground truth and predictions at dataframe {i}"

        # Standardize the 'scorer' level
        df = standardize_scorer_level(df, scorer_name)

        print(f"DataFrame {i} initial columns:", df.columns)
        print(f" the df_preds shape is {df.shape}")

        # # Remove likelihood columns
        # cols_to_keep = [
        #     col for col in df.columns
        #     if not col[2].endswith('likelihood') and 'zscore' not in col[2]]
        # # Keep only columns matching the keypoint_ensemble_list
        # cols_to_keep = [col for col in cols_to_keep if col[1] in keypoint_ensemble_list]
        # df = df[cols_to_keep]

        cols_to_keep = [
            col for col in df.columns
            if col[2] in ['x', 'y'] and col[1] in keypoint_ensemble_list
        ]
        df = df[cols_to_keep]

        print(f"DataFrame {i} kept columns:", df.columns)

        # Check for NaNs in the DataFrame
        if df.isna().any().any():
            print(f"Warning: NaN values detected in DataFrame {i} after filtering.")
            nan_indices = df[df.isna().any(axis=1)].index
            nan_columns = df.columns[df.isna().any()]
            print(f"NaN values found at indices: {nan_indices} in columns: {nan_columns}")

        # Print the order of the column headers
        if cols_order is None:
            cols_order = df.columns
        else:
            # Check if column lengths match before comparing
            if len(df.columns) != len(cols_order):
                print(f"Column length mismatch detected in DataFrame {i}")
                print(f"Expected length: {len(cols_order)}, Actual length: {len(df.columns)}")
                print("Expected columns:", cols_order)
                print("Actual columns:", df.columns)
            elif not (df.columns == cols_order).all():
                print(f"Column order mismatch detected in DataFrame {i}")
                print("Expected order:", cols_order)
                print("Actual order:", df.columns)
                # Ensure bodyparts and coordinates are consistent
                expected_bodyparts_coords = cols_order.droplevel(0).unique()
                actual_bodyparts_coords = df.columns.droplevel(0).unique()
                if not expected_bodyparts_coords.equals(actual_bodyparts_coords):
                    print("Bodyparts and coordinates mismatch detected")
                    print("Expected bodyparts and coordinates:", expected_bodyparts_coords)
                    print("Actual bodyparts and coordinates:", actual_bodyparts_coords)

        # Reshape the DataFrame to the appropriate shape
        try:
            arr = df.to_numpy().reshape(df.shape[0], -1, 2)
        except ValueError as e:
            print(f"Reshape error: {e}")
            print(f"DataFrame shape: {df.shape}")
            print(f"Array shape after reshape attempt: {df.to_numpy().shape}")
            raise

        preds.append(arr[..., None])

    preds = np.concatenate(preds, axis=3)

    # Check for NaNs in preds
    if np.isnan(preds).any():
        print("Warning: NaN values detected in preds array.")
        nan_indices = np.argwhere(np.isnan(preds))
        print(f"NaN values found at indices: {nan_indices}")
    else:
        print("No NaN values detected in preds array.")

    stddevs = np.std(preds, axis=-1).mean(axis=-1)
    print(f"Stddevs: {stddevs}")
    return stddevs


def standardize_scorer_level(df, new_scorer='standard_scorer'):
    """
    Standardizes the 'scorer' level in the MultiIndex to a common name.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to standardize.
    new_scorer : str
        The new name for the 'scorer' level.

    Returns
    -------
    pd.DataFrame
        The DataFrame with the standardized 'scorer' level.
    """
    df.columns = pd.MultiIndex.from_tuples(
        [(new_scorer, bodypart, coord) for scorer, bodypart, coord in df.columns],
        names=df.columns.names
    )
    return df


def compute_percentiles(arr, std_vals, percentiles):
    num_pts = arr[0]
    vals = []
    prctiles = []
    for p in percentiles:
        v = num_pts * p / 100
        idx = np.argmin(np.abs(arr - v))
        # maybe we don't have enough data
        if idx == len(arr) - 1:
            p_ = arr[idx] / num_pts * 100
        else:
            p_ = p
        vals.append(std_vals[idx])
        prctiles.append(np.round(p_, 2))
    return vals, prctiles

def get_base_model_name(model_name):
    """Extract base model name without the run number."""
    if '.0' in model_name or '.1' in model_name or '.2' in model_name or '.3' in model_name or '.4' in model_name:
        return model_name.rsplit('.', 1)[0]
    return model_name


def process_view_data(ground_truth_csv, view_data, view_name, keypoints_to_use=None, ensemble_methods=None, sessions_to_ignore=None, frames_to_exclude=None):
    """Process data for a specific view (top or bottom)/other views.
    
    Args:
        ground_truth_csv: Path to ground truth CSV file
        view_data: Dictionary mapping model names to prediction CSV paths
        view_name: Name of the view (e.g., 'top', 'bottom')
        keypoints_to_use: List of keypoints to use (default: all keypoints)
        ensemble_methods: List of ensemble method names for special processing
        sessions_to_ignore: List of session IDs (UUIDs) to exclude from processing
        frames_to_exclude: List of frame indices/identifiers to exclude from processing (e.g., [22928])
    """
    # Set default ensemble methods if not provided
    if ensemble_methods is None:
        ensemble_methods = ['ensemble_median', 'ensemble_mean', 'eks_singleview', 
                           'eks_multiview', 'eks_multiview_varinf', 'eks_multiview_10000', 
                           'eks_multiview_smooth']
    
    # Set default sessions to ignore if not provided
    if sessions_to_ignore is None:
        sessions_to_ignore = []
    
    # Set default frames to exclude if not provided
    if frames_to_exclude is None:
        frames_to_exclude = []
    
    # Initialize lists for all models
    all_pred_csv_list = []
    all_model_names_list = []
    
    # Reformat models
    for key, val in view_data.items():
        all_model_names_list.append(key)
        all_pred_csv_list.append(val)
    
    error_csv_list = [p.replace('.csv', '_pixel_error.csv') for p in all_pred_csv_list]
    print(f"Error CSVs: {error_csv_list}")
    # Load data
    df_pred_list = []
    df_error_list = []
    
    for pred_csv, error_csv in zip(all_pred_csv_list, error_csv_list):
        
        print(f"Loading predictions from {pred_csv}")
        # Load prediction data
        raw_df_pred = pd.read_csv(pred_csv, header=[0, 1, 2], index_col=0)
        raw_df_pred = io_utils.fix_empty_first_row(raw_df_pred)
        raw_df_pred.sort_index(inplace=True)
        
        # Filter out sessions to ignore
        if sessions_to_ignore:
            original_len = len(raw_df_pred)
            mask = ~raw_df_pred.index.astype(str).str.contains('|'.join(sessions_to_ignore), regex=True)
            raw_df_pred = raw_df_pred[mask]
            filtered_len = len(raw_df_pred)
            if original_len != filtered_len:
                print(f"Filtered out {original_len - filtered_len} rows from ignored sessions")
        
        # Filter out specific frames to exclude (using substring matching)
        if frames_to_exclude:
            original_len = len(raw_df_pred)
            frames_to_exclude_str = [str(f) for f in frames_to_exclude]
            pattern = '|'.join(frames_to_exclude_str)
            mask = ~raw_df_pred.index.astype(str).str.contains(pattern, regex=True)
            raw_df_pred = raw_df_pred[mask]
            filtered_len = len(raw_df_pred)
            if original_len != filtered_len:
                print(f"Filtered out {original_len - filtered_len} frames containing: {frames_to_exclude}")
        
        # Check if any ensemble method is in the prediction CSV
        is_ensemble = any(method in pred_csv for method in ensemble_methods)
        
        if is_ensemble:
            print(f"Processing ensemble data: {pred_csv}")
            # Filter for 'x', 'y', 'likelihood' columns
            df_pred = raw_df_pred.loc[
                :, raw_df_pred.columns.get_level_values(2).isin(['x', 'y', 'likelihood'])
            ]
        else:
            # General processing
            df_pred = raw_df_pred
        
        df_pred_list.append(df_pred)
        
        # Process the error file
        df = pd.read_csv(error_csv, header=[0], index_col=0).sort_index()
        
        # Filter out sessions to ignore from error file
        if sessions_to_ignore:
            original_len = len(df)
            mask = ~df.index.astype(str).str.contains('|'.join(sessions_to_ignore), regex=True)
            df = df[mask]
            filtered_len = len(df)
            if original_len != filtered_len:
                print(f"Filtered out {original_len - filtered_len} rows from error file for ignored sessions")
        
        # Filter out specific frames from error file (using substring matching)
        if frames_to_exclude:
            original_len = len(df)
            frames_to_exclude_str = [str(f) for f in frames_to_exclude]
            pattern = '|'.join(frames_to_exclude_str)
            mask = ~df.index.astype(str).str.contains(pattern, regex=True)
            df = df[mask]
            filtered_len = len(df)
            if original_len != filtered_len:
                print(f"Filtered out {original_len - filtered_len} frames from error file containing: {frames_to_exclude}")
        
        if 'set' in df.columns:
            df = df.drop(columns=['set'])
        df_error_list.append(df)
    
    # Load ground truth data
    df_gt = pd.read_csv(ground_truth_csv, header=[0, 1, 2], index_col=0)
    df_gt = io_utils.fix_empty_first_row(df_gt)
    column_structure = df_gt.loc[:, df_gt.columns.get_level_values(2).isin(['x', 'y', 'likelihood'])].columns
    df_gt.sort_index(inplace=True)
    
    # Filter out sessions to ignore from ground truth
    if sessions_to_ignore:
        original_len = len(df_gt)
        mask = ~df_gt.index.astype(str).str.contains('|'.join(sessions_to_ignore), regex=True)
        df_gt = df_gt[mask]
        filtered_len = len(df_gt)
        if original_len != filtered_len:
            print(f"Filtered out {original_len - filtered_len} rows from ground truth for ignored sessions")
    
    # Filter out specific frames from ground truth (using substring matching)
    if frames_to_exclude:
        original_len = len(df_gt)
        frames_to_exclude_str = [str(f) for f in frames_to_exclude]
        pattern = '|'.join(frames_to_exclude_str)
        mask = ~df_gt.index.astype(str).str.contains(pattern, regex=True)
        df_gt = df_gt[mask]
        filtered_len = len(df_gt)
        if original_len != filtered_len:
            print(f"Filtered out {original_len - filtered_len} frames from ground truth containing: {frames_to_exclude}")
    
    print("Number of models for ensemble std calculation:", len(all_pred_csv_list))
    print("Shape of predictions:", df_pred_list[0].shape)
    print("Column names:", df_pred_list[0].columns)
    
    # Get unique keypoints
    all_keypoints = list(dict.fromkeys(df_gt.columns.get_level_values(1)))
    print(f"Available Keypoints: {all_keypoints}")
    keypoints = keypoints_to_use if keypoints_to_use else all_keypoints
    print(f"Keypoints to process: {keypoints}")
    
    # Filter ground truth data for selected keypoints
    df_gt = df_gt.loc[:, df_gt.columns.get_level_values(1).isin(keypoints)]

    # First, compute the intersection of requested keypoints and available keypoints in ALL error files
    available_keypoints = set(keypoints)
    for df_error in df_error_list:
        available_keypoints = available_keypoints.intersection(set(df_error.columns))
    
    # Update keypoints to only include those available in all error files
    keypoints = [kp for kp in keypoints if kp in available_keypoints]
    print(f"Final keypoints after filtering: {keypoints}")
    
    if len(keypoints) == 0:
        raise ValueError(f"No valid keypoints found. Requested keypoints not available in error files. "
                        f"Available keypoints in error files: {df_error_list[0].columns.tolist() if df_error_list else 'N/A'}")
    
    # Now filter error dataframes to only include the final keypoints
    df_error_list_filtered = []
    for df_error in df_error_list:
        df_error_filtered = df_error[keypoints]
        df_error_list_filtered.append(df_error_filtered)
    
    # Also filter ground truth to match the final keypoints
    df_gt = df_gt.loc[:, df_gt.columns.get_level_values(1).isin(keypoints)]
    
    # Compute ensemble standard deviation using ALL models
    ens_stddev = compute_ensemble_stddev(df_gt, df_pred_list, keypoints)
    
    print(f"Models used for ensemble std: {all_model_names_list}")
    
    # Align indices across all dataframes (in case session filtering caused mismatches)
    common_index = df_gt.index
    for df_error in df_error_list_filtered:
        common_index = common_index.intersection(df_error.index)
    for df_pred in df_pred_list:
        common_index = common_index.intersection(df_pred.index)
    
    print(f"Common index length after alignment: {len(common_index)}")
    
    # Filter all dataframes to common index
    df_gt = df_gt.loc[common_index]
    df_error_list_filtered = [df.loc[common_index] for df in df_error_list_filtered]
    df_pred_list = [df.loc[common_index] for df in df_pred_list]
    
    # Recompute ensemble stddev with aligned data
    ens_stddev = compute_ensemble_stddev(df_gt, df_pred_list, keypoints)
    
    # Record pixel errors with ensemble variances
    df_w_vars = []
    for df_error, df_pred, model_name in zip(df_error_list_filtered, df_pred_list, all_model_names_list):
        total_pixel_error = df_error.sum().sum()
        print(f"Total pixel error for model {model_name} ({view_name}): {total_pixel_error}")
        
        for i, kp in enumerate(keypoints):
            index = [f'{idx}_{model_name}_{kp}' for idx in df_error.index]
            
            # Check if likelihood column exists for this keypoint
            likelihood_key = (df_pred.columns.get_level_values(0)[0], kp, 'likelihood')
            has_likelihood = likelihood_key in df_pred.columns
            
            # Create the base DataFrame structure
            df_row_data = {
                'pixel_error': df_error[kp].values,
                'ens-std': ens_stddev[:, i],
                'ens-std-prctile': [np.sum(ens_stddev < p) / ens_stddev.size for p in ens_stddev[:, i]],
                'ens-std-prctile-kp': [np.sum(ens_stddev[:, i] < p) / ens_stddev[:, i].size for p in ens_stddev[:, i]],
                'keypoint': kp,
                'model': model_name,
                'view': view_name
            }
            
            # Add likelihood column only if it exists
            if has_likelihood:
                df_row_data['likelihood'] = df_pred.loc[:, likelihood_key].values
            else:
                # Optional: add a default value or NaN if likelihood doesn't exist
                df_row_data['likelihood'] = np.nan
                print(f"Warning: No likelihood column found for model {model_name}, keypoint {kp}")
            
            df_w_vars.append(pd.DataFrame(df_row_data, index=index))
    
    if len(df_w_vars) == 0:
        raise ValueError(f"No data to concatenate for view {view_name}. "
                        f"Check that keypoints exist in the data and that there are valid rows after filtering.")
    
    return pd.concat(df_w_vars)


def prepare_line_data(df_w_vars, model_names_dict, std_vals=np.arange(0, 8, 0.2)):
    """
    Prepare line plot data grouped by views.
    
    Parameters:
    df_w_vars: DataFrame with the input data
    model_names_dict: Dictionary mapping views to lists of model names
    std_vals: Array of standard deviation values to evaluate
    """
    # Initialize n_points_dict for each view and model
    n_points_dict = {}
    for view, models in model_names_dict.items():
        print(f"View: {view}")
        n_points_dict[view] = {m: np.nan * np.zeros_like(std_vals) for m in models}
        print(f"Models: {models}")
        print(f"n_points_dict: {n_points_dict}")
    df_line2 = []
    
    for s, std in enumerate(std_vals):
        df_tmp_ = df_w_vars[df_w_vars['ens-std'] > std]
        
        
        for view, models in model_names_dict.items():
            for model_name in models:
                # Filter data for current view and model
                d = df_tmp_[(df_tmp_.model == model_name) & (df_tmp_['view'] == view)]
                if d.empty:
                    print(f"No data for view={view}, model_name={model_name}, std={std}")
                
                n_points = np.sum(~d['pixel_error'].isna())
                n_points_dict[view][model_name][s] = n_points
                
                index = []
                body_parts = []
                
                for row, k in zip(d.index, d['keypoint'].to_numpy()):
                    index.append(f'{row}_{model_name}_{s}_{k}_{0}_{view}')
                    body_parts.append(k)
                
                if len(d) > 0:
                    df_line2.append(pd.DataFrame({
                        'ens-std': std,
                        'model': model_name,
                        'mean': d.pixel_error.to_numpy(),
                        'n_points': n_points,
                        'body_part': body_parts,
                        'view': view  # Add view to output
                    }, index=index))
    
    # Combine all dataframes
    result = pd.concat(df_line2) if df_line2 else pd.DataFrame()
    
    # Add model2 column using major_model function
    if len(result) > 0:
        result['model2'] = result['model'].apply(major_model)
    
    return result, n_points_dict

# need to add here the model names for each view --> remember to try to make this automatic 
def major_model(row_data):
    """Extract major model name."""
    if 'ensemble_mean' in row_data or 'ensemble_median' in row_data or 'eks_singleview' in row_data or 'eks_multiview' in row_data or 'eks_multiview_varinf' in row_data:
        return row_data
    return '.'.join(row_data.split('.')[:-1])


def generate_model_configs(views):
    """
    Generate models_to_plot and color_mapping dictionaries for visualization
    
    Args:
        views (list): List of view names (e.g., ['A', 'B', 'C'])
        ensemble_methods (list): List of ensemble method names
        
    Returns:
        tuple: (models_to_plot, color_mapping) containing model names and colors
    """
    # Color definitions
    colors = {
        'base_model': plot_colors.get('rngs'),     
        'ensemble_mean': plot_colors.get('ensemble_mean'),  # OrangeRed
        'ensemble_median': plot_colors.get('ensemble_median'),
        'eks_singleview': plot_colors.get('eks_singleview'),   # LimeGreen
        'eks_multiview': plot_colors.get('eks_multiview'),     # HotPink
        'eks_multiview_varinf': plot_colors.get('eks_multiview_varinf'),  # DarkOrchid
        'eks_multiview_10000': plot_colors.get('eks_multiview'),
        'eks_multiview_smooth': plot_colors.get('eks_multiview_varinf'),  # DarkOrchid
        'eks_multiview_varinf_smooth': plot_colors.get('eks_multiview_varinf_smooth'),  # DarkOrchid
        
        
        'eks_multiview_postpred': plot_colors.get('eks_multiview_postpred'),
        'eks_multiview_varinf_postpred': plot_colors.get('eks_multiview_varinf_postpred'),
        'eks_multiview_concat': plot_colors.get('eks_multiview_concat'),
        'eks_multiview_varinf_concat': plot_colors.get('eks_multiview_varinf_concat'),
        'eks_multiview_no_object': plot_colors.get('eks_multiview_no_object'),
        'eks_multiview_no_object_100': plot_colors.get('eks_multiview_no_object_100'),
        'eks_multiview_varinf_no_object': plot_colors.get('eks_multiview_varinf_no_object'),
        
        'ens_med_anipose': plot_colors.get('ens_med_anipose'),
        # 'ens_med_anipose': plot_colors.get('orange'),
        # 'ens_med_anipose': 'orange',
        'eks_multiview_videos_new': plot_colors.get('eks_multiview_videos_new'),
        
    }
    
    # Generate configurations
    models_to_plot = {}
    color_mapping = {}
    
    for view in views:
        # Base model name for this view
        base_model = f'v0_5k_{view}_128'
        
        # List of models to plot for this view
        # models_to_plot[view] = [
        #     base_model,  # Base model
        #     *[f'{base_model}_{method}.0' for method in ensemble_methods]  # Ensemble methods
        # ]
        
        # Color mapping for this view
        view_colors = {
            base_model: colors['base_model'],  # Base model color
            f'{base_model}.0': colors['base_model'],  # Individual models
            f'{base_model}.1': colors['base_model'],
            f'{base_model}.2': colors['base_model'],
            f'{base_model}.3': colors['base_model'],
            f'{base_model}.4': colors['base_model']
        }
        
        # # Add ensemble method colors
        # for method in ensemble_methods:
        #     model_name = f'{base_model}_{method}.0'
        #     view_colors[model_name] = colors[method]
        
        color_mapping[view] = view_colors
    
    return models_to_plot, color_mapping

def save_plots(fig, output_dir, filename="pixel_error_vs_ensemble_std"):
    """
    Save plots to a specified directory
    Args:
        fig: matplotlib figure object
        output_dir (str): Directory path where plots should be saved
        filename (str): Name for the saved file
    """

    os.makedirs(output_dir, exist_ok=True)
    
    # Add file extension to the filename
    filepath = os.path.join(output_dir, f"{filename}.pdf")
    
    # Save the figure
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")


def plot_comparison(df_line2, n_points_dict, models_to_plot, color_mapping, percentiles=[95, 50, 5], figsize = (5,7), dataset_name = 'fly-anipose',std_vals = np.arange(0, 20, 0.5), ylim=None):
    """
    Create a single plot that combines data across views for the same model type.
    
    Parameters:
    df_line2: DataFrame with the line plot data
    n_points_dict: Dictionary with points count information
    models_to_plot: Dictionary mapping views to lists of model names
    color_mapping: Dictionary mapping views and models to colors
    percentiles: Percentile values to show on the plot
    """
    print("\nStarting plot_comparison with view aggregation...")
    # std_vals = np.arange(0, 8, 0.2)
    # std_vals = np.arange(0, 20, 0.5)
    std_vals = std_vals
    
    # Create a single figure
    # for paper 
    # fig, ax = plt.subplots(figsize=(2.7,2.7)) # for paper was (2.7,4), in general I use 5,7 - fo figure 4
    # fig, ax = plt.subplots(figsize=(2.7,4)) # for paper was (2.7,4), in general I use 5,7
    # fig, ax = plt.subplots(figsize=(2.7,3.2)) # for paper was (2.7,4), in general I use 5,7
    # fig, ax = plt.subplots(figsize=(3.2,3.2)) # for paper was (2.7,4), in general I use 5,7
    fig, ax = plt.subplots(figsize=figsize) # for paper was (2.7,4), in general I use 5,7
    
    # def normalize_model_name(model_name):
    #     """Remove view-specific part (top/bot) from model name."""
    #     parts = model_name.split('_')
    #     if len(parts) < 4:
    #         return model_name
    #     normalized = parts[0:2] + parts[3:]
    #     return '_'.join(normalized)

    def normalize_model_name(model_name):
        """Remove view-specific part from model name."""
        # Handle the specific case for human36m-crop dataset
        if 'ca_' in model_name:
            # For ca_01, ca_02, etc., remove the ca_XX part
            import re
            # Remove ca_XX pattern (where XX is digits)
            normalized = re.sub(r'_ca_\d+_', '_', model_name)
            return normalized
        
        # Original logic for other datasets
        parts = model_name.split('_')
        if len(parts) < 4:
            return model_name
        normalized = parts[0:2] + parts[3:]
        return '_'.join(normalized)
    
    # Add normalized model name to the dataframe
    df_line2['normalized_model'] = df_line2['model2'].apply(normalize_model_name)
    
    # Get unique normalized model names
    unique_models = df_line2['normalized_model'].unique()
    print(f"Unique normalized models: {unique_models}")
    
        # TODO: add the new models here
    # Create unified color mapping for normalized models
    model_colors = {

        'v0_5k_128': '#7f7f7f',   # Middle Gray
        'v1_5k_128': '#d62728',   # Brick Red
        'v2_5k_128': 'gray',   # Muted Blue
        'v3_5k_128': 'orange',   # Safety Orange
        'v4_5k_128': 'orange' , # strong yellow    
        'v5_5k_128': '#9467bd',   # Muted Purple
        # 'v6_5k_128': '#e377c2',  #'#8c564b',   # Chestnut Brown #8c564b --> this is for the case we don't compare 3d aug and dlc
        'v6_5k_128': 'orange',  #'#8c564b',   # Chestnut Brown #8c564b
        'v7_5k_128': '#e377c2',   # Raspberry Pink
        'v8_5k_128': 'lightseagreen',   # Olive Green
        'v9_5k_128': 'green',   # Blue-Green Cyan
        'v10_5k_128': 'brown',  # Light Blue
        'v11_5k_128': '#ffbb78',  # Light Orange
        'v12_5k_128': '#98df8a',  # Light Green
        'v13_5k_128': '#ff9896',  # Light Red
        'v14_5k_128': 'pink',
        'v16_5k_128': 'green',
        'v18_5k_128': 'purple', # was blue
        'v19_5k_128': 'brown',
        'v20_5k_128': 'lightgreen',
        # 'v21_5k_128': 'lightred',
        'v22_5k_128': 'blue',
        # 'v23_5k_128': 'lightsalmon',
        'v23_5k_128': 'purple',
        'v24_5k_128': 'blue',
        'v25_5k_128': '#ff9896',
        'v26_5k_128': 'orange',
        'v27_5k_128': 'purple',
        'v28_5k_128': 'lightseagreen',
        # 'v29_5k_128': 'green',
        # 'v30_5k_128': 'green',
        'v29_5k_128': 'blue',
        'v30_5k_128': 'blue',
        'v31_5k_128': 'lightcoral',
        'v32_5k_128': 'purple',
        'v33_5k_128': 'blue',
        'v34_5k_128': 'orange',

        'v36_5k_128': 'steelblue',
        # 'v37_5k_128': 'brown',
        'v37_5k_128': 'orange',
        'v38_5k_128': 'brown',
        # 'v29_5k_128': 'green',
        # 'v30_5k_128': 'green',
        'v39_5k_128': 'hotpink',
        'v40_5k_128': 'gray',
        'v55_5k_128': 'blue',
        'v57_5k_128': 'hotpink',
        'v58_5k_128': 'orange',
        'v59_5k_128': 'orange',
        'v60_5k_128': 'green',
        'v61_5k_128': 'blue',
        'v62_5k_128': 'hotpink',
        'v63_5k_128': 'orange',
        'v80_5k_128': 'green',
        'v86_5k_128': 'brown',
        'v88_5k_128': 'steelblue',


        'v90_5k_128': 'red',
        'v91_5k_128': 'blue',
        'v92_5k_128': 'purple',
        'v93_5k_128': 'brown',
        'v94_5k_128': 'green',
        'v95_5k_128': 'lightseagreen',
        'v96_5k_128': 'yellow',
        'v97_5k_128': 'brown',
        'v98_5k_128': 'lightcoral',


        'v99_5k_128': 'steelblue',
        'v100_5k_128': 'gray',
        
        'v101_5k_128': 'purple',
        'v102_5k_128': 'red',
        'v103_5k_128': 'green',
        'v104_5k_128': 'brown',

        'v200_5k_128': 'steelblue',
        'v201_5k_128': 'gray',
        
        
        'v0_5k_128_non_linear_eks': 'green',
        'v0_5k_128_non_linear_eks_varinf': 'limegreen',
        'v0_5k_128_ensemble_median.0': 'orange',
        'v0_5k_128_anipose_ens_med': 'orange',
        'v0_5k_128_anipose_ens_med_4views': 'pink',
        'v0_5k_128_eks_singleview.0': 'blue',
        'v0_5k_128_eks_multiview.0': 'green',
        'v0_5k_128_eks_multiview_postpred.0': '#D8BFD8',
        'v0_5k_128_eks_multiview_concat.0': 'green',
        'v0_5k_128_eks_multiview_varinf.0': 'limegreen',
        # 'v0_5k_128_eks_multiview_varinf.0': 'green',# this is only for the appendix figure files - for fly and chickadee
        'v0_5k_128_eks_multiview_varinf_postpred.0': 'limegreen',
        'v0_5k_128_eks_multiview_varinf_concat.0': 'limegreen',
        'v0_5k_128_eks_multiview_videos_new.0': 'purple',
        'v0_5k_128_eks_multiview_resnet50.0': 'green',
        'v0_5k_128_ensemble_median_resnet50.0': 'red',
        'v0_5k_128_resnet50_anipose.0': 'gray',
        'v0_5k_128_dlc_anipose.0': 'orange',
        'v0_5k_128_ensemble_median_2d_filters': 'orange',
        'v0_5k_128_ensemble_median_temporal': 'brown',
        'v0_5k_128_ensemble_median_spatial': 'pink',
        'v0_5k_128_ensemble_median_spatiotemporal': 'blue',
        'v0_5k_128_ensemble_median_2d_filters_updated': 'lightcoral',
        'v0_5k_128_non_linear_eks_geometric': 'darkgreen',
        'v0_5k_128_ensemble_median_2d_filters.0': 'orange',
        'v0_5k_128_ensemble_median_temporal.0': 'brown',
        'v0_5k_128_ensemble_median_spatial.0': 'pink',
        'v0_5k_128_ensemble_median_spatiotemporal.0': 'blue',
        'v0_5k_128_ensemble_median_2d_filters_updated.0': 'lightcoral',
        'v0_5k_128_non_linear_eks_geometric.0': 'darkgreen',
    }



    custom_labels = {
            'v0_5k_128': 'SV_resnet50_3d',
            'v1_5k_128': 'SV_resnet50_dlc-mv',
            'v2_5k_128': 'SV_resnet50_dlc',
            'v3_5k_128_dlc': 'DLC',
            
            'v4_5k_128': 'MVT_imagenet', 
            'v5_5k_128': 'MVT-dino-dlc-patch-mask',
            'v6_5k_128': 'MVT-dino-3d',
            'v7_5k_128': 'MVT-dino-dlc',
            'v8_5k_128': 'SV_vits_dino',
            'v9_5k_128': 'MVT_pretrained',
            'v10_5k_128': 'MVT_beast_pretrained',
            
            'v11_5k_128': 'MVT_vits_dinov2',
            'v12_5k_128': 'MVT_vitb_dinov2',
            'v13_5k_128': 'MVT_vitb_dino',

            'v14_5k_128': 'SV_vitb_dino',
            'v15_5k_128': 'SV_vits_dinov2',
            'v16_5k_128': 'SV_vitb_dinov2',
    
            # the real ones - the above is for testing 
            'v18_5k_128': 'MVT_3d_loss(0.3)',
            'v19_5k_128': 'MVT_3d_loss(0.5)',
            'v20_5k_128': 'MVT_3d_loss(1)',
            'v21_5k_128': 'MVT_3d_loss(2)',

            'v22_5k_128': 'MVT_3d_loss_view_mask',
            'v23_5k_128': 'LP3D',
            # 'v23_5k_128': 'MVT_patch_mask_3d',
            'v24_5k_128': 'MVT_view_mask',
            'v25_5k_128': 'MVT_patch_mask',
            'v26_5k_128': 'DANNCE',
            'v27_5k_128': 'MVT_dlc_patch_mask',

            'v80_5k_128': 'MVT_dlc_patch_mask_dinov2',
            'v86_5k_128': 'SV_vits_dino_3d_aug',
            'v87_5k_128': 'MVT_dlc_no_rotation',
            'v88_5k_128': 'MVT_3d_aug_rotation',
            'v89_5k_128': 'MVT_3d_aug_rotation_similarity_transform',
            
            'v33_5k_128': 'MVT_3d_loss_distil (eks_sv,3218,rand)',
            'v34_5k_128': 'MVT_3d_loss_distil (eks_sv,3218,var)',
            'v30_5k_128': 'SV_vitb_dino',
            'v31_5k_128': 'SV_vitb_imagenet',
            'v32_5k_128': 'SV_vitb_sam',
            'v35_5k_128': 'anipose_reprojected',
            'v55_5k_128': 'Anipose + MVT',
            
            
            'v36_5k_128': 'MVT distil EKS(1000)',
            'v37_5k_128': 'MVT distil EKS(3000)',
            'v38_5k_128': 'MVT distil EKS(3000,rand)',
            'v39_5k_128': 'MVT distil Med(3000)',
            'v40_5k_128': 'MVT distil Med(3000,rand)',
            'v57_5k_128': 'anipose + distil',

            # unsupervised losses 
            'v59_5k_128': 'MVT + uns_loss (start 50)',
            'v60_5k_128': 'MVT + uns_loss (start 0)',
            'v61_5k_128': 'MVT + uns_loss (start 0, patch)',


            'v90_5k_128': 'Try',
            # 'v91_5k_128': 'MVT 3d loss heatmap (1)',
            'v91_5k_128': 'MVT 3d loss heatmap (0_1) + patch mask',
            'v92_5k_128': 'MVT 3d loss heatmap (2)',
            # 'v93_5k_128': 'MVT 3d loss heatmap (3)',
            'v93_5k_128': 'MVT 3d loss pairwise (3)',
            'v94_5k_128': 'MVT 3d loss heatmap (0.5)',
            'v95_5k_128': 'MVT 3d loss heatmap (0.1)',
            # 'v96_5k_128': 'MVT 3d loss heatmap (0.01)',
            'v96_5k_128': 'MVT 3d aug change scale transform heatmap loss (0.1)',
            'v97_5k_128': 'MVT 3d loss heatmap (5)',
            'v98_5k_128': 'MVT 3d loss heatmap (0.05)',
            
            'v99_5k_128': 'MVT 3d aug mvt_nature',
            'v100_5k_128': 'MVT 3d aug mvt_nature clamp_Matt',

            'v101_5k_128': 'MVT 3d aug patch masking',
            'v102_5k_128': 'MVT 3d loss patch mask',
            'v103_5k_128': 'MVT 3d loss supervised (4) patch masking (16,32) later freeze',
            'v104_5k_128': 'MVT 3d loss heatmap (0_05) patch masking (16,32) later freeze',
            'v105_5k_128': 'MVT 3d loss supervised (4) patch masking',
            'v110_5k_128': 'MVT distilled (3325)',
            'v111_5k_128': 'MVT distilled (1235)',

            'v200_5k_128': 'MVT (vitb_dinov3)',
            'v201_5k_128': 'LP3D (vitb_dinov3)',

            

            # 'v5_5k_128': 'Model Try',
            'v0_5k_128_ensemble_median.0': 'Ensemble Median',
            'v0_5k_128_eks_singleview.0': 'Single-view EKS',
            'v0_5k_128_eks_multiview.0': 'Multi-view EKS',
            'v0_5k_128_eks_multiview_postpred.0': 'Multi-view EKS (paper)',
            'v0_5k_128_eks_multiview_varinf.0': 'Multi-view EKS (Var Inflated)',
            'v0_5k_128_eks_multiview_varinf_concat.0': 'Multi-view EKS (Var Inflated)',
            'v0_5k_128_eks_multiview_concat.0': 'Multi-view EKS',
            'v0_5k_128_non_linear_eks': 'Non-linear EKS',
            # 'v0_5k_128_non_linear_eks_varinf': 'Non-linear EKS (Var Inflated)',
            'v0_5k_128_non_linear_eks_varinf': 'Non-lin_Varinf',
            'v0_5k_128_anipose_ens_med': 'Ensemble Median Anipose',
            'v0_5k_128_resnet50_anipose.0': 'Resnet50 + Anipose',
        
            'v0_5k_128_anipose_ens_med': 'Med + Ani',
            'v0_5k_128_anipose_ens_med_4views': 'Med + Ani 4 views',
            

            'v0_5k_128_eks_multiview_no_object.0': 'Multi-view EKS Without PCA Object',
            'v0_5k_128_eks_multiview_smooth.0': 'EKS Multi-View Smooth',
            'v0_5k_128_eks_multiview_varinf_smooth.0': 'EKS Multi-View Smooth VarInf (Var Inflated)',
            'v0_5k_128_eks_multiview_new.0': 'EKS Multi-View PCA Object',
            'v0_5k_128_ens_med_anipose.0': 'Ensemble Median Anipose',
            'v0_5k_128_eks_multiview_videos_new.0': 'mveks extracted from full videos',

            'v0_5k_128_eks_multiview_resnet50.0': 'EKS Multi-View Resnet50',
            'v0_5k_128_ensemble_median_resnet50.0': 'Ensemble Median Resnet50',
            'v0_5k_128_dlc_anipose.0': 'DLC (Median) + Triangulation',
            'v0_5k_128_ensemble_median_2d_filters.0': 'Ensemble Median 2D Filters',
            'v0_5k_128_ensemble_median_temporal.0': 'Ensemble Median Temporal',
            'v0_5k_128_ensemble_median_spatial.0': 'Ensemble Median Spatial',
            'v0_5k_128_ensemble_median_spatiotemporal.0': 'Ensemble Median Spatiotemporal',
            'v0_5k_128_ensemble_median_2d_filters_updated.0': 'Ensemble Median All 2D Filters',
            'v0_5k_128_non_linear_eks_geometric.0': 'Non-linear EKS Geometric',
        }
    
    if dataset_name == 'mirror-mouse-separate':
        custom_labels = {
            'v2_5k_128': 'SV_resnet50_dlc',
            'v3_5k_128_dlc': 'DLC',
            'v7_5k_128': 'MVT-dino-dlc',
            'v8_5k_128': 'SV_vits_dino',


            'v11_5k_128': 'MVT_vits_dinov2',
            'v12_5k_128': 'MVT_vitb_dinov2',
            'v13_5k_128': 'MVT_vitb_dino',

            'v14_5k_128': 'SV_vitb_dino',
            'v15_5k_128': 'SV_vits_dinov2',
            'v16_5k_128': 'SV_vitb_dinov2',



            'v22_5k_128': 'MVT_dlc_view_mask',
            'v23_5k_128': 'LP3D',
            'v26_5k_128': 'MVT_distil_1239_patch_mask',
            'v28_5k_128': 'MVT_distil_3239_patch_mask',
            'v30_5k_128': 'SV_vitb_dino',
            'v31_5k_128': 'SV_vitb_imagenet',
            'v32_5k_128': 'SV_vitb_sam',


            'v80_5k_128': 'MVT_dlc_patch_mask_dinov2',



            'v59_5k_128': 'MVT + uns_loss (start 50)',
            'v60_5k_128': 'MVT + uns_loss (start 0)',
            'v61_5k_128': 'MVT + uns_loss (start 0, patch)',

            'v36_5k_128': 'MVT distil EKS(1000)',
            'v37_5k_128': 'MVT distil EKS(3000)',
            'v38_5k_128': 'MVT distil EKS(3000,rand)',
            'v39_5k_128': 'MVT distil Med(3000)',
            'v40_5k_128': 'MVT distil Med(3000,rand)',

            'v0_5k_128_ensemble_median.0': 'Ensemble Median',
            'v0_5k_128_eks_singleview.0': 'Single-view EKS',
            'v0_5k_128_eks_multiview.0': 'Multi-view EKS',
            'v0_5k_128_eks_multiview_postpred.0': 'Multi-view EKS (paper)',
            'v0_5k_128_eks_multiview_varinf.0': 'Multi-view EKS (Var Inflated)',
            'v0_5k_128_eks_multiview_varinf_concat.0': 'Multi-view EKS (Var Inflated)',
            'v0_5k_128_eks_multiview_concat.0': 'Multi-view EKS',

            'v0_5k_128_eks_multiview_no_object.0': 'Multi-view EKS Without PCA Object',
            'v0_5k_128_eks_multiview_smooth.0': 'EKS Multi-View Smooth',
            'v0_5k_128_eks_multiview_varinf_smooth.0': 'EKS Multi-View Smooth VarInf (Var Inflated)',
            'v0_5k_128_eks_multiview_new.0': 'EKS Multi-View PCA Object',
            'v0_5k_128_ens_med_anipose.0': 'Ensemble Median Anipose',
            'v0_5k_128_anipose_ens_med_4views': 'Ensemble Median Anipose 4 views',
            'v0_5k_128_eks_multiview_videos_new.0': 'mveks extracted from full videos',
        }
        


    # Plot each normalized model
    for model in unique_models:
        # ax.set_ylim([0, 45])
        # Get all data points for this normalized model
        model_data = df_line2[df_line2['normalized_model'] == model]
        print(df_line2[(df_line2['ens-std']==0) & (df_line2['model2']==model)])
        
        # --- Add print statement for standard error for each model ---
        # We'll calculate standard error of 'mean' for this model_data if available
        if len(model_data) > 0:
            se = model_data[model_data['ens-std']==0]['mean'].std() / np.sqrt(len(model_data[model_data['ens-std']==0]))
            print(f"Standard error for model {model}: {se}")
            print(f"Plotting normalized model: {model} with {len(model_data[model_data['ens-std']==0])} data points")
            print(f" the mean is {model_data[model_data['ens-std']==0]['mean'].mean()}")
            
            # Get the display label (custom or default)
            display_label = custom_labels.get(model, model)
            
            # Plot aggregated data
            sns.lineplot(
                x='ens-std', 
                y='mean', 
                data=model_data, 
                label=display_label,
                color=model_colors.get(model, 'gray'), 
                ax=ax, 
                errorbar='se', 
                linewidth=2
            )
    
    # Set y-axis limit if provided
    if ylim is not None:
        ax.set_ylim(ylim)
    
    # Style the plot
    # ax.set_title(f'Combined View - {dataset_name} (n_hand_labels = {n_hand_labels})', fontsize=6, pad=10)
    ax.set_title(f'{dataset_name}', fontsize=6, pad=10)
    ax.set_ylabel('Pixel error', fontsize=6)
    ax.set_xlabel('Ensemble std dev', fontsize=6)
    ax.tick_params(axis='both', which='major', labelsize=6)
    
    # Add percentile lines if data is available
    if n_points_dict:
        # Combine all n_points across views for the same model type
        combined_n_points = np.zeros_like(std_vals)
        count = 0
        
        # Sum up points across all views for a consistent calculation
        for view in n_points_dict:
            for model in n_points_dict[view]:
                if '.0' in model and not np.isnan(n_points_dict[view][model]).all():
                    # Skip models with only NaN values
                    if np.all(np.isnan(n_points_dict[view][model])):
                        continue
                        
                    # Add valid data points (replace NaN with 0)
                    combined_n_points += np.nan_to_num(n_points_dict[view][model], 0)
                    count += 1
        
        if count > 0:
            # We don't average for percentile calculation
            # We want to know what percentage of ALL points are included
            print(f"Computing percentiles using combined data from {count} models")
            print(f"Total data points: {np.sum(combined_n_points)}")
            
            # Show the actual data distribution
            percentage_of_data = combined_n_points / np.sum(combined_n_points) * 100
            cumulative_percentage = np.cumsum(percentage_of_data)
            print(f"Data distribution by std value: {np.round(percentage_of_data[:10], 2)}...")
            print(f"Cumulative percentage: {np.round(cumulative_percentage[:10], 2)}...")
            
            # Calculate percentiles
            vals, prctiles = compute_percentiles(
                arr=combined_n_points,
                std_vals=std_vals,
                percentiles=percentiles,  # [95, 50, 5] by default
            )
            
            # Add vertical lines at the percentile positions
            for p, v in zip(prctiles, vals):
                ax.axvline(v, ymax=0.95, linestyle='--', linewidth=1, color='black', alpha=0.5)
                ax.text(v, ax.get_ylim()[1], f'{p}%', ha='left', va='top', fontsize=6, rotation=90)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Add grid and legend
    ax.set_ylabel('Pixel error', fontsize=6, labelpad=1)
    ax.set_xlabel('Ensemble std dev', fontsize=6, labelpad=1)
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6, frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    

    
    return fig