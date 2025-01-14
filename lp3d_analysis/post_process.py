import os
import pandas as pd
import numpy as np
from typing import Optional, Union


from omegaconf import DictConfig
from typing import List, Literal
from pathlib import Path

from lightning_pose.utils.scripts import (
    compute_metrics,
)

from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam
from eks.multicam_smoother import ensemble_kalman_smoother_multicam


#TODO
# 1. Implement post_process_ensemble_multiview

def process_predictions(pred_file: str, column_structure=None):
    """
    Process predictions from a CSV file and return relevant data structures.
    
    Args:
        pred_file (str): Path to the prediction CSV file
        column_structure: Existing column structure (if any)
        
    Returns:
        tuple: (column_structure, array_data, numeric_cols, keypoint_names, df_index)
               Returns (None, None, None, None, None) if file doesn't exist
    """
    keypoint_names = []  # Initialize keypoint_names 
    
    if not os.path.exists(pred_file):
        print(f"Warning: Could not find predictions file: {pred_file}")
        return None, None, None, None, None
        
    df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
    
    if column_structure is None:
        column_structure = df.loc[:, df.columns.get_level_values(2).isin(['x', 'y', 'likelihood'])].columns
        keypoint_names = list(dict.fromkeys(column_structure.get_level_values(1)))
        print(f'Keypoint names are {keypoint_names}')
    
    numeric_cols = df.loc[:, column_structure]
    array_data = numeric_cols.to_numpy()
    
    return column_structure, array_data, numeric_cols, keypoint_names, df.index

def post_process_ensemble(
    cfg_lp: DictConfig,
    results_dir: str,
    model_type: str,
    n_labels: int,
    seed_range: tuple[int, int],
    views: list[str], 
    mode: Literal['ensemble_mean', 'ensemble_median', 'eks_singleview', 'eks_multiview'],
    inference_dirs: List[str],
    overwrite: bool,
) -> None:
    """
    Post-processes ensemble predictions from multiple model seeds by aggregating their outputs.
    
    Args:
        cfg_lp (DictConfig): Configuration dictionary containing data and processing parameters
        results_dir (str): Base directory containing the model results
        model_type (str): Type of model used for predictions
        n_labels (int): Number of labels/keypoints in the model
        seed_range (tuple[int, int]): Range of seeds (start, end) to include in ensemble
        views (list[str]): List of camera views to process
        mode (Literal['ensemble_mean', 'ensemble_median', 'eks_singleview']): Aggregation method:
            - ensemble_mean: Takes mean across all seed predictions
            - ensemble_median: Takes median across all seed predictions
            - eks_singleview: Applies Extended Kalman Smoothing to single view predictions
        inference_dirs (List[str]): List of inference directory names to process
        overwrite (bool): Whether to overwrite existing processed files
    
    Returns:
        None: Results are saved to disk in the specified output directory
        
    The function:
    1. Creates an ensemble directory structure based on model type and seeds
    2. For each inference directory and view:
        - Loads predictions from all seed models
        - Stacks predictions into a single array
        - Applies the specified aggregation method (mean/median/eks)
        - Saves processed results to CSV
        - Computes metrics on the aggregated predictions
    """

    # setup directories 
    base_dir = os.path.dirname(results_dir)
    ensemble_dir = os.path.join(
        base_dir,
        f"{model_type}_{n_labels}_{seed_range[0]}-{seed_range[1]}"
    )
    seed_dirs = [
            os.path.join(base_dir, f"{model_type}_{n_labels}_{seed}")
            for seed in range(seed_range[0], seed_range[1] + 1)
    ]

    for inference_dir in inference_dirs:
        output_dir = os.path.join(ensemble_dir, mode, inference_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dirs} " )

        if inference_dir == 'videos-for-each-labeled-frame':
            if mode == 'eks_multiview':
                # collect predictions from all seeds for all views 
                markers_list_by_view = []
                column_structures = {}
                df_indices = {}
                keypoint_names = [] 
                for view in views:
                    print(f"\nProcessing view {view}")
                    view_markers = []
                    for seed_dir in seed_dirs:
                        pred_file = os.path.join(
                            seed_dir,
                            inference_dir,
                            f'predictions_{view}_new.csv'
                        )
                        print(f"Reading predictions from {pred_file}")
                        column_structure, array_data, numeric_cols, keypoints, df_index = process_predictions(
                            pred_file, 
                            column_structures.get(view)
                        )
                        if array_data is not None:
                            print(f"Found valid data with shape {array_data.shape}")
                            view_markers.append(numeric_cols)
                            column_structures[view] = column_structure
                            df_indices[view] = df_index
                            keypoint_names = keypoints if keypoints else keypoint_names
                        else:
                            print(f"No valid data found in {pred_file}")
                    
                    if view_markers:  # Only append if we have valid markers for this view
                        print(f"Collected {len(view_markers)} prediction sets for view {view}")
                        markers_list_by_view.append(view_markers)
                    else:
                        print(f"No valid predictions found for view {view}")
                
                if not markers_list_by_view:
                    print("No valid predictions found for any view")
                    continue
                
                print(f"\nProcessing {len(markers_list_by_view)} views with data")
                print(f"Found {len(keypoint_names)} keypoints: {keypoint_names}")

                # Process multi-view data
                results_arrays, results_dfs = run_eks_multiview(
                    markers_list_by_view=markers_list_by_view,
                    keypoint_names=keypoint_names,
                    views=views,
                )
     
                #Save results for each view
                for view, result_df in results_dfs.items():
                    # Get column structure from smoothed dataframe
                    column_structure = result_df.columns[
                        result_df.columns.get_level_values(2).isin([
                            'x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
                            'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var'
                        ])
                    ]
                    
                    # Create DataFrame with proper structure
                    result_df = pd.DataFrame(
                        data=results_arrays[view],
                        index=df_indices[view],
                        columns=column_structure
                    )
                    
                    # Add train set indicator
                    result_df.loc[:,("set", "", "")] = "train"
                    
                    preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
                    result_df.to_csv(preds_file)
                    print(f"Saved multi-view EKS predictions for {view} to {preds_file}")
                    
                    # Update cfg_lp for metrics computation
                    cfg_lp_view = cfg_lp.copy()
                    cfg_lp_view.data.csv_file = [f'CollectedData_{view}_new.csv']
                    cfg_lp_view.data.view_names = [view]
                    
                    try:
                        compute_metrics(cfg=cfg_lp_view, preds_file=preds_file, data_module=None)
                        print(f"Successfully computed metrics for {preds_file}")
                    except Exception as e:
                        print(f"Error computing metrics\n{e}")

            else:
                new_predictions_files = []   
                for view in views: 
                    stacked_arrays = []
                    stacked_dfs = []
                    column_structure = None
                    keypoint_names = []
                    
                    for seed_dir in seed_dirs:
                        pred_file = os.path.join(
                            seed_dir,
                            inference_dir,
                            f'predictions_{view}_new.csv'
                        )
                        column_structure, array_data, numeric_cols, keypoints, df_index = process_predictions(pred_file, column_structure)
                        if array_data is not None:
                            stacked_arrays.append(array_data)
                            stacked_dfs.append(numeric_cols)
                            keypoint_names = keypoints if keypoints else keypoint_names

                    # Stack all arrays along the third dimension
                    stacked_arrays = np.stack(stacked_arrays, axis=-1)

                    # Compute the mean/median along the third dimension
                    if mode == 'ensemble_mean':
                        aggregated_array = np.nanmean(stacked_arrays, axis=-1)
                    elif mode == 'ensemble_median':
                        aggregated_array = np.nanmedian(stacked_arrays, axis=-1)
                    elif mode == 'eks_singleview':
                        aggregated_array, smoothed_df = run_eks_singleview(
                            markers_list= stacked_dfs,
                            keypoint_names=keypoint_names
                        )
                        # filtered_df = smoothed_df.loc[
                        #     : , smoothed_df.columns.get_level_values(2).isin(['x','y','likelihood'])
                        # ]
                        # # Update the aggregated_array to match the filtered columns
                        # aggregated_array = filtered_df.to_numpy()
                        # Dynamically update column structure based on smoothed_df
                        column_structure = smoothed_df.columns[
                            smoothed_df.columns.get_level_values(2).isin(
                                ['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median','x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var']
                            )
                        ]          

                    else:
                        print(f"Invalid mode: {mode}")
                        continue

                    # Create a new DataFrame with the aggregated data
                    result_df = pd.DataFrame(
                        data=aggregated_array,
                        index= df_index, #df.index,
                        columns=column_structure
                    )

                    result_df.loc[:,("set", "", "")] = "train"

                    preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
                    result_df.to_csv(preds_file)
                    new_predictions_files.append(preds_file)
                    print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")

                    # Update cfg_lp for each view specifically
                    cfg_lp_view = cfg_lp.copy()
                    # Dynamically create the CSV filename based on view
                    csv_filename = f'CollectedData_{view}_new.csv'
                    cfg_lp_view.data.csv_file = [csv_filename]
                    cfg_lp_view.data.view_names = [view]

                    try:
                        compute_metrics(cfg=cfg_lp_view, preds_file=preds_file, data_module=None)
                        print(f"Successfully computed metrics for {preds_file}")
                    except Exception as e:
                        print(f"Error computing metrics\n{e}")

        else:
            if mode == 'eks_multiview':
                markers_list_by_view = []
                column_structures = {}
                df_indices = {}
                keypoint_names = []

                for view in views:
                    view_markers = []
                    for seed_dir in seed_dirs:
                        base_files = os.listdir(os.path.join(seed_dir, inference_dir))
                        # Check if any files include `{view}_` in a case-sensitive manner
                        csv_files = [f for f in base_files if f.endswith('.csv') and f"{view}_" in f]
                        pred_file = os.path.join(seed_dir, inference_dir, csv_files[0])
                        column_structure, array_data, numeric_cols, keypoints, df_index = process_predictions(
                            pred_file, 
                            column_structures.get(view)
                        )
                        if array_data is not None:
                            view_markers.append(numeric_cols)
                            column_structures[view] = column_structure
                            df_indices[view] = df_index
                            keypoint_names = keypoints if keypoints else keypoint_names

                    if view_markers:
                        markers_list_by_view.append(view_markers)

                if not markers_list_by_view:
                    print("No valid predictions found for any view")
                    continue

                # Process multi-view data
                results_arrays, results_dfs = run_eks_multiview(
                    markers_list_by_view=markers_list_by_view,
                    keypoint_names=keypoint_names,
                    views=views,
                )

                # Save results for each view
                for view in views:
                    result_df = results_dfs[view]
                    # Get column structure from smoothed dataframe
                    column_structure = result_df.columns[
                        result_df.columns.get_level_values(2).isin([
                            'x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
                            'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var'
                        ])
                    ]

                    # Create DataFrame with proper structure
                    base_files = os.listdir(os.path.join(seed_dirs[0], inference_dir))
                    csv_files = [f for f in base_files if f.endswith('.csv') and f"{view}_" in f]
                    
                    if csv_files:
                        # Use the same base filename pattern that was found in the input directory
                        base_name = csv_files[0]
                        result_df = pd.DataFrame(
                            data=results_arrays[view],
                            index=df_indices[view],
                            columns=column_structure
                        )
                        
                        preds_file = os.path.join(output_dir, base_name)
                        result_df.to_csv(preds_file)
                        print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")
            
            else:
                for view in views: 
                    stacked_arrays = []
                    stacked_dfs = []
                    column_structure = None
                    keypoint_names = []

                    for seed_dir in seed_dirs:
                        base_files = os.listdir(os.path.join(seed_dir, inference_dir))
                        print(f"base_files: {base_files}")
                        # Check if any files match the `view`
                        #csv_files = [f for f in base_files if f.endswith(f'{view}.csv')] # this can be dependent on a dataset or something like that... 
                        # we need to think about it 
                        # Check if any files include `{view}-` in a case-sensitive manner
                        csv_files = [f for f in base_files if f.endswith('.csv') and f"{view}_" in f] 
                        print(f"csv_files: {csv_files}")
                        print("the length of csv_files is ", len(csv_files))

                        # Handle the case where no matching files are found
                        if not csv_files:
                            print(f"No matching files found for view: {view} in {seed_dir}")
                            continue

                        pred_file = os.path.join(seed_dir, inference_dir, csv_files[0])

                        column_structure, array_data, numeric_cols, keypoints, df_index = process_predictions(pred_file, column_structure)
                        if array_data is not None:
                            stacked_arrays.append(array_data)
                            stacked_dfs.append(numeric_cols)
                            keypoint_names = keypoints if keypoints else keypoint_names
                        
                    # Stack all arrays along the third dimension
                    stacked_arrays = np.stack(stacked_arrays, axis=-1)

                    # Compute the mean/median along the third dimension
                    if mode == 'ensemble_mean':
                        aggregated_array = np.nanmean(stacked_arrays, axis=-1)
                    elif mode == 'ensemble_median':
                        aggregated_array = np.nanmedian(stacked_arrays, axis=-1)
                    elif mode == 'eks_singleview':
                        aggregated_array, smoothed_df = run_eks_singleview(
                            markers_list= stacked_dfs,
                            keypoint_names=keypoint_names
                        )
                        # Dynamically update column structure based on smoothed_df
                        column_structure = smoothed_df.columns[
                            smoothed_df.columns.get_level_values(2).isin(
                                ['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median','x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var']
                            )
                        ]             

                    else:
                        print(f"Invalid mode: {mode}. Skipping this view")
                        continue

                    # Create a new DataFrame with the aggregated data
                    result_df = pd.DataFrame(
                        data=aggregated_array,
                        index= stacked_dfs[0].index, #df.index,
                        columns=column_structure
                    )
                    # Use the same base filename pattern that was found in the input directory
                    base_name = os.path.basename(pred_file)
                    preds_file = os.path.join(output_dir, base_name)
                    # preds_file = os.path.join(output_dir, f'180607_004_{view}.csv')
                    result_df.to_csv(preds_file)
                    print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")
                    


def run_eks_singleview(
    markers_list: List[pd.DataFrame],
    keypoint_names : List[str], # can't I take it from the configrations?
    blocks : list = [], # will need to take care of that 
    avg_mode: str = 'median',
    var_mode : str = 'confidence_weighted_var',
  
) -> tuple[np.ndarray, pd.DataFrame]: # it was only np.ndarray

    """
    Process single view data using Ensemble Kalman Smoother.
    Args:
        markers_list: List of DataFrames containing predictions from different ensemble members
        keypoint_names: List of keypoint names to process
        blocks: List of keypoint blocks for correlated noise
        avg_mode: Mode for averaging across ensemble
        var_mode: Mode for computing ensemble variance
    Returns:
        tuple:
            - np.ndarray: A NumPy array containing smoothed predictions for the keypoints. 
              This includes dimensions like 'x', 'y', 'likelihood', ensemble medians, 
              ensemble variances, and posterior variances.
            - pd.DataFrame: A DataFrame with the full smoothed data, including detailed 
              statistics for all keypoints and their dimensions.
    """

    print(f'Input data loaded for keypoints: {keypoint_names}')
    print(f'Number of ensemble members: {len(markers_list)}')

    #  # Convert DataFrame to have simple string column names
    simple_markers_list = []
    for df in markers_list:
        simple_df = df.copy()
        simple_df.columns = [f"{col[1]}_{col[2]}" for col in df.columns]
        simple_markers_list.append(simple_df)

    print(f'First few columns of simplified DataFrame: {simple_markers_list[0].columns[:6]}')


    # Run the smoother with the simplified data
    df_smoothed, smooth_params_final = ensemble_kalman_smoother_singlecam(
        markers_list=simple_markers_list,
        keypoint_names=keypoint_names,
        smooth_param= 1000, # estimating a smoothing param and it is computing the negative log likelihood of the sequence under the smoothing parametr --> this will be 10 for now 
        s_frames=None, # the frames using to actually run the optimization --> None for now but if we have a long video will probably use 10,000 frames 
        blocks=blocks, 
        avg_mode=avg_mode,
        var_mode=var_mode,
    )

    #result = df_smoothed.loc[:, df_smoothed.columns.get_level_values(2).isin(['x', 'y', 'likelihood'])].to_numpy()
    result = df_smoothed.loc[:, df_smoothed.columns.get_level_values(2).isin(['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median','x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var'])].to_numpy()
    
    return result, df_smoothed




def run_eks_multiview(
    markers_list_by_view: List[List[pd.DataFrame]],
    keypoint_names: List[str],
    views: List[str],
    blocks: list = [],
    #smooth_param: Optional[Union[float, list]] = None,
    s_frames: Optional[list] = None,
    quantile_keep_pca: float = 95,
    avg_mode: str = 'median',
    zscore_threshold: float = 2,
) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
    """
    Process multi-view data using Ensemble Kalman Smoother.
    
    Args:
        markers_list_by_view: List of lists of DataFrames containing predictions from different 
                            ensemble members, organized by view
        keypoint_names: List of keypoint names to process
        views: List of view names corresponding to the input data
        blocks: List of keypoint blocks for correlated noise
        smooth_param: Value for smoothing parameter
        s_frames: Frames for automatic optimization if smooth_param not provided
        quantile_keep_pca: Percentage of points kept for PCA
        avg_mode: Mode for averaging across ensemble ('median' or 'mean')
        zscore_threshold: Z-score threshold for filtering low ensemble std
        
    Returns:
        tuple:
            - Dict[str, np.ndarray]: Dictionary mapping view names to NumPy arrays containing 
              smoothed predictions
            - Dict[str, pd.DataFrame]: Dictionary mapping view names to DataFrames with full 
              smoothed data
    """
    print(f"Processing {len(keypoint_names)} keypoints across {len(views)} views")
    
    # Initialize result containers with None instead of lists
    final_results = {view: None for view in views}
    results_arrays = {}
    
    # Loop over keypoints; apply EKS for each individually
    for keypoint in keypoint_names:
        # Initialize camera-specific markers for this keypoint - seperate body part predictions by camera name 
        markers_by_cam = [[] for _ in range(len(views))]
        # Collect data for each view and ensemble member
        for view_idx, view_data in enumerate(markers_list_by_view):
            print(f"Processing view {views[view_idx]} data")
            for df in view_data:  # Each df is an ensemble member
                # Extract x, y, likelihood columns for this keypoint
                keypoint_cols = df.loc[:, df.columns.get_level_values(1) == keypoint]
                
                if keypoint_cols.empty:
                    print(f"No data found for keypoint {keypoint} in view {views[view_idx]}")
                    continue
                
                # Create a new dataframe with required columns
                coords_df = pd.DataFrame()
                
                # Get x, y, likelihood values maintaining original column structure
                x_val = keypoint_cols.loc[:, keypoint_cols.columns.get_level_values(2) == 'x'].iloc[:, 0]
                y_val = keypoint_cols.loc[:, keypoint_cols.columns.get_level_values(2) == 'y'].iloc[:, 0]
                likelihood_val = keypoint_cols.loc[:, keypoint_cols.columns.get_level_values(2) == 'likelihood'].iloc[:, 0]
                
                coords_df[f'{keypoint}_x'] = x_val
                coords_df[f'{keypoint}_y'] = y_val
                coords_df[f'{keypoint}_likelihood'] = likelihood_val
                
                markers_by_cam[view_idx].append(coords_df)
        
        # Check if we have valid data for all views
        if not all(cam_data for cam_data in markers_by_cam):
            print(f"Missing data for some views for keypoint {keypoint}, skipping...")
            continue
        
        try:
            print(f"Running multi-camera EKS for keypoint {keypoint}")
            
            # Run multi-camera EKS
            smoothed_dfs, smooth_params_final, nll_values = ensemble_kalman_smoother_multicam(
                markers_list_cameras=markers_by_cam,
                keypoint_ensemble=keypoint,
                smooth_param=1000,
                quantile_keep_pca=quantile_keep_pca,
                camera_names=views,
                s_frames=None,
                ensembling_mode=avg_mode,
                zscore_threshold=zscore_threshold,
            )
            
            # Store results for each view 
            for view, smoothed_df in smoothed_dfs.items():
                cols_to_keep = ['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
                              'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var']
                filtered_df = smoothed_df.loc[:, smoothed_df.columns.get_level_values(2).isin(cols_to_keep)]

                if final_results[view] is None:
                    final_results[view] = filtered_df
                else:
                    final_results[view] = pd.concat([final_results[view], filtered_df], axis=1)
                
                print(f"Updated results for view {view}, keypoint {keypoint}")
    
        except Exception as e:
            print(f"Error processing keypoint {keypoint}: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # convert final_results to arrays 
    for view, result_df in final_results.items():
        if result_df is not None:
            results_arrays[view] = result_df.to_numpy()
            print(f"Processed view {view}: shape {results_arrays[view].shape}")
        else:
            print(f"No valid results for view {view}")

    return results_arrays, final_results 




# def run_eks_multiview(
#     markers_list_by_view: List[List[pd.DataFrame]],
#     keypoint_names: List[str],
#     views: List[str],
#     blocks: list = [],
#     #smooth_param: Optional[Union[float, list]] = None,
#     s_frames: Optional[list] = None,
#     quantile_keep_pca: float = 95,
#     avg_mode: str = 'median',
#     zscore_threshold: float = 2,
# ) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
#     """
#     Process multi-view data using Ensemble Kalman Smoother.
    
#     Args:
#         markers_list_by_view: List of lists of DataFrames containing predictions from different 
#                             ensemble members, organized by view
#         keypoint_names: List of keypoint names to process
#         views: List of view names corresponding to the input data
#         blocks: List of keypoint blocks for correlated noise
#         smooth_param: Value for smoothing parameter
#         s_frames: Frames for automatic optimization if smooth_param not provided
#         quantile_keep_pca: Percentage of points kept for PCA
#         avg_mode: Mode for averaging across ensemble ('median' or 'mean')
#         zscore_threshold: Z-score threshold for filtering low ensemble std
        
#     Returns:
#         tuple:
#             - Dict[str, np.ndarray]: Dictionary mapping view names to NumPy arrays containing 
#               smoothed predictions
#             - Dict[str, pd.DataFrame]: Dictionary mapping view names to DataFrames with full 
#               smoothed data
#     """
#     print(f"Processing {len(keypoint_names)} keypoints across {len(views)} views")
    
#     # Initialize result containers
#     all_smoothed_dfs = {view: [] for view in views}
    
#     # Loop over keypoints; apply EKS for each individually
#     for keypoint in keypoint_names:
#         print(f"\nProcessing keypoint: {keypoint}")
        
#         # Initialize camera-specific markers for this keypoint - seperate body part predictions by camera name 
#         markers_by_cam = [[] for _ in range(len(views))]
        
#         # Collect data for each view and ensemble member
#         for view_idx, view_data in enumerate(markers_list_by_view):
#             print(f"Processing view {views[view_idx]} data")
            
#             for df in view_data:  # Each df is an ensemble member
#                 # Extract x, y, likelihood columns for this keypoint
#                 keypoint_cols = df.loc[:, df.columns.get_level_values(1) == keypoint]
                
#                 if keypoint_cols.empty:
#                     print(f"No data found for keypoint {keypoint} in view {views[view_idx]}")
#                     continue
                
#                 # Create a new dataframe with required columns
#                 coords_df = pd.DataFrame()
                
#                 # Get x, y, likelihood values maintaining original column structure
#                 x_val = keypoint_cols.loc[:, keypoint_cols.columns.get_level_values(2) == 'x'].iloc[:, 0]
#                 y_val = keypoint_cols.loc[:, keypoint_cols.columns.get_level_values(2) == 'y'].iloc[:, 0]
#                 likelihood_val = keypoint_cols.loc[:, keypoint_cols.columns.get_level_values(2) == 'likelihood'].iloc[:, 0]
                
#                 coords_df[f'{keypoint}_x'] = x_val
#                 coords_df[f'{keypoint}_y'] = y_val
#                 coords_df[f'{keypoint}_likelihood'] = likelihood_val
                
#                 markers_by_cam[view_idx].append(coords_df)
        
#         # Check if we have valid data for all views
#         if not all(cam_data for cam_data in markers_by_cam):
#             print(f"Missing data for some views for keypoint {keypoint}, skipping...")
#             continue
        
#         try:
#             print(f"Running multi-camera EKS for keypoint {keypoint}")
            
#             # Run multi-camera EKS
#             smoothed_dfs, smooth_params_final, nll_values = ensemble_kalman_smoother_multicam(
#                 markers_list_cameras=markers_by_cam,
#                 keypoint_ensemble=keypoint,
#                 smooth_param=1000,
#                 quantile_keep_pca=quantile_keep_pca,
#                 camera_names=views,
#                 s_frames=None,
#                 ensembling_mode=avg_mode,
#                 zscore_threshold=zscore_threshold,
#             )
            
#             # Store results for each view
#             for view, smoothed_df in smoothed_dfs.items():
#                 all_smoothed_dfs[view].append(smoothed_df)
#                 print(f"Stored results for view {view}, keypoint {keypoint}")
            
#         except Exception as e:
#             print(f"Error processing keypoint {keypoint}: {str(e)}")
#             import traceback
#             print(traceback.format_exc())
#             continue
    
#     # Combine results across keypoints for each view
#     results_arrays = {}
#     results_dfs = {}
    
#     for view in views:
#         if not all_smoothed_dfs[view]:
#             print(f"No valid results for view {view}")
#             continue
        
#         # Combine all keypoint results for this view
#         combined_df = pd.concat(all_smoothed_dfs[view], axis=1)
        
#         # Extract columns we want in the final output
#         cols_to_keep = ['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median',
#                        'x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var']
        
#         filtered_df = combined_df.loc[:, combined_df.columns.get_level_values(2).isin(cols_to_keep)]
#         results_arrays[view] = filtered_df.to_numpy()
#         results_dfs[view] = filtered_df
        
#         print(f"Processed view {view}: shape {results_arrays[view].shape}")
    
#     return results_arrays, results_dfs



'''
The one above is the one that works 

'''

# def run_eks_multiview(
#     markers_list_by_view: List[List[pd.DataFrame]],
#     keypoint_names: List[str],
#     views: List[str],
#     blocks: list = [],
#     #smooth_param: Optional[Union[float, list]] = None,
#     s_frames: Optional[list] = None,
#     quantile_keep_pca: float = 95,
#     avg_mode: str = 'median',
#     zscore_threshold: float = 2,
# ) -> tuple[dict[str, np.ndarray], dict[str, pd.DataFrame]]:
#     """
#     Process multi-view data using Ensemble Kalman Smoother.
#     """
#     print(f'Input data loaded for keypoints: {keypoint_names}')
#     print(f'Processing views: {views}')
#     print(f'Number of views with data: {len(markers_list_by_view)}')
    
#     # # If s_frames is None, create default frames
#     # n_frames = markers_list_by_view[0][0].shape[0] if markers_list_by_view and markers_list_by_view[0] else 0
#     # if s_frames is None and n_frames > 0:
#     #     # Use 1000 frames or all frames if less than 1000
#     #     n_sample = min(1000, n_frames)
#     #     s_frames = np.linspace(0, n_frames-1, n_sample, dtype=int).tolist()
#     #     print(f"Created default s_frames with {len(s_frames)} frames")
    
#     # Process results for each keypoint
#     all_smoothed_dfs = {view: [] for view in views}
    
#     for keypoint in keypoint_names:
#         print(f"\nProcessing keypoint: {keypoint}")
        
#         # Prepare data for each view
#         markers_by_cam = []
#         for view_idx, view_markers in enumerate(markers_list_by_view):
#             view_name = views[view_idx]
#             print(f"Processing view {view_name} for keypoint {keypoint}")
            
#             cam_markers = []
#             for df in view_markers:
#                 # Filter columns for current keypoint
#                 keypoint_cols = [col for col in df.columns if col[1] == keypoint]
                
#                 if not keypoint_cols:
#                     print(f"No columns found for keypoint {keypoint} in view {view_name}")
#                     continue
                    
#                 # Create a new dataframe with x, y, likelihood for this keypoint
#                 coords_df = pd.DataFrame()
                
#                 # Extract x, y, likelihood coordinates
#                 x_cols = [col for col in keypoint_cols if col[2] == 'x']
#                 y_cols = [col for col in keypoint_cols if col[2] == 'y']
#                 likelihood_cols = [col for col in keypoint_cols if col[2] == 'likelihood']
                
#                 if x_cols and y_cols and likelihood_cols:
#                     # Preserve the order expected by ensemble_kalman_smoother_multicam: x, y, likelihood
#                     x_data = df[x_cols[0]]
#                     y_data = df[y_cols[0]]
#                     likelihood_data = df[likelihood_cols[0]]
                    
#                     # Create DataFrame with columns in the correct order
#                     coords_df[f'{keypoint}_x'] = x_data
#                     coords_df[f'{keypoint}_y'] = y_data
#                     coords_df[f'{keypoint}_likelihood'] = likelihood_data
                    
#                     cam_markers.append(coords_df)
#                     print(f"Extracted data for {keypoint} with columns: {coords_df.columns}")
#                 else:
#                     print(f"Missing required columns for keypoint {keypoint} in view {view_name}")
            
#             if cam_markers:
#                 print(f"Found {len(cam_markers)} valid predictions for {keypoint} in view {view_name}")
#                 print(f"Sample data shape: {cam_markers[0].shape}")
#                 markers_by_cam.append(cam_markers)
#             else:
#                 print(f"No valid markers found for {keypoint} in view {view_name}")
        
#         if not markers_by_cam:
#             print(f"No valid data found for keypoint {keypoint} across any views")
#             continue
            
#         print(f"Prepared {len(markers_by_cam)} camera views for processing")
        
#         # Run the multi-camera smoother for this keypoint
#         try:
#             print(f"Running EKS for keypoint {keypoint} with {len(markers_by_cam)} views")
#             print(f"Using s_frames with {len(s_frames) if s_frames else 0} frames")
            
#             smoothed_dfs, smooth_params_final, nll_values = ensemble_kalman_smoother_multicam(
#                 markers_list_cameras=markers_by_cam,
#                 keypoint_ensemble=keypoint,
#                 smooth_param=1000,
#                 quantile_keep_pca=quantile_keep_pca,
#                 camera_names=[views[i] for i in range(len(markers_by_cam))],
#                 s_frames=None,  # Now properly initialized
#                 ensembling_mode=avg_mode,
#                 zscore_threshold=zscore_threshold,
#             )
            
#             # Store results for each view
#             for view in views:
#                 if view in smoothed_dfs:
#                     all_smoothed_dfs[view].append(smoothed_dfs[view])
#                     print(f"Stored results for view {view}, keypoint {keypoint}")
            
#         except Exception as e:
#             print(f"Error processing keypoint {keypoint}: {str(e)}")
#             print(f"Error details: {type(e).__name__}")
#             import traceback
#             print(traceback.format_exc())
#             continue
    
#     # Combine results for all keypoints
#     results_arrays = {}
#     results_dfs = {}
    
#     for view in views:
#         if not all_smoothed_dfs[view]:
#             print(f"No valid results for view {view}")
#             continue
            
#         # Combine DataFrames for all keypoints
#         combined_df = pd.concat(all_smoothed_dfs[view], axis=1)
#         print(f"Combined results for view {view}: {combined_df.shape}")
        
#         # Extract relevant columns for the numpy array
#         cols_to_keep = [
#             'x', 'y', 'likelihood', 
#             'x_ens_median', 'y_ens_median',
#             'x_ens_var', 'y_ens_var', 
#             'x_posterior_var', 'y_posterior_var'
#         ]
        
#         result_array = combined_df.loc[:, combined_df.columns.get_level_values(2).isin(cols_to_keep)].to_numpy()
        
#         results_arrays[view] = result_array
#         results_dfs[view] = combined_df
    
#     return results_arrays, results_dfs






'''
The code below is what works for the post process ensemble 
'''


# def post_process_ensemble(
#     cfg_lp: DictConfig,
#     results_dir: str,
#     model_type: str,
#     n_labels: int,
#     seed_range: tuple[int, int],
#     views: list[str], 
#     mode: Literal['ensemble_mean', 'ensemble_median', 'eks_singleview'],
#     inference_dirs: List[str],
#     overwrite: bool,
# ) -> None:
#     """
#     Post-processes ensemble predictions from multiple model seeds by aggregating their outputs.
    
#     Args:
#         cfg_lp (DictConfig): Configuration dictionary containing data and processing parameters
#         results_dir (str): Base directory containing the model results
#         model_type (str): Type of model used for predictions
#         n_labels (int): Number of labels/keypoints in the model
#         seed_range (tuple[int, int]): Range of seeds (start, end) to include in ensemble
#         views (list[str]): List of camera views to process
#         mode (Literal['ensemble_mean', 'ensemble_median', 'eks_singleview']): Aggregation method:
#             - ensemble_mean: Takes mean across all seed predictions
#             - ensemble_median: Takes median across all seed predictions
#             - eks_singleview: Applies Extended Kalman Smoothing to single view predictions
#         inference_dirs (List[str]): List of inference directory names to process
#         overwrite (bool): Whether to overwrite existing processed files
    
#     Returns:
#         None: Results are saved to disk in the specified output directory
        
#     The function:
#     1. Creates an ensemble directory structure based on model type and seeds
#     2. For each inference directory and view:
#         - Loads predictions from all seed models
#         - Stacks predictions into a single array
#         - Applies the specified aggregation method (mean/median/eks)
#         - Saves processed results to CSV
#         - Computes metrics on the aggregated predictions
#     """

#     # setup directories 
#     base_dir = os.path.dirname(results_dir)
#     ensemble_dir = os.path.join(
#         base_dir,
#         f"{model_type}_{n_labels}_{seed_range[0]}-{seed_range[1]}"
#     )
#     seed_dirs = [
#             os.path.join(base_dir, f"{model_type}_{n_labels}_{seed}")
#             for seed in range(seed_range[0], seed_range[1] + 1)
#     ]

#     for inference_dir in inference_dirs:
#         output_dir = os.path.join(ensemble_dir, mode, inference_dir)
#         os.makedirs(output_dir, exist_ok=True)
        
#         print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dirs} " )

#         if inference_dir == 'videos-for-each-labeled-frame':
#             new_predictions_files = []

#             for view in views: 
#                 stacked_arrays = []
#                 stacked_dfs = []
#                 column_structure = None
#                 keypoint_names = []
#                 #markers_for_this_view = [] # inner list of markers for specific camera view
                
#                 for seed_dir in seed_dirs:
#                     pred_file = os.path.join(
#                         seed_dir,
#                         inference_dir,
#                         f'predictions_{view}_new.csv'
#                     )
#                     column_structure, array_data, numeric_cols, keypoints, df_index = process_predictions(pred_file, column_structure)
#                     if array_data is not None:
#                         stacked_arrays.append(array_data)
#                         stacked_dfs.append(numeric_cols)
#                         keypoint_names = keypoints if keypoints else keypoint_names

#                 # Stack all arrays along the third dimension
#                 stacked_arrays = np.stack(stacked_arrays, axis=-1)

#                 # Compute the mean/median along the third dimension
#                 if mode == 'ensemble_mean':
#                     aggregated_array = np.nanmean(stacked_arrays, axis=-1)
#                 elif mode == 'ensemble_median':
#                     aggregated_array = np.nanmedian(stacked_arrays, axis=-1)
#                 elif mode == 'eks_singleview':
#                     aggregated_array, smoothed_df = run_eks_singleview(
#                         markers_list= stacked_dfs,
#                         keypoint_names=keypoint_names
#                     )
#                     # filtered_df = smoothed_df.loc[
#                     #     : , smoothed_df.columns.get_level_values(2).isin(['x','y','likelihood'])
#                     # ]
#                     # # Update the aggregated_array to match the filtered columns
#                     # aggregated_array = filtered_df.to_numpy()
#                     # Dynamically update column structure based on smoothed_df
#                     column_structure = smoothed_df.columns[
#                         smoothed_df.columns.get_level_values(2).isin(
#                             ['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median','x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var']
#                         )
#                     ]          

#                 else:
#                     print(f"Invalid mode: {mode}")
#                     continue

#                 # Create a new DataFrame with the aggregated data
#                 result_df = pd.DataFrame(
#                     data=aggregated_array,
#                     index= df_index, #df.index,
#                     columns=column_structure
#                 )

#                 result_df.loc[:,("set", "", "")] = "train"

#                 preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
#                 result_df.to_csv(preds_file)
#                 new_predictions_files.append(preds_file)
#                 print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")

#                 # Update cfg_lp for each view specifically
#                 cfg_lp_view = cfg_lp.copy()
#                 # Dynamically create the CSV filename based on view
#                 csv_filename = f'CollectedData_{view}_new.csv'
#                 cfg_lp_view.data.csv_file = [csv_filename]
#                 cfg_lp_view.data.view_names = [view]

#                 try:
#                     compute_metrics(cfg=cfg_lp_view, preds_file=preds_file, data_module=None)
#                     print(f"Successfully computed metrics for {preds_file}")
#                 except Exception as e:
#                     print(f"Error computing metrics\n{e}")

#         else:
#             for view in views: 
#                 stacked_arrays = []
#                 stacked_dfs = []
#                 column_structure = None
#                 keypoint_names = []

#                 for seed_dir in seed_dirs:
#                     base_files = os.listdir(os.path.join(seed_dir, inference_dir))
#                     print(f"base_files: {base_files}")
#                     # Check if any files match the `view`
#                     #csv_files = [f for f in base_files if f.endswith(f'{view}.csv')] # this can be dependent on a dataset or something like that... 
#                     # we need to think about it 
#                     # Check if any files include `{view}-` in a case-sensitive manner
#                     csv_files = [f for f in base_files if f.endswith('.csv') and f"{view}_" in f] 
#                     print(f"csv_files: {csv_files}")
#                     print("the length of csv_files is ", len(csv_files))

#                     # Handle the case where no matching files are found
#                     if not csv_files:
#                         print(f"No matching files found for view: {view} in {seed_dir}")
#                         continue

#                     pred_file = os.path.join(seed_dir, inference_dir, csv_files[0])

#                     column_structure, array_data, numeric_cols, keypoints, df_index = process_predictions(pred_file, column_structure)
#                     if array_data is not None:
#                         stacked_arrays.append(array_data)
#                         stacked_dfs.append(numeric_cols)
#                         keypoint_names = keypoints if keypoints else keypoint_names
                     
#                 # Stack all arrays along the third dimension
#                 stacked_arrays = np.stack(stacked_arrays, axis=-1)

#                 # Compute the mean/median along the third dimension
#                 if mode == 'ensemble_mean':
#                     aggregated_array = np.nanmean(stacked_arrays, axis=-1)
#                 elif mode == 'ensemble_median':
#                     aggregated_array = np.nanmedian(stacked_arrays, axis=-1)
#                 elif mode == 'eks_singleview':
#                     aggregated_array, smoothed_df = run_eks_singleview(
#                         markers_list= stacked_dfs,
#                         keypoint_names=keypoint_names
#                     )
#                     # Dynamically update column structure based on smoothed_df
#                     column_structure = smoothed_df.columns[
#                         smoothed_df.columns.get_level_values(2).isin(
#                             ['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median','x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var']
#                         )
#                     ]             

#                 else:
#                     print(f"Invalid mode: {mode}. Skipping this view")
#                     continue

#                 # Create a new DataFrame with the aggregated data
#                 result_df = pd.DataFrame(
#                     data=aggregated_array,
#                     index= stacked_dfs[0].index, #df.index,
#                     columns=column_structure
#                 )
#                 # Use the same base filename pattern that was found in the input directory
#                 base_name = os.path.basename(pred_file)
#                 preds_file = os.path.join(output_dir, base_name)
#                 # preds_file = os.path.join(output_dir, f'180607_004_{view}.csv')
#                 result_df.to_csv(preds_file)
#                 print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")

'''
The code commented above is the original thing that I had 

'''



# def run_eks_multiview(
#     markers_list: List[pd.DataFrame],
#     keypoint_names : List[str], # can't I take it from the configrations?
#     camera_names : List[str], # can't I take it from the configrations?
#     quantile_keep_pca: float = 95,
#     blocks : list = [], # will need to take care of that 
#     avg_mode: str = 'median',
#     zscore_threshold: float = 2,

# ) -> tuple[np.ndarray, pd.DataFrame]:

#     """
#     Process Multi view data using Ensemble Kalman Smoother.
#     Args:
#         markers_list: List of DataFrames containing predictions from different ensemble members
#         keypoint_names: List of keypoint names to process
#         blocks: List of keypoint blocks for correlated noise
#         avg_mode: Mode for averaging across ensemble
#     Returns:
#         tuple:
#             - np.ndarray: A NumPy array containing smoothed predictions for the keypoints. 
#               This includes dimensions like 'x', 'y', 'likelihood', ensemble medians, 
#               ensemble variances, and posterior variances.
#             - pd.DataFrame: A DataFrame with the full smoothed data, including detailed 
#               statistics for all keypoints and their dimensions.
        
#     """

#     print(f'Input data loaded for keypoints: {keypoint_names}')
#     print(f'Number of ensemble members: {len(markers_list)}')    

#      #  # Convert DataFrame to have simple string column names
#     simple_markers_list = []
#     for df in markers_list:
#         simple_df = df.copy()
#         simple_df.columns = [f"{col[1]}_{col[2]}" for col in df.columns]
#         simple_markers_list.append(simple_df)

#     print(f'First few columns of simplified DataFrame: {simple_markers_list[0].columns[:6]}')


# def format_data_post_process (markers_list, camera_names = None):

#     if camera_names is None:
        


















# I will work on the singlecam_example from the scripts in eks 
# use fir_eks_singlecam - I will give a list of the csv files - the prediction files of the ensemble (just one view) 
# For eks multiview I won't be able to loop through the views and seeds 
# for now in the pipeline_simple file when I do the post processing I can check 
# got to eks github repo and look at how they fit the eks for single view - the example 
# https://github.com/paninski-lab/eks/blob/main/README.md - run the example from there in the studio 
    








# Here will start looping over the post processes 
# want to check if want to run the particular post process and have a couple of if statmetns
# combine predictions from multiple models in the ensemble if want ensmble_mean run this function and if want eks run this
# make a new py file called pose processing and basically want to load predictions from different models, want to take the mean / median of all the x and y and also of likelihood - that will be the ensemble mean and median 
# that will all be saved as a data frame in csv file inside the supervised_100_0-1 directory and make another directory for each post processor - ensemble_mean, ensemble_median 
# once have that data frame  I can run compute metrics from the new set of predictions and it will do the pixel_error 







# after loop through all the seeds want to run through the post=processes  
# for this I need to implement ensemble mean and median 
# take the predictions files in the videos-for-each-labeled-frame and load the csv files from each seed and each view 
# I want the prediction files from supervised-100-0 and supervised 100-1 
#. I will have to make a new directory supervised_100_0 and supervised_100_1 and the directory for the ensemble will be supervised_100_0-1 (if had more it is 0-5 for example)
