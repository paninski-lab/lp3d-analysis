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

from eks.singlecam_smoother import ensemble_kalman_smoother_singlecam, fit_eks_singlecam
from eks.multicam_smoother import ensemble_kalman_smoother_multicam
from eks.utils import convert_lp_dlc, format_data


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
    model_name = df.columns[0][0]
    numeric_cols = df.loc[:, column_structure]
    print(f"numeric_cols are {numeric_cols}")
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

        if mode == 'eks_multiview':

            input_dfs_list = [] 
            column_structure = None
            keypoint_names = []
            view_indices = {}
            all_pred_files = []
            

            for view_idx, view in enumerate(views):
                pred_files_for_view = []
                for seed_dir in seed_dirs:
                    if inference_dir == 'videos-for-each-labeled-frame':
                        pred_files = os.path.join(
                            seed_dir,
                            inference_dir,
                            f'predictions_{view}_new.csv'
                        )
                        pred_files_for_view.append(pred_files)

                    else:  
                        base_files = os.listdir(os.path.join(seed_dir, inference_dir))
                        print(f"base_files: {base_files}")
                        pred_files = [
                            os.path.join(seed_dir, inference_dir, f)
                            for f in base_files 
                            if f.endswith('.csv') and (f"{view}_" in f or f.endswith(f"_{view}.csv"))
                        ]
                        pred_files_for_view.extend(pred_files)


                if pred_files_for_view:
                # Read one file to get the indices
                    print(f"pred_files_for_view: {pred_files_for_view}")
                    sample_df = pd.read_csv(pred_files_for_view[0], header=[0, 1, 2], index_col=0)
                    view_indices[view] = sample_df.index.tolist()  # Store as a list of indices
                
                all_pred_files.extend(pred_files_for_view)

            print(f" the views names are {views}") 
            print(f" the type of views names are {type(views)}")           
            # Get input_dfs_list and keypoint_names using format_data
            input_dfs_list, keypoint_names = format_data(
                input_source=all_pred_files,
                camera_names=views,
            )

            # # Clean keypoint names by removing unexpected entries
            # keypoint_names = [
            #     keypoint for keypoint in keypoint_names if not keypoint.startswith("Unnamed")
            # ]

            # Run multiview EKS
            results_dfs = run_eks_multiview(
                markers_list=input_dfs_list,
                keypoint_names=keypoint_names,
                views=views,
            )                    
            # save results for each view and compute metris 
            for view in views:
                
                result_df = results_dfs[view] 

                if inference_dir == 'videos-for-each-labeled-frame':
                    if view in view_indices:
                        result_df.index = view_indices[view]
                        
                    else:
                        print(f"Warning: No df_index found for view {view}")
                    # I need to to have the index too 
                    result_df.loc[:,("set", "", "")] = "train"

                    preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
                    result_df.to_csv(preds_file)
                    print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")

                    # Update cfg_lp for each view specifically
                    cfg_lp_view = cfg_lp.copy()
                    cfg_lp_view.data.csv_file = [f'CollectedData_{view}_new.csv']
                    cfg_lp_view.data.view_names = [view]
                    
                    try:
                        compute_metrics(cfg=cfg_lp_view, preds_file=preds_file, data_module=None)
                        print(f"Successfully computed metrics for {preds_file}")
                    except Exception as e:
                        print(f"Error computing metrics for {view}: {str(e)}")
                
                else: 
                    # I need to save the results for each view                    
                    base_files = os.listdir(os.path.join(seed_dirs[0], inference_dir))
                    print(f"base_files: {base_files}")
                    #csv_files = [f for f in base_files if f.endswith('.csv') and f"{view}_" in f]
                    csv_files = [f for f in base_files if f.endswith('.csv') and (f"{view}_" in f or f.endswith(f"_{view}.csv"))]
                    if csv_files:
                        base_name = csv_files[0]
                        print("base_name is ", base_name)
                        preds_file = os.path.join(output_dir, base_name)
                        result_df.to_csv(preds_file)
                        print(f"Saved ensemble {mode} predictions for {view} to {preds_file}")
            
        else: 
            new_predictions_files = []   
            for view in views: 
                stacked_arrays = []
                stacked_dfs = []
                column_structure = None
                keypoint_names = []
                df_index = None
                pred_files = []
                

                for seed_dir in seed_dirs:
                    if inference_dir == 'videos-for-each-labeled-frame':
                        pred_file = os.path.join(
                            seed_dir,
                            inference_dir,
                            f'predictions_{view}_new.csv'
                        ) # remember that it wasn't 
                        pred_files.append(pred_file)
                    else:  
                        base_files = os.listdir(os.path.join(seed_dir, inference_dir))
                        print(f"base_files: {base_files}")
                        # we need to think about it 
                        csv_files = [f for f in base_files if f.endswith('.csv') and (f"{view}_" in f or f.endswith(f"_{view}.csv"))]
                        # assert that the length of csv_files is 1
                        assert len(csv_files) == 1
                        pred_file = os.path.join(seed_dir, inference_dir, csv_files[0])
                        pred_files.append(pred_file)
                        
                    
                # Get input_dfs_list and keypoint_names using format_data
                input_dfs_list, keypoint_names = format_data(
                    input_source = pred_files,
                    camera_names = None,
                )
                print(f"keypoint_names after format_data are: {keypoint_names}")
                # # Clean keypoint names by removing unexpected entries
                # keypoint_names = [
                #     keypoint for keypoint in keypoint_names if not keypoint.startswith("Unnamed")
                # ]

                column_structure,_,_,_,_ = process_predictions(pred_files[0], column_structure)
                

                for markers_curr_fmt in input_dfs_list:
                    # Assuming markers_curr_fmt is a DataFrame
                    if df_index is None:
                        # Capture the index from the first valid DataFrame
                        df_index = markers_curr_fmt.index

                    array_data = markers_curr_fmt.to_numpy()
                    numeric_cols = markers_curr_fmt.select_dtypes(include=[np.number])
                    stacked_arrays.append(array_data)
                    stacked_dfs.append(numeric_cols)


                # Stack all arrays along the third dimension
                stacked_arrays = np.stack(stacked_arrays, axis=-1)

                # Compute the mean/median along the third dimension
                if mode == 'ensemble_mean':
                    aggregated_array = np.nanmean(stacked_arrays, axis=-1)
                elif mode == 'ensemble_median':
                    aggregated_array = np.nanmedian(stacked_arrays, axis=-1)
                elif mode == 'eks_singleview':
                    results_df = run_eks_singleview(
                        markers_list= stacked_dfs,
                        keypoint_names=keypoint_names
                    )
                    # Dynamically update column structure based on smoothed_df
                    column_structure = results_df.columns[
                        results_df.columns.get_level_values(2).isin(
                            ['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median','x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var']
                        )
                    ]    
                    aggregated_array = results_df.loc[:, column_structure].to_numpy()      

                else:
                    print(f"Invalid mode: {mode}")
                    continue
                
                # Create a new DataFrame with the aggregated data
                results_df = pd.DataFrame(
                    data=aggregated_array,
                    index= df_index, #df.index, --> check about it because for the other ones that are not videos for each labeled frames I used stacked_dfs[0].index
                    columns=column_structure
                )

                if inference_dir == 'videos-for-each-labeled-frame':
                    results_df.loc[:,("set", "", "")] = "train"
                    preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
                    results_df.to_csv(preds_file)
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
                    # Use the same base filename pattern that was found in the input directory
                    base_files = os.listdir(os.path.join(seed_dirs[0], inference_dir))
                    print(f"base_files: {base_files}")
                    csv_files = [f for f in base_files if f.endswith('.csv') and (f"{view}_" in f or f.endswith(f"_{view}.csv"))]
                    print(f"csv_files: {csv_files}")
                    if csv_files:
                        base_name = csv_files[0]
                        print("base_name is ", base_name)
                        preds_file = os.path.join(output_dir, base_name)
                        results_df.to_csv(preds_file)
                        print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")



                    
def run_eks_singleview(
    markers_list: List[pd.DataFrame],
    keypoint_names : List[str], # can't I take it from the configrations?
    blocks : list = [], # will need to take care of that 
    avg_mode: str = 'median',
    var_mode : str = 'confidence_weighted_var',
  
) -> pd.DataFrame:
    # tuple[np.ndarray, pd.DataFrame]:
    """
    Process single view data using Ensemble Kalman Smoother.
    Args:
        markers_list: List of DataFrames containing predictions from different ensemble members
        keypoint_names: List of keypoint names to process
        blocks: List of keypoint blocks for correlated noise
        avg_mode: Mode for averaging across ensemble
        var_mode: Mode for computing ensemble variance
    Returns:
        pd.DataFrame: A DataFrame with the full smoothed data, including detailed
        statistics for all keypoints and their dimensions.
    """

    print(f'Input data loaded for keypoints: {keypoint_names}')
    print(f'Number of ensemble members: {len(markers_list)}')



    # Run the smoother with the simplified data
    results_df, smooth_params_final = ensemble_kalman_smoother_singlecam(
        markers_list= markers_list, #simple_markers_list,
        keypoint_names=keypoint_names,
        smooth_param= 1000, # estimating a smoothing param and it is computing the negative log likelihood of the sequence under the smoothing parametr --> this will be 10 for now 
        s_frames=None, # the frames using to actually run the optimization --> None for now but if we have a long video will probably use 10,000 frames 
        blocks=blocks, 
        avg_mode=avg_mode,
        var_mode=var_mode,
    )
    
    return results_df



def run_eks_multiview(
    markers_list: List[List[pd.DataFrame]],
    keypoint_names: List[str],
    views: List[str],
    blocks: list = [],
    smooth_param: Optional[Union[float, list]] = None,
    s_frames: Optional[list] = None,
    quantile_keep_pca: float = 95,
    avg_mode: str = 'median',
    var_mode: str = 'confidence_weighted_var',
    verbose: bool = False,
) -> dict[str, pd.DataFrame]:

    """
    Process multi-view data using Ensemble Kalman Smoother. This function handles multiple camera views
    and ensemble predictions for each keypoint.

    Args:
        markers_list: Nested list of DataFrames containing predictions from different ensemble members
                    for each camera view. Structure: [camera_view][ensemble_member]
        keypoint_names: List of keypoint names to process (e.g., ['nose', 'left_ear', 'right_ear'])
        views: List of camera view names (e.g., ['cam1', 'cam2', 'cam3'])
        blocks: List of keypoint blocks for correlated noise processing. Default is empty list.
        smooth_param: Smoothing parameter(s) for the Kalman filter. Can be a single float or list.
                    Default is None, which uses internal default value of 1000.
        s_frames: List of frames to use for optimization. Default is None (uses all frames).
        quantile_keep_pca: Quantile threshold for PCA component retention. Default is 95.
        avg_mode: Method for averaging across ensemble members. Default is 'median'.
        var_mode: Method for computing ensemble variance. Default is 'confidence_weighted_var'.
        verbose: Enable detailed progress output. Default is False.

    Returns:
        dict[str, pd.DataFrame]: Dictionary mapping view names to smoothed DataFrames containing
        processed keypoint data with statistics including coordinates, likelihoods, ensemble
        medians, ensemble variances, and posterior variances.
    """


    print(f'Input data loaded for keypoints for multiview data: {keypoint_names}')
    
    # Initialize markers list
    markers_list_all = []
    
    # 1. iterate over keypoints
    for keypoint in keypoint_names:
        # seperate predictions by camera view for current keypoint 
        markers_list_cameras = [[] for _ in range(len(views))]
        # 2. organize data by camera view
        for c, camera_name in enumerate(views):
            ensemble_members = markers_list[c] # get ensemble member for this camera 
            # 3. each dataframe in markers_list is an ensemble member - process each ensemble member 
            for markers_curr in ensemble_members:
                non_likelihood_keys = [
                    key
                    for key in markers_curr.keys()
                    if keypoint in str(key)  # Match keypoint name in column names
                ]
                
                markers_list_cameras[c].append(markers_curr[non_likelihood_keys])
            
        markers_list_all.append(markers_list_cameras)

    # run the ensemble kalman smoother for multiview data
    camera_dfs, smooth_params_final = ensemble_kalman_smoother_multicam(
        markers_list = markers_list_all,
        keypoint_names = keypoint_names,
        smooth_param = 1000,
        quantile_keep_pca= 50, #quantile_keep_pca
        camera_names = views,
        s_frames = None,
        avg_mode = avg_mode,
        var_mode = var_mode,
        inflate_vars = False,
        verbose = verbose,
    )

    # Process results for each view
    results_arrays = {}
    results_dfs = {}

    for view_idx, view in enumerate(views):
        if view_idx >= len(camera_dfs) or camera_dfs[view_idx] is None:
            print(f'No results available for view {view}')
            continue
            
        # Extract relevant columns for numpy array
        result_cols = [
            'x', 'y', 'likelihood', 
            'x_ens_median', 'y_ens_median',
            'x_ens_var', 'y_ens_var', 
            'x_posterior_var', 'y_posterior_var'
        ]
        
        df = camera_dfs[view_idx]
        array_data = df.loc[
            :, 
            df.columns.get_level_values(2).isin(result_cols)
        ].to_numpy()
        
        results_arrays[view] = array_data
        results_dfs[view] = df
        
        print(f'Successfully processed view {view}')
    
    return results_dfs

                    





    



