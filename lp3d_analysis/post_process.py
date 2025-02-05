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


def process_final_predictions(
    view: str,
    output_dir: str,
    seed_dirs: List[str],
    inference_dir: str,
    cfg_lp: DictConfig,
) -> None:
    """Process and save final predictions for a view, computing metrics"""
    
    # Load original CSV file to get frame paths
    orig_pred_file = os.path.join(seed_dirs[0], inference_dir, f'predictions_{view}_new.csv')
    original_df = pd.read_csv(orig_pred_file, header=[0, 1, 2], index_col=0)
    original_index = original_df.index
    results_list = []
    
    # Process each path from original predictions
    for img_path in original_index:
        # Get sequence name and base filename
        sequence_name = img_path.split('/')[1]  # e.g., "180623_000_bot"
        base_filename = os.path.splitext(os.path.basename(img_path))[0]  # e.g., "img107262"
        
        # Get the sequence CSV path
        snippet_file = os.path.join(output_dir, sequence_name, f"{base_filename}.csv")
        print(f"\nProcessing {img_path}")
        print(f"Looking for file: {snippet_file}")
        
        if os.path.exists(snippet_file):
            try:
                # Load the CSV with multi-index columns
                snippet_df = pd.read_csv(snippet_file, header=[0, 1, 2], index_col=0)
                print(f"Loaded snippet file with shape: {snippet_df.shape}")
                
                # Ensure odd number of frames
                assert snippet_df.shape[0] & 1 == 1, f"Expected odd number of frames, got {snippet_df.shape[0]}"
                
                # Get center frame index
                idx_frame = int(np.floor(snippet_df.shape[0] / 2))
                print(f"Extracting center frame at index {idx_frame}")
                
                # Extract center frame and set index
                center_frame = snippet_df.iloc[[idx_frame]]  # Keep as DataFrame with one row
                center_frame.index = [img_path]
                results_list.append(center_frame)
                
                print(f"Successfully extracted center frame")
                
            except Exception as e:
                print(f"Error processing {snippet_file}: {str(e)}")
                print("Full error:")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Could not find file {snippet_file}")
    
    if results_list:
        print(f"\nCombining {len(results_list)} processed frames")
        # Combine all results in original order
        results_df = pd.concat(results_list)
        print(f"Combined DataFrame shape: {results_df.shape}")
        
        # Reindex to match original
        results_df = results_df.reindex(original_index)
        print(f"Final DataFrame shape: {results_df.shape}")
        
        # Add "set" column for labeled data predictions
        results_df.loc[:,("set", "", "")] = "train"
        
        # Save predictions
        preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
        results_df.to_csv(preds_file)
        print(f"Saved predictions to {preds_file}")
        
        # Compute metrics
        cfg_lp_view = cfg_lp.copy()
        cfg_lp_view.data.csv_file = [f'CollectedData_{view}_new.csv']
        cfg_lp_view.data.view_names = [view]
        
        try:
            compute_metrics(cfg=cfg_lp_view, preds_file=preds_file, data_module=None)
            print(f"Successfully computed metrics for {preds_file}")
        except Exception as e:
            print(f"Error computing metrics for {view}: {str(e)}")
    else:
        print(f"Warning: No frames processed for view {view}")



def post_process_ensemble_labels(
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

    # setup directories 
    base_dir = os.path.dirname(results_dir)
    ensemble_dir = os.path.join(
        base_dir,
        f"{model_type}_{n_labels}_{seed_range[0]}-{seed_range[1]}"
    )
    print(f"ensemble_dir: {ensemble_dir}")
    seed_dirs = [
            os.path.join(base_dir, f"{model_type}_{n_labels}_{seed}")
            for seed in range(seed_range[0], seed_range[1] + 1)
    ]
    print(f"seed_dirs: {seed_dirs}")

    for inference_dir in inference_dirs:
        
        print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dirs} " )
        first_seed_dir = seed_dirs[0]
        inf_dir_in_first_seed = os.path.join(first_seed_dir, inference_dir)
        # Check if the inference_dir inside the first_seed_dir has subdirectories
        entries = os.listdir(inf_dir_in_first_seed)
        has_subdirectories = any(os.path.isdir(os.path.join(inf_dir_in_first_seed, entry)) for entry in entries)
        if not has_subdirectories:
            print(f"No subdirectories found in {inf_dir_in_first_seed}. Skipping.")
            continue  # Skip this inference directory and move to the next one
        print(f"Subdirectories found in {inf_dir_in_first_seed}. Continuing to process {inference_dir}...")

        output_dir = os.path.join(ensemble_dir, mode, inference_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f" the first seed dir is {first_seed_dir}")
        video_dir = os.path.join(first_seed_dir, inference_dir)

        # Create a mapping of original directories and their files
        original_structure = {}
        for view in views:
            view_dirs = [
                d for d in os.listdir(video_dir) 
                if os.path.isdir(os.path.join(video_dir, d)) and d.endswith(f'_{view}')
            ]
            print(f"view_dirs: {view_dirs}")
            for dir_name in view_dirs:
                dir_path = os.path.join(video_dir, dir_name)
                csv_files = [f for f in os.listdir(dir_path) if f.endswith('.csv')]
                original_structure[dir_name] = csv_files


        if mode == 'eks_multiview':
            # Process each view-specific directory
            for original_dir, csv_files in original_structure.items():
                print(f"Processing directory: {original_dir}")
                base_name = original_dir.rsplit('_', 1)[0]  # e.g., "180607_004"
                curr_view = original_dir.split('_')[-1]     # e.g., "bot" or "top"

                # Create output directory matching original structure
                sequence_output_dir = os.path.join(output_dir, original_dir)
                os.makedirs(sequence_output_dir, exist_ok=True)

                # Process each CSV file in the directory
                for csv_file in csv_files:
                    print(f"Processing file: {csv_file}")
                
                    all_pred_files = []
                    for view in views:
                        curr_dir_name = f"{base_name}_{view}"
                        for seed_dir in seed_dirs:
                            seed_video_dir = os.path.join(seed_dir, inference_dir)
                            seed_sequence_dir = os.path.join(seed_video_dir, curr_dir_name)
                            if os.path.exists(seed_sequence_dir):
                                pred_file = os.path.join(seed_sequence_dir, csv_file)
                                if os.path.exists(pred_file):
                                    all_pred_files.append(pred_file)

                    if all_pred_files:
                        # Get input_dfs_list and keypoint_names for this file
                        input_dfs_list, keypoint_names = format_data(
                            input_source=all_pred_files,
                            camera_names=views,
                        )

                        results_dfs = run_eks_multiview(
                            markers_list=input_dfs_list,
                            keypoint_names=keypoint_names,
                            views=views,
                            inflate_vars_likelihood_thresh=0,
                            inflate_vars_v_quantile_thresh=70,
                        )

                        # Save results using original filename
                        for view, result_df in results_dfs.items():
                            if view == curr_view:  # Only save for current view
                                result_file = os.path.join(sequence_output_dir, csv_file)
                                result_df.to_csv(result_file)
                                print(f"Saved EKS results to {result_file}")
                
            for view in views:   
                process_final_predictions(
                    view=view,
                    output_dir=output_dir,
                    seed_dirs=seed_dirs,
                    inference_dir=inference_dir,
                    cfg_lp=cfg_lp
                )

        else:
            for view in views:
                print(f"\nProcessing view: {view}")
                # Process only directories for current view
                view_dirs = {
                    dir_name: files for dir_name, files in original_structure.items() 
                    if dir_name.endswith(f'_{view}')
                }
                
                # Process each view-specific directory
                for original_dir, csv_files in view_dirs.items():
                    print(f"Processing directory: {original_dir}")
                    base_name = original_dir.rsplit('_', 1)[0]  # e.g., "180607_004"

                    # Create output directory matching original structure
                    sequence_output_dir = os.path.join(output_dir, original_dir)
                    os.makedirs(sequence_output_dir, exist_ok=True)

                    # Process each CSV file in the directory
                    for csv_file in csv_files:
                        print(f"Processing file: {csv_file}")

                        all_pred_files = []
                        curr_dir_name = f"{base_name}_{view}"
                        for seed_dir in seed_dirs:
                            seed_video_dir = os.path.join(seed_dir, inference_dir)
                            seed_sequence_dir = os.path.join(seed_video_dir, curr_dir_name)
                            if os.path.exists(seed_sequence_dir):
                                pred_file = os.path.join(seed_sequence_dir, csv_file)
                                if os.path.exists(pred_file):
                                    all_pred_files.append(pred_file)
                        
                        if all_pred_files:
                            # Get input_dfs_list and keypoint_names for this file
                            input_dfs_list, keypoint_names = format_data(
                                input_source=all_pred_files,
                                camera_names=None,
                            )

                            # Process predictions to get column structure
                            column_structure,_,_,_,_ = process_predictions(all_pred_files[0], None)

                            # stack arrays for processing 
                            stacked_arrays = []
                            stacked_dfs = []
                            for markers_curr_fmt in input_dfs_list:
                                df_index = markers_curr_fmt.index
                                array_data = markers_curr_fmt.to_numpy()
                                numeric_cols = markers_curr_fmt.select_dtypes(include=[np.number])
                                stacked_arrays.append(array_data)
                                stacked_dfs.append(numeric_cols)
                            
                            stacked_arrays = np.stack(stacked_arrays, axis=-1)
                            if mode == 'ensemble_mean':
                                aggregated_array = np.nanmean(stacked_arrays, axis=-1)
                            elif mode == 'ensemble_median':
                                aggregated_array = np.nanmedian(stacked_arrays, axis=-1)
                            elif mode == 'eks_singleview':
                                results_df = run_eks_singleview(
                                    markers_list=stacked_dfs,
                                    keypoint_names=keypoint_names
                                )

                                # Dynamically update column structure based on smoothed_df
                                column_structure = results_df.columns[
                                    results_df.columns.get_level_values(2).isin(
                                        ['x', 'y', 'likelihood', 'x_ens_median', 'y_ens_median','x_ens_var', 'y_ens_var', 'x_posterior_var', 'y_posterior_var']
                                    )
                                ]    
                                aggregated_array = results_df.loc[:, column_structure].to_numpy()
                            
                            # Create a new DataFrame with the aggregated data
                            results_df = pd.DataFrame(
                                data=aggregated_array,
                                index=df_index,
                                columns=column_structure
                            )

                            result_file = os.path.join(sequence_output_dir, csv_file)
                            results_df.to_csv(result_file)
                            print(f"Saved ensemble {mode} predictions for {view} view to {result_file}")

                # Process final predictions for this view
                process_final_predictions(
                    view=view,
                    output_dir=output_dir,
                    seed_dirs=seed_dirs,
                    inference_dir=inference_dir,
                    cfg_lp=cfg_lp
                )


def post_process_ensemble_videos(
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

    # setup directories 
    base_dir = os.path.dirname(results_dir)
    ensemble_dir = os.path.join(
        base_dir,
        f"{model_type}_{n_labels}_{seed_range[0]}-{seed_range[1]}"
    )
    print(f"ensemble_dir: {ensemble_dir}")
    seed_dirs = [
            os.path.join(base_dir, f"{model_type}_{n_labels}_{seed}")
            for seed in range(seed_range[0], seed_range[1] + 1)
    ]
    print(f"seed_dirs: {seed_dirs}")

    for inference_dir in inference_dirs:
        
        print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dirs}")
        first_seed_dir = seed_dirs[0]
        inf_dir_in_first_seed = os.path.join(first_seed_dir, inference_dir)
        
        # Check if the inference_dir contains ANY subdirectories
        entries = os.listdir(inf_dir_in_first_seed)
        contains_subdirectory = any(os.path.isdir(os.path.join(inf_dir_in_first_seed, entry)) for entry in entries)
        
        if contains_subdirectory:
            print(f"Found subdirectories in {inf_dir_in_first_seed}. Skipping.")
            continue  # Skip if there are any subdirectories
        
        print(f"Files found in {inf_dir_in_first_seed}. Continuing to process {inference_dir}...")
        print(f"Directory contains only files. Continuing to process {inference_dir}...")

        output_dir = os.path.join(ensemble_dir, mode, inference_dir)
        os.makedirs(output_dir, exist_ok=True)

        if mode == 'eks_multiview':
            input_dfs_list = [] 
            column_structure = None
            keypoint_names = []
            view_indices = {}
            all_pred_files = []

            for view_idx, view in enumerate(views):
                pred_files_for_view = []
                for seed_dir in seed_dirs:
                    base_files = os.listdir(os.path.join(seed_dir, inference_dir))
                    print(f"base_files: {base_files}")
                    pred_files = [
                        os.path.join(seed_dir, inference_dir, f)
                        for f in base_files 
                        if f.endswith('.csv') and (f"{view}_" in f or f.endswith(f"_{view}.csv"))
                    ]
                    pred_files_for_view.extend(pred_files)

                if pred_files_for_view:
                    print(f"pred_files_for_view: {pred_files_for_view}")
                    sample_df = pd.read_csv(pred_files_for_view[0], header=[0, 1, 2], index_col=0)
                    view_indices[view] = sample_df.index.tolist()
                
                all_pred_files.extend(pred_files_for_view)
                
            print(f"The views names are {views}")
            print(f"The type of views names are {type(views)}")
            
            # Get input_dfs_list and keypoint_names using format_data
            input_dfs_list, keypoint_names = format_data(
                input_source=all_pred_files,
                camera_names=views,
            )

            print(f"The shape of input_dfs_list is {len(input_dfs_list)}")
            print(f"Input dfs list is {input_dfs_list}")

            # Run multiview EKS
            results_dfs = run_eks_multiview(
                markers_list=input_dfs_list,
                keypoint_names=keypoint_names,
                views=views,
            )

            # Save results for each view
            for view in views:
                result_df = results_dfs[view]
                # Use the same base filename pattern
                base_files = os.listdir(os.path.join(seed_dirs[0], inference_dir))
                print(f"base_files: {base_files}")
                csv_files = [f for f in base_files if f.endswith('.csv') and (f"{view}_" in f or f.endswith(f"_{view}.csv"))]
                
                if csv_files:
                    base_name = csv_files[0]
                    print("base_name is ", base_name)
                    preds_file = os.path.join(output_dir, base_name)
                    result_df.to_csv(preds_file)
                    print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")

        else:
            for view in views: 
                stacked_arrays = []
                stacked_dfs = []
                column_structure = None
                keypoint_names = []
                df_index = None
                pred_files = []
                
                for seed_dir in seed_dirs:
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
    inflate_vars_likelihood_thresh = None,
    inflate_vars_v_quantile_thresh = None,
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
        smooth_param = 10,
        quantile_keep_pca= 50, #quantile_keep_pca
        camera_names = views,
        s_frames = None,
        avg_mode = avg_mode,
        var_mode = var_mode,
        inflate_vars = False,
        inflate_vars_likelihood_thresh = inflate_vars_likelihood_thresh,
        inflate_vars_v_quantile_thresh = inflate_vars_v_quantile_thresh,
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

                    



