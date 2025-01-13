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

from  eks.singlecam_smoother import ensemble_kalman_smoother_singlecam

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
    mode: Literal['ensemble_mean', 'ensemble_median', 'eks_singleview'],
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
        np.ndarray: Smoothed predictions array
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


# def post_process_ensemble_multiview(
#     cfg_lp: DictConfig,
#     results_dir: str,
#     model_type: str,
#     n_labels: int,
#     seed_range: tuple[int, int],
#     views: list[str], 
#     mode: Literal['eks_multiview'],
#     overwrite: bool,
# )-> None:

#     base_dir = os.path.dirname(results_dir)
#     ensemble_dir = os.path.join(
#         base_dir,
#         f"{model_type}_{n_labels}_{seed_range[0]}-{seed_range[1]}"
#     )
#     output_dir = os.path.join(ensemble_dir, mode)
#     os.makedirs(output_dir, exist_ok=True)

#     seed_dirs = [
#         os.path.join(base_dir, f"{model_type}_{n_labels}_{seed}")
#         for seed in range(seed_range[0], seed_range[1] + 1)
#     ]

#     new_predictions_files = []

















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
