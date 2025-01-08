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
#1. in the variables for the function pose_process_ensemble, try to change the issue with the mode variable so we can add the eks mode and things like that 
#2. Change the way we are using the cfp_lp.data.csv_file - it will make more sense in the function 
#3 change variables names so it will make more sense 
# 4 after I work on the plotting We are going to do the EKS  
#5. look at format_data in eks 



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

    for inference_dir in inference_dirs:
        print(f"Running post-processing for inference directory: {inference_dir}")
        base_dir = os.path.dirname(results_dir)
        print(f"The results directory is {base_dir}")
        ensemble_dir = os.path.join(
            base_dir,
            f"{model_type}_{n_labels}_{seed_range[0]}-{seed_range[1]}"
        )
        output_dir = os.path.join(ensemble_dir, mode, inference_dir)
        os.makedirs(output_dir, exist_ok=True)

        seed_dirs = [
            os.path.join(base_dir, f"{model_type}_{n_labels}_{seed}")
            for seed in range(seed_range[0], seed_range[1] + 1)
        ]
        
        print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dirs} " )

        if inference_dir == 'videos-for-each-labeled-frame':
            new_predictions_files = []

            for view in views: 
                stacked_arrays = []
                stacked_dfs = []
                column_structure = None
                #base_keypoints = []
                keypoint_names = []
                
                for seed_dir in seed_dirs:
                    pred_file = os.path.join(
                        seed_dir,
                        inference_dir,
                        f'predictions_{view}_new.csv'
                    )
                    if os.path.exists(pred_file):
                        df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
                        if column_structure is None:
                            column_structure = df.loc[:, df.columns.get_level_values(2).isin(['x', 'y', 'likelihood'])].columns
                            #Extract keypoint names in the original order from the second level of the MultiIndex
                            keypoint_names = list(dict.fromkeys(column_structure.get_level_values(1)))
                            #keypoint_names = list(column_structure.get_level_values(1)) # keep all repetitions 
                            # base_keypoints = list(dict.fromkeys(column_structure.get_level_values(1)))
                            # print(f' basekeypoint names are {base_keypoints}')
                            
                            print(f'Keypoint names are {keypoint_names}')

                        # Select only numeric columns (x, y, likelihood)
                        numeric_cols = df.loc[:, column_structure]

                        # Convert DataFrame to a 2D array (numeric values only)
                        stacked_arrays.append(numeric_cols.to_numpy())
                        # create stacked_dfs for eks_singleview --> this will be the markers list 
                        stacked_dfs.append(numeric_cols) # this will be markers list 
                        print(f'the shape of the stacked arrays is {np.shape(stacked_arrays)}')
                        print(f'the shape of the stacked dfs is {stacked_dfs[0].shape}')

                    else:
                        print(f"Warning: Could not find predictions file: {pred_file}")

                if not stacked_arrays or column_structure is None:
                    print(f"Could not find predictions for view: {view}")
                    continue

                # Stack all arrays along the third dimension
                stacked_arrays = np.stack(stacked_arrays, axis=-1)

                # Compute the mean/median along the third dimension
                if mode == 'ensemble_mean':
                    aggregated_array = np.nanmean(stacked_arrays, axis=-1)
                elif mode == 'ensemble_median':
                    aggregated_array = np.nanmedian(stacked_arrays, axis=-1)
                elif mode == 'eks_singleview':
                    aggregated_array = run_eks_singleview(markers_list= stacked_dfs, keypoint_names=keypoint_names) # need to implement this function 
                else:
                    print(f"Invalid mode: {mode}")
                    continue

                # Create a new DataFrame with the aggregated data
                result_df = pd.DataFrame(
                    data=aggregated_array,
                    index=df.index,
                    columns=column_structure
                )

                result_df.loc[:,("set", "", "")] = "train"

                preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
                result_df.to_csv(preds_file)
                new_predictions_files.append(preds_file)
                print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")

                # Update cfg_lp for each view specifically
                cfg_lp_view = cfg_lp.copy()
                if view == 'bot':
                    cfg_lp_view.data.csv_file = ['CollectedData_bot_new.csv']
                elif view == 'top':
                    cfg_lp_view.data.csv_file = ['CollectedData_top_new.csv']
                cfg_lp_view.data.view_names = [view]

                try:
                    compute_metrics(cfg=cfg_lp_view, preds_file=preds_file, data_module=None)
                    print(f"Successfully computed metrics for {preds_file}")
                except Exception as e:
                    print(f"Error computing metrics\n{e}")

        else:
            #new_predictions_files = [] # check if I need that
            for view in views: 
                stacked_arrays = []
                stacked_dfs = []
                column_structure = None
                base_keypoints = []
                keypoint_names = []

                for seed_dir in seed_dirs:
                    pred_file = os.path.join(
                        seed_dir,
                        inference_dir,
                        f'180607_004_{view}.csv' # I will have to modify this to be more general
                    )
                    if os.path.exists(pred_file):
                        df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
                        if column_structure is None:
                            column_structure = df.loc[:, df.columns.get_level_values(2).isin(['x', 'y', 'likelihood'])].columns
                            #Extract keypoint names in the original order from the second level of the MultiIndex
                            keypoint_names = list(dict.fromkeys(column_structure.get_level_values(1)))
                            print(f'Keypoint names are {keypoint_names}')
                        # Select only numeric columns (x, y, likelihood)
                        numeric_cols = df.loc[:, column_structure]
                        # Convert DataFrame to a 2D array (numeric values only)
                        stacked_arrays.append(numeric_cols.to_numpy())
                        # create stacked_dfs for eks_singleview --> this will be the markers list 
                        stacked_dfs.append(numeric_cols) # this will be markers list 
                        print(f'the shape of the stacked arrays is {np.shape(stacked_arrays)}')
                        print(f'the shape of the stacked dfs is {stacked_dfs[0].shape}')
                    else:
                        print(f"Warning: Could not find predictions file: {pred_file}")
                
                if not stacked_arrays or column_structure is None:
                    print(f"Could not find predictions for view: {view}")
                    continue            
                # Stack all arrays along the third dimension
                stacked_arrays = np.stack(stacked_arrays, axis=-1)

                # Compute the mean/median along the third dimension
                if mode == 'ensemble_mean':
                    aggregated_array = np.nanmean(stacked_arrays, axis=-1)
                elif mode == 'ensemble_median':
                    aggregated_array = np.nanmedian(stacked_arrays, axis=-1)
                elif mode == 'eks_singleview':
                    aggregated_array = run_eks_singleview(markers_list= stacked_dfs, keypoint_names=keypoint_names) # need to implement this function 
                else:
                    print(f"Invalid mode: {mode}")
                    continue

                # Create a new DataFrame with the aggregated data
                result_df = pd.DataFrame(
                    data=aggregated_array,
                    index=df.index,
                    columns=column_structure
                )

                #result_df.loc[:,("set", "", "")] = "train"

                preds_file = os.path.join(output_dir, f'180607_004_{view}.csv')
                result_df.to_csv(preds_file)
                #new_predictions_files.append(preds_file)
                print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")

                print((f" I don't need to update the cfg_lp for each view specifically for the inference directory: {inference_dir}"))
                print((f" I don't need to compute metrics for the inference directory: {inference_dir}"))





    # base_dir = os.path.dirname(results_dir)
    # ensemble_dir = os.path.join(
    #     base_dir,
    #     f"{model_type}_{n_labels}_{seed_range[0]}-{seed_range[1]}"
    # )
    # output_dir = os.path.join(ensemble_dir, mode)
    # os.makedirs(output_dir, exist_ok=True)

    # seed_dirs = [
    #     os.path.join(base_dir, f"{model_type}_{n_labels}_{seed}")
    #     for seed in range(seed_range[0], seed_range[1] + 1)
    # ]
    
    # print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]} for {inference_dirs} " )

    


    # #new_predictions_files = []

    # #print(f"Post-processing ensemble predictions for {model_type} {n_labels} {seed_range[0]}-{seed_range[1]}")


    # for view in views: 
    #     stacked_arrays = []
    #     stacked_dfs = []
    #     column_structure = None
    #     base_keypoints = []
    #     keypoint_names = []
        
    #     for seed_dir in seed_dirs:
    #         pred_file = os.path.join(
    #             seed_dir,
    #             'videos-for-each-labeled-frame',
    #             f'predictions_{view}_new.csv'
    #         )
    #         if os.path.exists(pred_file):
    #             df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
    #             if column_structure is None:
    #                 column_structure = df.loc[:, df.columns.get_level_values(2).isin(['x', 'y', 'likelihood'])].columns
    #                 #Extract keypoint names in the original order from the second level of the MultiIndex
    #                 keypoint_names = list(dict.fromkeys(column_structure.get_level_values(1)))
    #                 #keypoint_names = list(column_structure.get_level_values(1)) # keep all repetitions 
    #                 # base_keypoints = list(dict.fromkeys(column_structure.get_level_values(1)))
    #                 # print(f' basekeypoint names are {base_keypoints}')
                    
    #                 print(f'Keypoint names are {keypoint_names}')

    #             # Select only numeric columns (x, y, likelihood)
    #             numeric_cols = df.loc[:, column_structure]

    #             # Convert DataFrame to a 2D array (numeric values only)
    #             stacked_arrays.append(numeric_cols.to_numpy())
    #             # create stacked_dfs for eks_singleview --> this will be the markers list 
    #             stacked_dfs.append(numeric_cols) # this will be markers list 
    #             print(f'the shape of the stacked arrays is {np.shape(stacked_arrays)}')
    #             print(f'the shape of the stacked dfs is {stacked_dfs[0].shape}')

    #         else:
    #             print(f"Warning: Could not find predictions file: {pred_file}")

    #     if not stacked_arrays or column_structure is None:
    #         print(f"Could not find predictions for view: {view}")
    #         continue

    #     # Stack all arrays along the third dimension
    #     stacked_arrays = np.stack(stacked_arrays, axis=-1)

    #     # Compute the mean/median along the third dimension
    #     if mode == 'ensemble_mean':
    #         aggregated_array = np.nanmean(stacked_arrays, axis=-1)
    #     elif mode == 'ensemble_median':
    #         aggregated_array = np.nanmedian(stacked_arrays, axis=-1)
    #     elif mode == 'eks_singleview':
    #          aggregated_array = run_eks_singleview(markers_list= stacked_dfs, keypoint_names=keypoint_names) # need to implement this function 
    #     else:
    #         print(f"Invalid mode: {mode}")
    #         continue

    #     # Create a new DataFrame with the aggregated data
    #     result_df = pd.DataFrame(
    #         data=aggregated_array,
    #         index=df.index,
    #         columns=column_structure
    #     )

    #     result_df.loc[:,("set", "", "")] = "train"

    #     preds_file = os.path.join(output_dir, f'predictions_{view}_new.csv')
    #     result_df.to_csv(preds_file)
    #     new_predictions_files.append(preds_file)
    #     print(f"Saved ensemble {mode} predictions for {view} view to {preds_file}")

    #     # Update cfg_lp for each view specifically
    #     cfg_lp_view = cfg_lp.copy()
    #     if view == 'bot':
    #         cfg_lp_view.data.csv_file = ['CollectedData_bot_new.csv']
    #     elif view == 'top':
    #         cfg_lp_view.data.csv_file = ['CollectedData_top_new.csv']
    #     cfg_lp_view.data.view_names = [view]

    #     try:
    #         compute_metrics(cfg=cfg_lp_view, preds_file=preds_file, data_module=None)
    #         print(f"Successfully computed metrics for {preds_file}")
    #     except Exception as e:
    #         print(f"Error computing metrics\n{e}")



def run_eks_singleview(
    markers_list: List[pd.DataFrame],
    keypoint_names : List[str], # can't I take it from the configrations?
    blocks : list = [], # will need to take care of that 
    avg_mode: str = 'median',
    var_mode : str = 'confidence_weighted_var',
) -> np.ndarray:

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

    # Select only the essential columns (x, y, likelihood) for each keypoint
    result = df_smoothed.loc[:, df_smoothed.columns.get_level_values(2).isin(['x', 'y', 'likelihood'])].to_numpy()
    
    return result


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
