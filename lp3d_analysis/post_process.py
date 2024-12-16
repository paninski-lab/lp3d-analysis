import os
import pandas as pd
import numpy as np

from omegaconf import DictConfig
from typing import List, Literal
from pathlib import Path

from lightning_pose.utils.scripts import (
    compute_metrics,
)

#TODO
#1. in the variables for the function pose_process_ensemble, try to change the issue with the mode variable so we can add the eks mode and things like that 
#2. Change the way we are using the cfp_lp.data.csv_file - it will make more sense in the function 
#3 change variables names so it will make more sense 



def post_process_ensemble(
    cfg_lp: DictConfig,
    results_dir: str,
    model_type: str,
    n_labels: int,
    seed_range: tuple[int, int],
    views: list[str], 
    mode: Literal['ensemble_mean', 'ensemble_median'],
    overwrite: bool,
) -> None:

    base_dir = os.path.dirname(results_dir)
    ensemble_dir = os.path.join(
        base_dir,
        f"{model_type}_{n_labels}_{seed_range[0]}-{seed_range[1]}"
    )
    output_dir = os.path.join(ensemble_dir, mode)
    os.makedirs(output_dir, exist_ok=True)

    seed_dirs = [
        os.path.join(base_dir, f"{model_type}_{n_labels}_{seed}")
        for seed in range(seed_range[0], seed_range[1] + 1)
    ]

    new_predictions_files = []

    for view in views:
        stacked_arrays = []
        column_structure = None
        
        for seed_dir in seed_dirs:
            pred_file = os.path.join(
                seed_dir,
                'videos-for-each-labeled-frame',
                f'predictions_{view}_new.csv'
            )
            if os.path.exists(pred_file):
                df = pd.read_csv(pred_file, header=[0, 1, 2], index_col=0)
                if column_structure is None:
                    column_structure = df.loc[:, df.columns.get_level_values(2).isin(['x', 'y', 'likelihood'])].columns
                
                # Select only numeric columns (x, y, likelihood)
                numeric_cols = df.loc[:, column_structure]
                
                # Convert DataFrame to a 2D array (numeric values only)
                stacked_arrays.append(numeric_cols.to_numpy())
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

        cfg_lp.data.csv_file = ['CollectedData_top_new.csv', 'CollectedData_bot_new.csv'] # try to 
        cfg_lp.data.view_names = [view]

        try:
            compute_metrics(cfg=cfg_lp, preds_file=preds_file, data_module=None)
            print(f"Successfully computed metrics for {pred_file}")
        except Exception as e:
            print(f"Error computing metrics\n{e}")
        


    
    

    





#     cfg_lp.data.csv_file = predictions_new_ensembles
#     if predictions_new_ensembles and not overwrite:
#         for preds_file in predictions_new_ensembles:
#             try:
#                 compute_metrics(cfg=cfg_lp, preds_file=preds_file, data_module=None)
#                 print(f"Succesfully computed metrics for {preds_file}")
#             except Exception as e:
#                 print(f"Error computing metrics for {preds_file}\n{e}")
#     else: 
#         print("No new predictions to compute metrics on")




        # cfg_lp.data.csv_file = preds_file
        # print(cfg_lp.data.csv_file)
        # cfg_lp.data.view_names = [view] 

        # try:
        #     compute_metrics(cfg=cfg_lp, preds_file= preds_file , data_module=None)
        #     print(f"Succesfully computed metrics for {preds_file}")
        # except Exception as e:
        #     print(f"Error computing metrics\n{e}")






        #         all_predictions.append(df)
        #     else:
        #         print(f"Warning: Could not find predictions file: {pred_file}")
        
        # if not all_predictions:
        #     print(f"Could not find predictions for view: {view}")
        #     continue
        
        # combined_df = pd.concat(all_predictions, axis =0)
        # print(combined_df.head())



        # # group by any identifying columns ( assuming frame/ timestamps columns exists)
        # # modify these groupby columns based on csv structure 
        # group_cols = [col for col in combined_df]


    



#     pp_dir = os.path.join(
# #         outputs_dir,
# #         'post-processors',
# #         f"{cfg['pseudo_labeler']}_rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
# #     )




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
