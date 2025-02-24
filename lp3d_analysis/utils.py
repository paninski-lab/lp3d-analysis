import copy
import os
import numpy as np
import pandas as pd

from lightning_pose.utils.scripts import compute_metrics_single

from omegaconf import DictConfig


#TODO 
# 1. need to make a decision about the naming of the output file and how to access the correct model name because it is not part of the results_dir 
# 2. change names of variables so will look better in the code and make more sense 

def extract_ood_frame_predictions(
    cfg_lp: DictConfig,
    data_dir: str,
    results_dir: str,
    overwrite: bool,
    video_dir: str,
) -> None:

    new_csv_files = [f for f in os.listdir(data_dir) if f.endswith('_new.csv')]

    for csv_file in new_csv_files:
        # load original csv 
        file_path = os.path.join(data_dir, csv_file)
        original_df = pd.read_csv(file_path, header=[0,1,2], index_col=0)
        original_index = original_df.index  

        # load each of the new csv files and iterate through the index 
        prediction_name = '_'.join(csv_file.split('_')[1:])
        preds_file = os.path.join(results_dir, video_dir , f'predictions_{prediction_name}') 
        #print(f"the preds file is {preds_file}")
        if os.path.exists(preds_file) and not overwrite:
            print(f'Predictions file {preds_file} already exists. Skipping.')
            continue
        
        results_list = []
        #file_path = os.path.join(data_dir, csv_file)
        #df = pd.read_csv(file_path, header=[0,1,2], index_col=0)
        
    

        # for img_path in df.index:
        for img_path in original_index:
            # process the paths 
            relative_img_path = '/'.join(img_path.split('/')[1:]) # removed 'labeled-data/'
            snippet_path = relative_img_path.replace('png', 'mp4')
            
            # Load the 51-frame csv file 
            snippet_file = os.path.join(results_dir, video_dir , snippet_path.replace('mp4', 'csv'))
            if os.path.exists(snippet_file):
                snippet_df = pd.read_csv(snippet_file, header=[0,1,2], index_col=0)

                # extract center frame 
                assert snippet_df.shape[0] & 2 != 0 # ensure odd number of frames 
                idx_frame = int(np.floor(snippet_df.shape[0] / 2))
                
                # create results with original image path as index 
                result = snippet_df[snippet_df.index == idx_frame].rename(index={idx_frame: img_path})
                results_list.append(result)

        # combine all results 
        if results_list:
            results_df = pd.concat(results_list)
            #results_df.sort_index(inplace=True)
            results_df = results_df.reindex(original_index)

            # Add "set" column so this df is interpreted as labeled data predictions
            results_df.loc[:,("set", "", "")] = "train"
            results_df.to_csv(preds_file)
            print(f'Saved predictions to {preds_file}')

            # make sure I don't make any changes to the original cfg_lp
            cfg_lp_copy = copy.deepcopy(cfg_lp)
            cfg_lp_copy.data.csv_file = csv_file
            print(f"the cfg_lp_copy is {cfg_lp_copy}")
        
            try:
                compute_metrics_single(
                    cfg=cfg_lp_copy,
                    labels_file = csv_file,
                    preds_file=preds_file,
                    data_module=None
                )
                print(f"Succesfully computed metrics for {preds_file}")
            except Exception as e:
                print(f"Error computing metrics\n{e}")

        
    # look for all files that end in _new.csv -> these are OOD labels
     # loop through these
     #load the csv file and iterate through the rows/index
    # loop through these
    # for each, load the csv file, and iterate through the rows/index
    # 'labeled-data/<vid_name>/img<#>.png'
    # s = 'labeled-data/vid_name/img0000.png'
    # s2 = '/'.join(s.split('/')[1:])
    # s3 = s2.replace('png', 'mp4')
    # load 51-frame csv file
    # extract center frame
    # put in dataframe
    # save out predictions_<cam_name>.csv
    # compute pixel

