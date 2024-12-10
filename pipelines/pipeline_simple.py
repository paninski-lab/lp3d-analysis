import argparse
import os

from lp3d_analysis.io import load_cfgs
from lp3d_analysis.train import train_and_infer


# TODO
# - before train_and_infer, will be nice to put cfg updates in their own function

VALID_MODEL_TYPES = [
    'supervised',
    'context',
]

def pipeline(config_file: str):

    # -------------------------------------------
    # Setup
    # -------------------------------------------

    # load cfg (pipeline yaml) and cfg_lp (lp yaml)
    cfg_pipe, cfg_lp = load_cfgs(config_file)  # cfg_lp is a DictConfig, cfg is not
    
    # Define + create directories
    data_dir = cfg_lp.data.data_dir
    pipeline_script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(pipeline_script_dir, f'../../outputs/{os.path.basename(data_dir)}')

    # -------------------------------------------------------------------------------------
    # Train ensembles
    # -------------------------------------------------------------------------------------

    if cfg_pipe.train_networks.run:
        for model_type in cfg_pipe.train_networks.model_types:
            for n_hand_labels in cfg_pipe.train_networks.n_hand_labels:
                for rng_seed in cfg_pipe.train_networks.ensemble_seeds:
                    print(
                        f'fitting {model_type} model (rng_seed={rng_seed}) with {n_hand_labels}' 
                        f'hand labels'
                    )
                    # specify output directory
                    results_dir = os.path.join(  # @lenny can update this how you see fit
                        outputs_dir, cfg_pipe.intermediate_results_dir, 
                        f'{model_type}-{n_hand_labels}-{rng_seed}',
                    )
                    # update cfg_lp
                    cfg_lp_copy = cfg_lp.copy()
                    cfg_lp_copy.training.rng_seed_data_pt = rng_seed
                    cfg_lp_copy.training.rng_seed_model_pt = rng_seed
                    cfg_lp_copy.training.train_frames = n_hand_labels
                    if model_type == 'supervised':
                        cfg_lp_copy.model.model_type = 'heatmap'
                        cfg_lp_copy.model.losses_to_use = []
                    elif model_type == 'context':
                        cfg_lp_copy.model.model_type = 'heatmap_mhcrnn'
                        cfg_lp_copy.model.losses_to_use = []
                    else:
                        raise ValueError(
                            f'{model_type} is not a valid model type in pipeline cfg; must choose'
                            f'from {VALID_MODEL_TYPES} or add a new model type'
                        )
                    # Main function call
                    train_and_infer(
                        cfg_pipe=cfg_pipe.copy(),
                        cfg_lp=cfg_lp_copy,
                        data_dir=data_dir,
                        results_dir=results_dir,
                        inference_dirs=cfg_pipe.train_networks.inference_dirs,
                        overwrite=cfg_pipe.train_networks.overwrite,
                    )

#     # # # -------------------------------------------------------------------------------------
#     # # # Post-process network outputs to generate potential pseudo labels (chosen in next step)
#     # # # -------------------------------------------------------------------------------------
#     pp_dir = os.path.join(
#         outputs_dir,
#         'post-processors',
#         f"{cfg['pseudo_labeler']}_rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
#     )
#     pseudo_labeler = cfg["pseudo_labeler"]
#     # Collect input eks csv paths from video names; skip existing
#     eks_csv_paths = collect_missing_eks_csv_paths(video_names, pp_dir)
#     print(f'Post-processing the following videos using {pseudo_labeler}: {eks_csv_paths}')
#     # ||| Main EKS function call ||| pipeline_eks will also handle ensemble_mean baseline
#     if pseudo_labeler == "eks" or pseudo_labeler == "ensemble_mean":
#         pipeline_eks(
#             input_csv_names=eks_csv_paths,
#             input_dir=networks_dir,
#             data_type=cfg["data_type"],
#             pseudo_labeler=pseudo_labeler,
#             cfg_lp=cfg_lp.copy(),
#             results_dir=pp_dir
#         )

#     # # -------------------------------------------------------------------------------------
#     # # run inference on OOD snippets (if specified) -- using network models
#     # # -------------------------------------------------------------------------------------
#     dataset_name = os.path.basename(data_dir)
#     if cfg["ood_snippets"]:
#         print(f'Starting OOD snippet analysis for {dataset_name}')
#         pipeline_ood_snippets(
#             cfg=cfg,
#             cfg_lp=cfg_lp,
#             data_dir=data_dir,
#             networks_dir=networks_dir,
#             pp_dir=pp_dir,
#             pseudo_labeler=pseudo_labeler
#         )

#     # # -------------------------------------------------------------------------------------
#     # # select frames to add to the dataset
#     # # -------------------------------------------------------------------------------------
#     selection_strategy = cfg["selection_strategy"]
#     print(
#         f'Selecting {cfg["n_pseudo_labels"]} pseudo-labels from {num_videos} '
#         f'{cfg["pseudo_labeler"]} outputs using ({selection_strategy} strategy)'
#     )
#     hand_labels = pd.read_csv(subsample_path, header=[0, 1, 2], index_col=0)
#     # Process each ensemble seed
#     for k in cfg["ensemble_seeds"]:
#         # Initialize seed_labels with hand labels for this seed
#         seed_labels = hand_labels.copy()
#         combined_csv_filename = (
#             f"CollectedData_hand={cfg['n_hand_labels']}"
#             f"_pseudo={cfg['n_pseudo_labels']}_k={k}_{selection_strategy}.csv"
#         )
#         combined_csv_path = os.path.join(hand_pseudo_combined, combined_csv_filename)

#         # Check if frame selection has already been done
#         if os.path.exists(combined_csv_path):
#             print(
#                 f'Selected frames already exist at {combined_csv_path}. '
#                 f'Skipping frame selection for rng{k}.'
#             )
#             seed_labels = pd.read_csv(combined_csv_path, header=[0, 1, 2], index_col=0)

#         else:
#             print(f'Selecting pseudo-labels using a {selection_strategy} strategy.')

#             if selection_strategy == 'random':
#                 seed_labels = select_frames_random(
#                     cfg=cfg.copy(),
#                     k=k,
#                     data_dir=data_dir,
#                     num_videos=num_videos,
#                     pp_dir=pp_dir,
#                     labeled_data_dir=labeled_data_dir,
#                     seed_labels=seed_labels
#                 )

#             elif selection_strategy == 'hand':
#                 seed_labels = select_frames_hand(
#                     unsampled_path=unsampled_path,
#                     n_frames_to_select=cfg['n_pseudo_labels'],
#                     k=k,
#                     seed_labels=seed_labels
#                 )
                
#             elif selection_strategy == 'frame_selection':
#                 frame_selection_path = os.path.join(source_dir, (
#                     f"outputs/{os.path.basename(data_dir)}/hand={cfg_lp.training.train_frames}_"
#                     f"pseudo={cfg['n_pseudo_labels']}/post-quality-frame/"
#                 ))

#                 final_selected_frames_path = select_frames_strategy_pipeline(
#                     cfg=cfg.copy(),
#                     cfg_lp=cfg_lp.copy(),
#                     data_dir=data_dir,
#                     source_dir=source_dir,
#                     frame_selection_path=frame_selection_path,
#                 )

#                 seed_labels = process_and_export_frame_selection(
#                     cfg=cfg.copy(),
#                     cfg_lp=cfg_lp.copy(),
#                     data_dir=data_dir,
#                     labeled_data_dir=labeled_data_dir,
#                     final_selected_frames_path=final_selected_frames_path,
#                     seed_labels=seed_labels
#                 )
            
#             # Filter out columns where the second level of the header contains 'zscore', 'nll', or 'ensemble_std'
#             columns_to_keep = [
#                 col for col in seed_labels.columns
#                 if not any(substring in col[2] for substring in ['zscore', 'nll', 'ensemble_std'])
#             ]

#             # Select the columns to keep
#             seed_labels_filtered = seed_labels[columns_to_keep]

#             # Export the filtered DataFrame to CSV
#             seed_labels_filtered.to_csv(combined_csv_path)
#             print(
#                 f"Saved combined hand labels and pseudo labels for seed {k} to "
#                 f"{combined_csv_path}"
#             )
            
#         # Check number of labels for this seed
#         expected_total_labels = cfg['n_hand_labels'] + cfg["n_pseudo_labels"]
#         if seed_labels.shape[0] != expected_total_labels:
#             print(
#                 f"Warning: Number of labels for seed {k} ({seed_labels.shape[0]}) "
#                 f"does not match expected count ({expected_total_labels})"
#             )
#         else:
#             print(f"Label count verified for seed {k}: {seed_labels.shape[0]} labels")

#         # # -------------------------------------------------------------------------------------
#         # # Train models on expanded dataset
#         # # -------------------------------------------------------------------------------------

#         csv_prefix = (
#             f"hand={cfg['n_hand_labels']}_rng={k}_"
#             f"pseudo={cfg['n_pseudo_labels']}_"
#             f"{cfg['pseudo_labeler']}_{cfg['selection_strategy']}_"
#             f"rng={cfg['ensemble_seeds'][0]}-{cfg['ensemble_seeds'][-1]}"
#         )
#         results_dir = os.path.join(aeks_dir, f"rng{k}")

#         # Run train_and_infer with the combined hand labels and pseudo labels
#         train_and_infer(
#             cfg=cfg.copy(),
#             cfg_lp=cfg_lp.copy(),
#             k=k,
#             data_dir=data_dir,
#             results_dir=results_dir,
#             csv_prefix=csv_prefix,
#             new_labels_csv=combined_csv_path,  # Use the combined CSV file for this seed
#             n_train_frames=expected_total_labels
#         )

#         print(
#             f"Completed training and inference for seed {k} "
#             f"using combined hand labels and pseudo labels"
#         )

#     print("Completed training and inference for all seeds using expanded datasets")

#     # # # -------------------------------------------------------------------------------------
#     # # # Run EKS on expanded dataset inferences
#     # # # -------------------------------------------------------------------------------------
#     pseudo_labeler = 'eks'
#     # Collect input csv names from video names; skip existing ones
#     eks_csv_paths = collect_missing_eks_csv_paths(video_names, aeks_eks_dir)
#     print(f'Post-processing the following videos using {pseudo_labeler}: {eks_csv_paths}')
#     # ||| Main second round EKS function call |||
#     pipeline_eks(
#         input_csv_names=eks_csv_paths,
#         input_dir=aeks_dir,
#         data_type=cfg["data_type"],
#         pseudo_labeler=pseudo_labeler,
#         cfg_lp=cfg_lp.copy(),
#         results_dir=results_dir
#     )

#     # # -------------------------------------------------------------------------------------
#     # # run inference on OOD snippets (if specified) -- using network models
#     # # -------------------------------------------------------------------------------------
#     dataset_name = os.path.basename(data_dir)
#     if cfg["ood_snippets"]:
#         print(f'Starting OOD snippet analysis for {dataset_name}')
#         pipeline_ood_snippets(
#             cfg=cfg,
#             cfg_lp=cfg_lp,
#             data_dir=data_dir,
#             networks_dir=aeks_dir,
#             pp_dir=aeks_eks_dir,
#             pseudo_labeler=pseudo_labeler
#         )
# # ------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------
# # ------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='absolute path to .yaml configuration file',
        type=str,
    )
    args = parser.parse_args()

    pipeline(args.config)
