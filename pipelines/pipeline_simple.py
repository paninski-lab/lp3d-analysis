import argparse
import os
import copy

from omegaconf import OmegaConf

from lp3d_analysis.io import load_cfgs
from lp3d_analysis.train import train_and_infer
from lp3d_analysis.utils import extract_ood_frame_predictions
from lp3d_analysis.post_process import post_process_ensemble_labels, post_process_ensemble_videos



# TODO
# - before train_and_infer, will be nice to put cfg updates in their own function
# - remove get_callbacks from lp3d_analysis.train.py 
# - replace train function with one from LP
# - faster inference with OOD videos: don't rebuild the model every time

VALID_MODEL_TYPES = [
    'supervised',
    'context',
]


def pipeline(config_file: str, for_seed: int | None = None) -> None:

    # -------------------------------------------
    # Setup
    # -------------------------------------------

    # load cfg (pipeline yaml) and cfg_lp (lp yaml)
    cfg_pipe, cfg_lp = load_cfgs(config_file)  # cfg_lp is a DictConfig, cfg_pipe is not
    
    # Define + create directories
    data_dir = cfg_lp.data.data_dir
    pipeline_script_dir = os.path.dirname(os.path.abspath(__file__)) # always have the outputs in one spesific spot. --> if want to run in a different studio
    outputs_dir = cfg_pipe.outputs_dir 
    #os.path.join(pipeline_script_dir, f'../../outputs/{os.path.basename(data_dir)}')

    # -------------------------------------------------------------------------------------
    # Train ensembles
    # -------------------------------------------------------------------------------------

    if cfg_pipe.train_networks.run:
        for model_type in cfg_pipe.train_networks.model_types:
            for n_hand_labels in cfg_pipe.train_networks.n_hand_labels:
                for rng_seed in cfg_pipe.train_networks.ensemble_seeds:
                    if for_seed is not None and for_seed != rng_seed:
                        continue
                    print(
                        f'fitting {model_type} model (rng_seed={rng_seed}) with {n_hand_labels}' 
                        f'hand labels'
                    )
                    # specify output directory
                    results_dir = os.path.join(  # @lenny can update this how you see fit
                        outputs_dir, cfg_pipe.intermediate_results_dir, 
                        f'{model_type}_{n_hand_labels}_{rng_seed}',
                    )
                    cfg_lp_copy = make_model_cfg(cfg_lp, cfg_pipe, data_dir, model_type, n_hand_labels, rng_seed)
                    # Main function call
                    train_and_infer(
                        cfg_lp=cfg_lp_copy,
                        model_dir=results_dir,
                        inference_dirs=cfg_pipe.train_networks.inference_dirs,
                        overwrite=cfg_pipe.train_networks.overwrite,
                    )

                    if 'videos-for-each-labeled-frame' in cfg_pipe.train_networks.inference_dirs:
                        # Clean up/reorganize OOD data
                        extract_ood_frame_predictions(
                            cfg_lp=cfg_lp_copy,
                            data_dir=data_dir,
                            results_dir=results_dir,
                            overwrite=cfg_pipe.train_networks.overwrite,
                            video_dir='videos-for-each-labeled-frame',
                        )

    # The rest of the pipeline only runs when you run without --for_seed.
    if for_seed is not None:
        return

    for mode, mode_config in cfg_pipe.post_processing_videos.items():
        for model_type in cfg_pipe.train_networks.model_types:
            for n_hand_labels in cfg_pipe.train_networks.n_hand_labels:
                if mode_config.run: # if the mode is mean or median or eks_singleview 
                    #print(f"Debug: Preparing to run {mode} for {model_type} with seed range {cfg_pipe.train_networks.ensemble_seeds}"
                    post_process_ensemble_videos(
                        cfg_lp=cfg_lp_copy.copy(),
                        results_dir=results_dir,
                        model_type=model_type,
                        n_labels= n_hand_labels,
                        seed_range=(cfg_pipe.train_networks.ensemble_seeds[0], cfg_pipe.train_networks.ensemble_seeds[-1]),
                        views= list(cfg_lp.data.view_names), # before it was not a list... 
                        mode=mode,
                        inference_dirs=cfg_pipe.train_networks.inference_dirs,
                        overwrite=mode_config.overwrite,
                    )

    for mode, mode_config in cfg_pipe.post_processing_labels.items():
        for model_type in cfg_pipe.train_networks.model_types:
            for n_hand_labels in cfg_pipe.train_networks.n_hand_labels:
                if mode_config.run: # if the mode is mean or median or eks_singleview 
                    #print(f"Debug: Preparing to run {mode} for {model_type} with seed range {cfg_pipe.train_networks.ensemble_seeds}"
                    post_process_ensemble_labels( # remember I changed that for a second 
                        cfg_lp=cfg_lp_copy.copy(),
                        results_dir=results_dir,
                        model_type=model_type,
                        n_labels= n_hand_labels,
                        seed_range=(cfg_pipe.train_networks.ensemble_seeds[0], cfg_pipe.train_networks.ensemble_seeds[-1]),
                        views= list(cfg_lp.data.view_names), # before it was not a list... 
                        mode=mode,
                        inference_dirs=cfg_pipe.train_networks.inference_dirs,
                        overwrite=mode_config.overwrite,
                    )

                # for mode, mode_config in cfg_pipe.post_processing.items():
                #     if mode_config.run: # if the mode is mean or median or eks_singleview 
                #         #print(f"Debug: Preparing to run {mode} for {model_type} with seed range {cfg_pipe.train_networks.ensemble_seeds}"
                #         post_process_ensemble(
                #             cfg_lp=cfg_lp_copy.copy(),
                #             results_dir=results_dir,
                #             model_type=model_type,
                #             n_labels= n_hand_labels,
                #             seed_range=(cfg_pipe.train_networks.ensemble_seeds[0], cfg_pipe.train_networks.ensemble_seeds[-1]),
                #             views= list(cfg_lp.data.view_names), # before it was not a list... 
                #             mode=mode,
                #             inference_dirs=cfg_pipe.train_networks.inference_dirs,
                #             overwrite=mode_config.overwrite,
                #         )
       

def make_model_cfg(cfg_lp, cfg_pipe, data_dir, model_type, n_hand_labels, rng_seed):
    # update cfg_lp
    cfg_overrides = [{
        "data": {
            "data_dir": data_dir,
        },
        "training": {
            "rng_seed_data_pt": rng_seed,
            "rng_seed_model_pt": rng_seed,
            "train_frames": n_hand_labels,
        }
    }]
    if model_type == 'supervised':
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap",
                "losses_to_use": [],
            },
        })
    elif model_type == 'context':
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_mhcrnn",
                "losses_to_use": [],
            },
        })
    elif model_type == 'semisupervised':
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap",
                "losses_to_use": ["pca_multiview"],
            },
        })
    
    else:
        raise ValueError(
            f'{model_type} is not a valid model type in pipeline cfg; must choose'
            f'from {VALID_MODEL_TYPES} or add a new model type'
        )
    # Parse params from config
    min_steps = cfg_pipe.train_networks.min_steps
    max_steps = cfg_pipe.train_networks.max_steps
    milestone_steps = cfg_pipe.train_networks.milestone_steps
    unfreezing_step = cfg_pipe.train_networks.unfreezing_step
    val_check_interval = cfg_pipe.train_networks.val_check_interval
    cfg_overrides.append({
        "training": {
            "min_steps": min_steps,
            "max_steps": max_steps,
            "min_epochs": None,
            "max_epochs": None,
            "val_check_interval": val_check_interval,
            "check_val_every_n_epoch": None,
            "unfreezing_step": unfreezing_step,
            "unfreezing_epoch": None,
            "lr_scheduler_params": {
                "multisteplr": {
                    "milestone_steps": milestone_steps,
                    "milestones": None,
                }
            },
        },
        "eval": {
            "predict_vids_after_training": False,
        }
    })
    cfg_lp_copy = OmegaConf.merge(cfg_lp, *cfg_overrides)
    del cfg_lp_copy.training.min_epochs
    del cfg_lp_copy.training.max_epochs
    del cfg_lp_copy.training.check_val_every_n_epoch
    del cfg_lp_copy.training.unfreezing_epoch
    del cfg_lp_copy.training.lr_scheduler_params.multisteplr.milestones
    return cfg_lp_copy


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='absolute path to .yaml configuration file',
        type=str,
    )
    parser.add_argument(
        '--for_seed',
        help='only run a specific seed (useful for distributing training on lightning jobs)',
        type=int,
    )
    args = parser.parse_args()

    pipeline(args.config, args.for_seed)
