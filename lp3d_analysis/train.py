"""Functions for training."""

import gc
import glob
import os
import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from lightning_pose.utils import pretty_print_cfg, pretty_print_str
from lightning_pose.utils.io import (check_video_paths,
                                     return_absolute_data_paths,
                                     return_absolute_path)
from lightning_pose.utils.predictions import (predict_dataset,
                                              predict_single_video)
from lightning_pose.utils.scripts import (  # get_callbacks,
    calculate_train_batches, 
    compute_metrics,
    get_data_module, 
    get_dataset,
    get_imgaug_transform, 
    get_loss_factories, 
    get_model,
)
from moviepy.editor import VideoFileClip
from omegaconf import DictConfig, OmegaConf
from typeguard import typechecked


@typechecked
def get_callbacks(
    cfg_lp: DictConfig,
    early_stopping=True,
    lr_monitor=True,
    ckpt_model=True,
    backbone_unfreeze=True,
) -> List:

    callbacks = []

    if early_stopping:
        early_stopping = pl.callbacks.EarlyStopping(
            monitor="val_supervised_loss",
            patience=cfg_lp.training.early_stop_patience,
            mode="min",
        )
        callbacks.append(early_stopping)

    if lr_monitor:
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")
        callbacks.append(lr_monitor)

    if ckpt_model:
        ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
            monitor="val_supervised_loss",
            mode="min",
            save_top_k=1,
            filename="best-checkpoint-step={step}-val_loss={val_supervised_loss:.2f}",
            save_last=True,
        )
        callbacks.append(ckpt_callback)

    if backbone_unfreeze:
        transfer_unfreeze_callback = pl.callbacks.BackboneFinetuning(
            unfreeze_backbone_at_epoch=cfg_lp.training.unfreezing_epoch,
            backbone_initial_ratio_lr=0.1,
            should_align=True,
            train_bn=True,
        )
        callbacks.append(transfer_unfreeze_callback)

    # we just need this callback for unsupervised models
    if (cfg_lp.model.losses_to_use != []) and (cfg_lp.model.losses_to_use is not None):
        anneal_weight_callback = AnnealWeight(**cfg_lp.callbacks.anneal_weight)
        callbacks.append(anneal_weight_callback)

    return callbacks


# TODO: matt, replace this with LP train function
def train(
    cfg_lp: DictConfig,
    results_dir: str,
    min_steps: int,
    max_steps: int,
    milestone_steps: List[int],
    val_check_interval: int,
) -> Tuple[str, pl.LightningDataModule, pl.Trainer]:

    MIN_STEPS = min_steps
    MAX_STEPS = max_steps
    MILESTONE_STEPS = milestone_steps

    # mimic hydra, change dir into results dir
    pwd = os.getcwd()
    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)

    # reset all seeds
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ----------------------------------------------------------------------------------
    # set up data/model objects
    # ----------------------------------------------------------------------------------

    pretty_print_cfg(cfg_lp)

    data_dir, video_dir = return_absolute_data_paths(data_cfg=cfg_lp.data)

    # imgaug transform
    imgaug_transform = get_imgaug_transform(cfg=cfg_lp)

    # dataset
    dataset = get_dataset(cfg=cfg_lp, data_dir=data_dir, imgaug_transform=imgaug_transform)

    # datamodule; breaks up dataset into train/val/test
    data_module = get_data_module(cfg=cfg_lp, dataset=dataset, video_dir=video_dir)

    data_module.setup()

    num_train_frames = len(data_module.train_dataset)

    step_per_epoch = num_train_frames / cfg_lp.training.train_batch_size
    print(f'step_per_epoch={step_per_epoch}')

    cfg_lp.training.max_epochs = int(MAX_STEPS / step_per_epoch)
    cfg_lp.training.min_epochs = int(MIN_STEPS / step_per_epoch)
    cfg_lp.training.lr_scheduler_params.multisteplr.milestones = \
        [int(s / step_per_epoch) for s in MILESTONE_STEPS]

    # build loss factory which orchestrates different losses
    loss_factories = get_loss_factories(cfg=cfg_lp, data_module=data_module)

    # model
    model = get_model(cfg=cfg_lp, data_module=data_module, loss_factories=loss_factories)

    # ----------------------------------------------------------------------------------
    # set up and run training
    # ----------------------------------------------------------------------------------

    # logger
    logger = pl.loggers.TensorBoardLogger("tb_logs", name=cfg_lp.model.model_name)

    callbacks = get_callbacks(cfg_lp, early_stopping=False)

    # calculate number of batches for both labeled and unlabeled data per epoch
    limit_train_batches = calculate_train_batches(cfg_lp, dataset)

    # set up trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=cfg_lp.training.max_epochs,
        min_epochs=cfg_lp.training.min_epochs,
        check_val_every_n_epoch=None,
        val_check_interval=val_check_interval,
        log_every_n_steps=cfg_lp.training.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        # limit_train_batches= int(1),
        limit_train_batches=limit_train_batches,
    )

    # train model!
    trainer.fit(model=model, datamodule=data_module)

    # save config file
    cfg_file_local = os.path.join(results_dir, "config.yaml")
    with open(cfg_file_local, "w") as fp:
        OmegaConf.save(config=cfg_lp, f=fp.name)

    # Ensure the directory exists
    os.makedirs(os.path.dirname(cfg_file_local), exist_ok=True)

    # ----------------------------------------------------------------------------------
    # post-training analysis: labeled frames
    # ----------------------------------------------------------------------------------
    hydra_output_directory = os.getcwd()
    print("Hydra output directory: {}".format(hydra_output_directory))

    # get best ckpt
    best_ckpt = os.path.abspath(trainer.checkpoint_callback.best_model_path)

    # check if best_ckpt is a file
    if not os.path.isfile(best_ckpt):
        raise FileNotFoundError("Cannot find checkpoint. Have you trained for too few epochs?")

    # make unaugmented data_loader if necessary
    if cfg_lp.training.imgaug != "default":
        cfg_pred = cfg_lp.copy()
        cfg_pred.training.imgaug = "default"
        imgaug_transform_pred = get_imgaug_transform(cfg=cfg_pred)
        dataset_pred = get_dataset(
            cfg=cfg_pred, data_dir=data_dir, imgaug_transform=imgaug_transform_pred)
        data_module_pred = get_data_module(
            cfg=cfg_pred, dataset=dataset_pred, video_dir=video_dir)
        data_module_pred.setup()
    else:
        data_module_pred = data_module

    # predict on all labeled frames (train/val/test)
    pretty_print_str("Predicting train/val/test images...")
    # compute and save frame-wise predictions
    preds_file = os.path.join(hydra_output_directory, "predictions.csv")
    predict_dataset(
        cfg=cfg_lp,
        trainer=trainer,
        model=model,
        data_module=data_module_pred,
        ckpt_file=best_ckpt,
        preds_file=preds_file,
    )
    # compute and save various metrics
    # for multiview, predict_dataset outputs one pred file per view.
    multiview_pred_files = [
        str(Path(hydra_output_directory) / p)
        for p in Path(hydra_output_directory).glob("predictions_*.csv")
    ]
    if len(multiview_pred_files) > 0:
        preds_file = multiview_pred_files
    # compute and save various metrics
    try:
        compute_metrics(cfg=cfg_lp, preds_file=preds_file, data_module=data_module_pred)
    except Exception as e:
        print(f"Error computing metrics\n{e}")

    # # ----------------------------------------------------------------------------------
    # # post-training analysis: predict on OOD labeled frames
    # # ----------------------------------------------------------------------------------
    # # update config file to point to OOD data
    # csv_file_ood = os.path.join(cfg_lp.data.data_dir, "CollectedData_new.csv")
    # if os.path.exists(csv_file_ood):
    #     cfg_ood = cfg_lp.copy()
    #     cfg_ood.data.csv_file = csv_file_ood
    #     cfg_ood.training.imgaug = "default"
    #     cfg_ood.training.train_prob = 1
    #     cfg_ood.training.val_prob = 0
    #     cfg_ood.training.train_frames = 1
    #     # build dataset/datamodule
    #     imgaug_transform_ood = get_imgaug_transform(cfg=cfg_ood)
    #     dataset_ood = get_dataset(
    #         cfg=cfg_ood, data_dir=data_dir, imgaug_transform=imgaug_transform_ood
    #     )
    #     data_module_ood = get_data_module(cfg=cfg_ood, dataset=dataset_ood, video_dir=video_dir)
    #     data_module_ood.setup()
    #     pretty_print_str("Predicting OOD images...")
    #     # compute and save frame-wise predictions
    #     preds_file_ood = os.path.join(hydra_output_directory, "predictions_new.csv")
    #     predict_dataset(
    #         cfg=cfg_ood,
    #         trainer=trainer,
    #         model=model,
    #         data_module=data_module_ood,
    #         ckpt_file=best_ckpt,
    #         preds_file=preds_file_ood,
    #     )
    #     # compute and save various metrics
    #     try:
    #         compute_metrics(
    #             cfg=cfg_ood, preds_file=preds_file_ood, data_module=data_module_ood
    #         )
    #     except Exception as e:
    #         print(f"Error computing metrics\n{e}")

    # # ----------------------------------------------------------------------------------
    # # post-training analysis: unlabeled videos
    # # ----------------------------------------------------------------------------------
    # if cfg_lp.eval.predict_vids_after_training:
    #     pretty_print_str("Predicting videos...")
    #     if cfg_lp.eval.test_videos_directory is None:
    #         filenames = []
    #     else:
    #         filenames = check_video_paths(return_absolute_path(cfg_lp.eval.test_videos_directory))
    #         pretty_print_str(
    #             f"Found {len(filenames)} videos to predict on "
    #             f"(in cfg.eval.test_videos_directory)"
    #         )

    #     for v, video_file in enumerate(filenames):
    #         assert os.path.isfile(video_file)
    #         pretty_print_str(f"Predicting video: {video_file}...")
    #         # get save name for prediction csv file
    #         video_pred_dir = os.path.join(hydra_output_directory, "video_preds")
    #         video_pred_name = os.path.splitext(os.path.basename(video_file))[0]
    #         prediction_csv_file = os.path.join(video_pred_dir, video_pred_name + ".csv")
    #         inference_with_metrics(
    #             video_file=video_file,
    #             ckpt_file=best_ckpt,
    #             cfg_lp=cfg_lp,
    #             preds_file=prediction_csv_file,
    #             data_module=data_module_pred,
    #             trainer=trainer,
    #         )

    # ----------------------------------------------------------------------------------
    # clean up
    # ----------------------------------------------------------------------------------
    # remove lightning logs
    shutil.rmtree(os.path.join(results_dir, "lightning_logs"), ignore_errors=True)

    # change directory back
    os.chdir(pwd)

    # clean up memory
    del imgaug_transform
    del dataset
    del data_module
    # del data_module_pred
    del loss_factories
    del model
    # del trainer
    gc.collect()
    torch.cuda.empty_cache()

    return best_ckpt, data_module_pred, trainer


def inference_with_metrics(
    video_file: str,
    cfg_lp: DictConfig,
    preds_file: str,
    ckpt_file: Optional[str] = None,
    data_module: Optional[callable] = None,
    trainer: Optional[pl.Trainer] = None,
    metrics: bool = True,
) -> pd.DataFrame:

    # compute predictions if they don't already exist
    if not os.path.exists(preds_file):
        preds_df = predict_single_video(
            video_file=video_file,
            ckpt_file=ckpt_file,
            cfg_file=cfg_lp,
            preds_file=preds_file,
            data_module=data_module,
            trainer=trainer,
        )
    else:
        preds_df = pd.read_csv(preds_file, header=[0, 1, 2], index_col=0)

    # compute and save various metrics
    if metrics:
        compute_metrics(cfg=cfg_lp, preds_file=preds_file, data_module=data_module)

    gc.collect()

    return preds_df


def train_and_infer(
    cfg_pipe: DictConfig,
    cfg_lp: DictConfig,
    data_dir: str,
    results_dir: str,
    inference_dirs: Optional[list] = None,
    csv_prefix: Optional[str] = None,
    new_labels_csv: Optional[str] = None,
    overwrite: bool = False,
) -> None:

    # Parse params from config
    min_steps = cfg_pipe.train_networks.min_steps
    max_steps = cfg_pipe.train_networks.max_steps
    milestone_steps = cfg_pipe.train_networks.milestone_steps
    val_check_interval = cfg_pipe.train_networks.val_check_interval

    # Update config
    cfg_lp.data.data_dir = data_dir
    if new_labels_csv is not None:
        cfg_lp.data.csv_file = new_labels_csv

    # TODO: can we get rid of this
    # Add iteration-specific fields to the config
    # cfg_lp.training.max_epochs = 10
    # cfg_lp.training.min_epochs = 10
    # cfg_lp.training.unfreeze_step = 30

    # Check if model has already been trained
    model_config_checked = os.path.join(results_dir, "config.yaml")

    if os.path.exists(model_config_checked) and not overwrite:
        print(
            f'config.yaml found for this model. Skipping training. ' 
            f'If you want to retrain this model, set overwrite=True '
            f'OR change cfg_pipeline.train_networks.intermediate_results_dir.'
        )

        checkpoint_pattern = os.path.join(
            results_dir, "tb_logs", "*", "version_*", "checkpoints", "best-checkpoint-*.ckpt")
        checkpoint_files = glob.glob(checkpoint_pattern)

        if checkpoint_files:
            best_ckpt = sorted(checkpoint_files)[-1]  # Get the latest best checkpoint

        data_module = None
        trainer = None

    else:
        print(f"No config.yaml found for this model. Training the model.")
        best_ckpt, data_module, trainer = train(
            cfg_lp=cfg_lp.copy(),
            results_dir=results_dir,
            min_steps=min_steps,
            max_steps=max_steps,
            milestone_steps=milestone_steps,
            val_check_interval=val_check_interval,
        )

    if inference_dirs is None:
        print(
            'inference_dirs None, not running inference. To run inference on videos in a given '
            'directory add it to cfg_pipeline.train_networks.inference_dirs'
        )
        return
    
    # Run inference on all InD/OOD videos and compute unsupervised metrics
    for video_dir in inference_dirs:
        
        video_files = [
            f for f in os.listdir(os.path.join(data_dir, video_dir)) if f.endswith('.mp4')
        ]

        if len(video_files) == 0:
            # assume we have a nested directory: directories full of mp4 files
            video_files = []
            sub_video_dirs = os.listdir(os.path.join(data_dir, video_dir))
            for sub_video_dir in sub_video_dirs:
                sub_video_dir_abs = os.path.join(data_dir, video_dir, sub_video_dir)
                files_tmp = os.listdir(sub_video_dir_abs)
                video_files += [f'{sub_video_dir}/{f}' for f in files_tmp if f.endswith('.mp4')]
        
        for video_file in video_files:
            if csv_prefix:
                raise NotImplementedError
                # inference_csv_name = f'{csv_prefix}_{video_file.replace(".mp4", ".csv")}'
            else:
                inference_csv_name = video_file.replace(".mp4", ".csv")
            inference_csv = os.path.join(results_dir, video_dir, inference_csv_name)

            if os.path.exists(inference_csv) and not overwrite:
                print(f"Inference file {inference_csv} already exists. "
                      f"Skipping inference for {video_file}")
            else:
                print(f"Running inference for {video_file}")
                inference_with_metrics(
                    video_file=os.path.join(data_dir, video_dir, video_file),
                    cfg_lp=cfg_lp.copy(),
                    preds_file=inference_csv,
                    ckpt_file=best_ckpt,
                    data_module=data_module,
                    trainer=trainer,
                    metrics=False,
                )
