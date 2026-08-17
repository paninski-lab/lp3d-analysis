import argparse
import os
import copy
import re

# Configure JAX memory settings BEFORE any imports to prevent segmentation faults
# These must be set before JAX is imported anywhere
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.5')  # Reduced to 30% for more headroom
xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_force_host_platform_device_count' not in xla_flags:
    os.environ['XLA_FLAGS'] = f'{xla_flags} --xla_force_host_platform_device_count=1'.strip()
# Disable JAX's aggressive compilation to prevent LLVM memory errors
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('XLA_PYTHON_CLIENT_ALLOCATOR', 'default')
# Disable JIT compilation to prevent LLVM memory allocation errors
# Set JAX_DISABLE_JIT=true to disable JIT (slower but prevents segfaults)
os.environ.setdefault('JAX_DISABLE_JIT', 'false')  # Disable JIT by default to prevent memory issues

from omegaconf import OmegaConf

from lp3d_analysis.io import load_cfgs
from lp3d_analysis.train import train_and_infer
from lp3d_analysis.utils import extract_ood_frame_predictions, rename_pixel_error_files
# from lp3d_analysis.post_process import  post_process_ensemble_videos , post_process_ensemble_labels #post_process_ensemble_labels,
from lp3d_analysis.post_process import post_process_ensemble_labels #post_process_ensemble_labels,
from lp3d_analysis.post_process_full_videos import  post_process_ensemble_videos, extract_labeled_frame_predictions
# from lp3d_analysis.post_process_concat import  post_process_ensemble_videos, post_process_ensemble_labels_concat
# from lp3d_analysis.post_process_concat_bbox import  post_process_ensemble_videos, post_process_ensemble_labels_concat
# from lp3d_analysis.post_process_concat_bbox_short_version_shorter import  post_process_ensemble_videos, post_process_ensemble_labels_concat
# from lp3d_analysis.post_process_no_concat import  post_process_ensemble_videos, post_process_ensemble_labels
# from lp3d_analysis.post_process_concat_bbox_aug26 import post_process_ensemble_labels_concat #post_process_ensemble_labels,
# from lp3d_analysis.post_process_full_videos import extract_labeled_frame_predictions #post_process_ensemble_labels,

# TODO
# - before train_and_infer, will be nice to put cfg updates in their own function
# - remove get_callbacks from lp3d_analysis.train.py 
# - replace train function with one from LP
# - faster inference with OOD videos: don't rebuild the model every time

VALID_MODEL_TYPES = [
    'supervised',
    'context',
    'multiview_transformer_learnable_crossview',
    'multiview_transformer',
    'multiview_transformer_mhcrnn',  # New multiview transformer MHCRNN model type
    'mvt_context',  # Multiview transformer with context (MHCRNN) - optimized version
    'mvt_3d_loss',
    'mvt_heatmap_3d_loss',
    'mvt_unsupervised_losses',
    'mvt_semisupervised',  # New model type for semi-supervised training with unsupervised losses
    'mvt_transformer_mhcrnn_semisupervised',  # New model type for semi-supervised multiview transformer MHCRNN

]


def convert_original_to_cropped_format(
    results_dir: str,
    animal_names: list[str],
    overwrite: bool = False,
) -> None:
    """Convert original-format prediction/pixel-error files to cropped format.

    Original format: 1 row per frame, 14 columns (7 per animal with animal prefix).
    Cropped format:  2 rows per frame (one per animal), 7 columns (no prefix).

    Operates in-place: reads each CSV, rewrites it in the cropped layout.
    A backup of the original file is kept with a ``_original_format`` suffix.
    """
    import pandas as pd

    files_to_convert = [
        f for f in os.listdir(results_dir)
        if f.startswith("predictions") and f.endswith(".csv")
    ]

    if not files_to_convert:
        print(f"No prediction files found in {results_dir}")
        return

    # Determine base keypoint names (strip animal prefix from first animal)
    # e.g. "black_mouse_nose" -> "nose"
    prefix = animal_names[0] + "_"

    for fname in files_to_convert:
        fpath = os.path.join(results_dir, fname)
        backup_path = os.path.join(results_dir, fname.replace(".csv", "_original_format.csv"))

        if os.path.exists(backup_path) and not overwrite:
            print(f"Skipping {fname} (already converted, backup exists)")
            continue

        is_pixel_error = "pixel_error" in fname or "pca_singleview_error" in fname

        if is_pixel_error:
            _convert_error_file(fpath, backup_path, animal_names)
        else:
            _convert_predictions_file(fpath, backup_path, animal_names)


def _convert_predictions_file(fpath, backup_path, animal_names):
    """Convert a predictions.csv (3-row multi-index header with x/y/likelihood)."""
    import pandas as pd
    import shutil

    df = pd.read_csv(fpath, header=[0, 1, 2], index_col=0)
    shutil.copyfile(fpath, backup_path)

    # Get the scorer name from first column
    scorer = df.columns.get_level_values(0)[0]

    # Extract base keypoint names from the first animal's columns
    first_animal = animal_names[0]
    prefix = first_animal + "_"
    base_kp_names = []
    seen = set()
    for bp in df.columns.get_level_values(1):
        if bp.startswith(prefix):
            base_name = bp[len(prefix):]
            if base_name not in seen:
                base_kp_names.append(base_name)
                seen.add(base_name)

    coords = ["x", "y", "likelihood"]
    new_columns = pd.MultiIndex.from_tuples(
        [(scorer, kp, c) for kp in base_kp_names for c in coords] + [(scorer, "", "set")],
        names=df.columns.names,
    )

    # Check if there's a "set" column
    has_set = any(c == "set" for c in df.columns.get_level_values(2))

    new_rows = []
    for img_path, row in df.iterrows():
        parts = img_path.split("/")
        session_dir = parts[-2] if len(parts) >= 2 else parts[0]
        img_filename = parts[-1] if len(parts) >= 2 else ""

        # Get the set value if present
        set_val = ""
        if has_set:
            set_cols = [c for c in df.columns if c[2] == "set"]
            if set_cols:
                set_val = row[set_cols[0]]

        for animal in animal_names:
            animal_prefix = animal + "_"
            # Build new path: session_animal/img
            new_path = "/".join(parts[:-2] + [f"{session_dir}_{animal}", img_filename]) if len(parts) >= 2 else img_path

            values = []
            for kp in base_kp_names:
                orig_kp = animal_prefix + kp
                for c in coords:
                    values.append(row[(scorer, orig_kp, c)])
            values.append(set_val)
            new_rows.append((new_path, values))

    new_df = pd.DataFrame(
        [v for _, v in new_rows],
        index=[p for p, _ in new_rows],
        columns=new_columns,
    )
    new_df.index.name = df.index.name
    new_df.to_csv(fpath)
    n_orig = len(df)
    n_new = len(new_df)
    print(f"Converted {os.path.basename(fpath)}: {n_orig} rows (14 kp) -> {n_new} rows (7 kp)")


def _convert_error_file(fpath, backup_path, animal_names):
    """Convert a pixel_error.csv (single header row with keypoint columns + set)."""
    import pandas as pd
    import shutil

    df = pd.read_csv(fpath, index_col=0)
    shutil.copyfile(fpath, backup_path)

    first_animal = animal_names[0]
    prefix = first_animal + "_"
    base_kp_names = []
    seen = set()
    for col in df.columns:
        if col.startswith(prefix):
            base_name = col[len(prefix):]
            if base_name not in seen:
                base_kp_names.append(base_name)
                seen.add(base_name)

    has_set = "set" in df.columns

    new_rows = []
    for img_path, row in df.iterrows():
        parts = img_path.split("/")
        session_dir = parts[-2] if len(parts) >= 2 else parts[0]
        img_filename = parts[-1] if len(parts) >= 2 else ""

        set_val = row["set"] if has_set else None

        for animal in animal_names:
            animal_prefix = animal + "_"
            new_path = "/".join(parts[:-2] + [f"{session_dir}_{animal}", img_filename]) if len(parts) >= 2 else img_path

            values = {}
            for kp in base_kp_names:
                values[kp] = row.get(animal_prefix + kp, float("nan"))
            if has_set:
                values["set"] = set_val
            new_rows.append((new_path, values))

    new_df = pd.DataFrame(
        [v for _, v in new_rows],
        index=[p for p, _ in new_rows],
    )
    new_df.index.name = df.index.name
    new_df.to_csv(fpath)
    n_orig = len(df)
    n_new = len(new_df)
    print(f"Converted {os.path.basename(fpath)}: {n_orig} rows (14 kp) -> {n_new} rows (7 kp)")


def create_paired_animal_csv(
    data_dir: str,
    csv_file: str,
    animal_names: list[str],
    n_hand_labels: int,
    rng_seed: int,
    results_dir: str,
) -> str:
    """Create a filtered CSV that contains properly paired frames across animals.

    For multi-animal single-view datasets (e.g. crim13 with black_mouse / white_mouse),
    frames must be selected in matched pairs so that every chosen time-point has an
    image for every animal.

    ``n_hand_labels`` is the **total** number of rows desired.  The function selects
    ``n_hand_labels // n_animals`` unique time-points and keeps all animal rows for
    each, giving exactly ``n_hand_labels`` rows (or the maximum available if the
    dataset is smaller).

    Returns the absolute path of the newly written CSV.
    """
    import pandas as pd
    import numpy as np

    n_animals = len(animal_names)
    if n_hand_labels % n_animals != 0:
        raise ValueError(
            f"n_hand_labels ({n_hand_labels}) must be divisible by the number of "
            f"animals ({n_animals}). Choose a multiple of {n_animals}."
        )

    n_timepoints = n_hand_labels // n_animals

    csv_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)

    # --- group rows by base frame (session without animal + image filename) ---
    # Image paths look like:
    #   labeled-data/030609_A25_Block11_BCfe1_t_black_mouse/img00010186.png
    groups = {}  # base_key -> {animal_name: row_index_in_df}
    for row_idx, img_path in enumerate(df.index):
        parts = img_path.split('/')
        if len(parts) < 2:
            continue
        session_dir = parts[-2]
        img_filename = parts[-1]

        # Identify which animal this row belongs to
        matched_animal = None
        base_session = session_dir
        for animal in animal_names:
            if animal in session_dir:
                matched_animal = animal
                # Strip the animal name to get the base session
                base_session = session_dir.replace(f'_{animal}', '').replace(f'{animal}_', '').replace(animal, '')
                # Clean up leftover separators
                base_session = re.sub(r'_+', '_', base_session).strip('_')
                break

        if matched_animal is None:
            print(f"Warning: could not match any animal in path '{img_path}', skipping.")
            continue

        base_key = f"{base_session}/{img_filename}"
        groups.setdefault(base_key, {})[matched_animal] = row_idx

    # Keep only fully paired groups (all animals present)
    paired_keys = [
        k for k, v in groups.items() if len(v) == n_animals
    ]
    print(
        f"Paired frame selection: {len(paired_keys)} fully paired time-points "
        f"found out of {len(groups)} total groups "
        f"({n_animals} animals: {animal_names})"
    )

    if len(paired_keys) == 0:
        raise RuntimeError(
            "No fully paired frames found. Check that animal_names in your config "
            "match the directory names in CollectedData.csv."
        )

    if n_timepoints > len(paired_keys):
        print(
            f"Warning: requested {n_timepoints} time-points but only "
            f"{len(paired_keys)} paired frames available. Using all."
        )
        n_timepoints = len(paired_keys)

    # Deterministic random selection
    rng = np.random.RandomState(rng_seed)
    selected_keys = sorted(rng.choice(paired_keys, size=n_timepoints, replace=False))

    # Collect all row indices for selected paired frames
    selected_indices = []
    for key in selected_keys:
        for animal in animal_names:
            selected_indices.append(groups[key][animal])
    selected_indices.sort()

    # Write filtered CSV
    filtered_df = df.iloc[selected_indices]
    os.makedirs(results_dir, exist_ok=True)
    out_filename = f"CollectedData_paired_{n_hand_labels}_seed{rng_seed}.csv"
    out_path = os.path.join(results_dir, out_filename)
    filtered_df.to_csv(out_path)

    print(
        f"Created paired CSV: {out_path} "
        f"({n_timepoints} time-points × {n_animals} animals = {len(selected_indices)} rows)"
    )

    # Also create the _new (OOD) paired CSV if the source _new file exists.
    # Lightning Pose looks for <stem>_new.csv for OOD evaluation; without this
    # it silently skips OOD predictions.
    csv_stem = os.path.splitext(csv_file)[0]
    csv_new_path = os.path.join(data_dir, f"{csv_stem}_new.csv")
    if os.path.isfile(csv_new_path):
        df_new = pd.read_csv(csv_new_path, header=[0, 1, 2], index_col=0)
        groups_new = {}
        for row_idx, img_path in enumerate(df_new.index):
            parts = img_path.split('/')
            if len(parts) < 2:
                continue
            session_dir = parts[-2]
            img_filename = parts[-1]
            matched_animal = None
            base_session = session_dir
            for animal in animal_names:
                if animal in session_dir:
                    matched_animal = animal
                    base_session = session_dir.replace(f'_{animal}', '').replace(f'{animal}_', '').replace(animal, '')
                    base_session = re.sub(r'_+', '_', base_session).strip('_')
                    break
            if matched_animal is None:
                continue
            base_key = f"{base_session}/{img_filename}"
            groups_new.setdefault(base_key, {})[matched_animal] = row_idx

        # Keep all fully paired OOD frames (no subsampling — we want full OOD eval)
        paired_new_indices = []
        for key, animals_dict in groups_new.items():
            if len(animals_dict) == n_animals:
                for animal in animal_names:
                    paired_new_indices.append(animals_dict[animal])
        paired_new_indices.sort()

        if paired_new_indices:
            filtered_new_df = df_new.iloc[paired_new_indices]
            out_stem = os.path.splitext(out_filename)[0]
            out_new_path = os.path.join(results_dir, f"{out_stem}_new.csv")
            filtered_new_df.to_csv(out_new_path)
            n_new_timepoints = len(paired_new_indices) // n_animals
            print(
                f"Created paired OOD CSV: {out_new_path} "
                f"({n_new_timepoints} time-points × {n_animals} animals = {len(paired_new_indices)} rows)"
            )

    return out_path


def _model_losses_to_use_from_cfg(cfg_lp, default_if_empty: list | None = None) -> list:
    """Resolve ``model.losses_to_use`` when applying pipeline overrides.

    If the base Lightning Pose config lists unsupervised losses, that list is used
    (semi-supervised: ``UnlabeledDataModule`` + ``SemiSupervisedHeatmapTracker*``).
    If the list is empty or missing, ``default_if_empty`` is used — e.g. ``[]`` for
    supervised-only, or ``[\"pca_multiview\"]`` for pipeline types that default to PCA.
    """
    raw = cfg_lp.model.get("losses_to_use", None)
    if raw is None:
        return list(default_if_empty) if default_if_empty is not None else []
    raw_list = [x for x in list(raw) if x is not None and str(x).strip() != ""]
    if len(raw_list) == 0:
        return list(default_if_empty) if default_if_empty is not None else []
    return raw_list


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

                    # For multi-animal datasets, create a paired-frame CSV so that
                    # every selected time-point includes all animals.
                    animal_names = cfg_lp.data.get("animal_names", None)
                    if animal_names is not None and len(animal_names) > 1:
                        paired_csv_dir = os.path.join(outputs_dir, 'paired_csvs')
                        paired_csv_path = create_paired_animal_csv(
                            data_dir=data_dir,
                            csv_file=cfg_lp.data.csv_file,
                            animal_names=list(animal_names),
                            n_hand_labels=n_hand_labels,
                            rng_seed=rng_seed,
                            results_dir=paired_csv_dir,
                        )
                        # Point to paired CSV. The CSV already contains exactly the
                        # right paired rows, so set train_frames=1 (meaning "use all
                        # available training frames") and let train_prob/val_prob
                        # handle the train-vs-validation split within those rows.
                        cfg_lp_copy = OmegaConf.merge(cfg_lp_copy, {
                            "data": {"csv_file": paired_csv_path},
                            "training": {"train_frames": 1},
                        })

                    # Main function call
                    train_and_infer(
                        cfg_lp=cfg_lp_copy,
                        model_dir=results_dir,
                        inference_dirs=cfg_pipe.train_networks.inference_dirs,
                        overwrite=cfg_pipe.train_networks.overwrite,
                    )
                    print(f"Debug: Rename pixel error files for {results_dir}")
                    # Rename pixel error files for this specific model
                    rename_pixel_error_files(results_dir)

                    if 'videos-for-each-labeled-frame' in cfg_pipe.train_networks.inference_dirs:
                        # Clean up/reorganize OOD data
                        extract_ood_frame_predictions(
                            cfg_lp=cfg_lp_copy,
                            data_dir=data_dir,
                            results_dir=results_dir,
                            overwrite=cfg_pipe.train_networks.overwrite,
                            # overwrite=True,
                            video_dir='videos-for-each-labeled-frame',
                        )

    # The rest of the pipeline only runs when you run without --for_seed.
    if for_seed is not None:
        return

    for mode, mode_config in cfg_pipe.post_processing_videos.items():
        for model_type in cfg_pipe.train_networks.model_types:
            for n_hand_labels in cfg_pipe.train_networks.n_hand_labels:
                if mode_config.run: # if the mode is mean or median or eks_singleview 
                    print(f"Debug: Preparing to run {mode} for {model_type} with seed range {cfg_pipe.train_networks.ensemble_seeds}")
                    # cfg_lp_copy = make_model_cfg(cfg_lp, cfg_pipe, data_dir, model_type, n_hand_labels, rng_seed)
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
                        **({"n_latent": mode_config.n_latent} if hasattr(mode_config, 'n_latent') else {}),
                        **({"non_linear": mode_config.non_linear} if hasattr(mode_config, 'non_linear') else {"non_linear": False}),
                        **({"output_folder_name": mode_config.output_folder_name} if hasattr(mode_config, 'output_folder_name') and mode_config.output_folder_name else {})
                    )
    
    # Second part - make labeled frames extraction completely independent
    if hasattr(cfg_pipe, "post_processing_labeled_frames") and cfg_pipe.post_processing_labeled_frames.run:
        print("\n----- Processing labeled frames for ensemble methods -----")
        for model_type in cfg_pipe.train_networks.model_types:
            for n_hand_labels in cfg_pipe.train_networks.n_hand_labels:
                for mode in cfg_pipe.post_processing_labeled_frames.modes:
                    print(f"\nExtracting labeled frame predictions for {model_type} {n_hand_labels} using {mode}")
                    # specify output directory for the ensemble
                    results_dir = os.path.join(
                        outputs_dir, cfg_pipe.intermediate_results_dir, 
                        f'{model_type}_{n_hand_labels}_{cfg_pipe.train_networks.ensemble_seeds[0]}',
                    )
                    cfg_lp_copy = make_model_cfg(cfg_lp, cfg_pipe, data_dir, model_type, n_hand_labels, cfg_pipe.train_networks.ensemble_seeds[0])
                    
                    # Run labeled frames extraction for each inference directory
                    for inference_dir in cfg_pipe.train_networks.inference_dirs:
                        # I want to run only if there are files in the directory and it is not empty
                        print(f"Checking if inference directory {inference_dir} exists and is not empty")
                        if os.path.exists(os.path.join(results_dir, inference_dir)) and os.listdir(os.path.join(results_dir, inference_dir)):
                            extract_labeled_frame_predictions(
                                cfg_lp=cfg_lp_copy,
                                results_dir=results_dir,
                                model_type=model_type,
                                n_labels=n_hand_labels,
                                seed_range=(cfg_pipe.train_networks.ensemble_seeds[0], cfg_pipe.train_networks.ensemble_seeds[-1]),
                                views=list(cfg_lp.data.view_names),
                                mode=mode,
                                inference_dir=inference_dir,
                                overwrite=cfg_pipe.post_processing_labeled_frames.overwrite
                            )
                        else:
                            print(f"Skipping {inference_dir} as it does not exist or is empty so we cannot extract labeled frames")

    for mode, mode_config in cfg_pipe.post_processing_labels.items():
        for model_type in cfg_pipe.train_networks.model_types:
            for n_hand_labels in cfg_pipe.train_networks.n_hand_labels:
                if mode_config.run: # if the mode is mean or median or eks_singleview 
                    #print(f"Debug: Preparing to run {mode} for {model_type} with seed range {cfg_pipe.train_networks.ensemble_seeds}"
                    # cfg_lp_copy = make_model_cfg(cfg_lp, cfg_pipe, data_dir, model_type, n_hand_labels, 0)
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
                        **({"n_latent": mode_config.n_latent} if hasattr(mode_config, 'n_latent') else {}),
                        **({"non_linear": mode_config.non_linear} if hasattr(mode_config, 'non_linear') else {"non_linear": False}),
                        **({"output_folder_name": mode_config.output_folder_name} if hasattr(mode_config, 'output_folder_name') and mode_config.output_folder_name else {})
                    )

    # # Add processing of labeled frames after post-processing videos
    # if hasattr(cfg_pipe, "post_processing_labeled_frames") and cfg_pipe.post_processing_labeled_frames.run:
    #     print("\n----- Processing labeled frames for ensemble methods -----")
    #     for model_type in cfg_pipe.train_networks.model_types:
    #         for n_hand_labels in cfg_pipe.train_networks.n_hand_labels:
    #             for mode in cfg_pipe.post_processing_labeled_frames.modes:
    #                 if mode not in cfg_pipe.post_processing_videos or not cfg_pipe.post_processing_videos[mode].run:
    #                     print(f"Skipping {mode} for labeled frames as it was not run in post_processing_videos")
    #                     continue
                        
    #                 print(f"\nExtracting labeled frame predictions for {model_type} {n_hand_labels} using {mode}")
    #                 # specify output directory for the ensemble
    #                 results_dir = os.path.join(
    #                     outputs_dir, cfg_pipe.intermediate_results_dir, 
    #                     f'{model_type}_{n_hand_labels}_{cfg_pipe.train_networks.ensemble_seeds[0]}',
    #                 )
    #                 cfg_lp_copy = make_model_cfg(cfg_lp, cfg_pipe, data_dir, model_type, n_hand_labels, cfg_pipe.train_networks.ensemble_seeds[0])
                    
    #                 # Run labeled frames extraction for each inference directory
    #                 for inference_dir in cfg_pipe.train_networks.inference_dirs:
    #                     extract_labeled_frame_predictions(
    #                         cfg_lp=cfg_lp_copy,
    #                         results_dir=results_dir,
    #                         model_type=model_type,
    #                         n_labels=n_hand_labels,
    #                         seed_range=(cfg_pipe.train_networks.ensemble_seeds[0], cfg_pipe.train_networks.ensemble_seeds[-1]),
    #                         views=list(cfg_lp.data.view_names),
    #                         mode=mode,
    #                         inference_dir=inference_dir,
    #                         overwrite=cfg_pipe.post_processing_labeled_frames.overwrite
    #                     )





    # for mode, mode_config in cfg_pipe.post_processing_labels.items():
    #     for model_type in cfg_pipe.train_networks.model_types:
    #         for n_hand_labels in cfg_pipe.train_networks.n_hand_labels:
    #             if mode_config.run: # if the mode is mean or median or eks_singleview 
    #                 #print(f"Debug: Preparing to run {mode} for {model_type} with seed range {cfg_pipe.train_networks.ensemble_seeds}"
    #                 # cfg_lp_copy = make_model_cfg(cfg_lp, cfg_pipe, data_dir, model_type, n_hand_labels, 0)
    #                 post_process_ensemble_labels( # remember I changed that for a second 
    #                     cfg_lp=cfg_lp_copy.copy(),
    #                     results_dir=results_dir,
    #                     model_type=model_type,
    #                     n_labels= n_hand_labels,
    #                     seed_range=(cfg_pipe.train_networks.ensemble_seeds[0], cfg_pipe.train_networks.ensemble_seeds[-1]),
    #                     views= list(cfg_lp.data.view_names), # before it was not a list... 
    #                     mode=mode,
    #                     inference_dirs=cfg_pipe.train_networks.inference_dirs,
    #                     overwrite=mode_config.overwrite,
    #                 )

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
            # Control 3D augmentations for multiview models:
            "imgaug_3d": cfg_lp.training.get("imgaug_3d", None),  # Read from config file
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
    elif model_type == 'multiview_cnn':
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview",
                "losses_to_use": [],
                "head": "heatmap_cnn"
            },
            
        })
    
    elif model_type == 'multiview_transformer_learnable_crossview':
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview",
                "losses_to_use": [],
                "head": "feature_transformer_learnable_crossview"
            },
            
        })
    elif model_type == 'multiview_transformer':
        # Honor ``losses_to_use`` from the LP yaml (e.g. ``[pca_multiview]`` for chickadee).
        # Empty list in the base config keeps fully supervised training.
        losses_to_use = _model_losses_to_use_from_cfg(cfg_lp, default_if_empty=[])
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview_transformer",
                "losses_to_use": losses_to_use,
                "head": "heatmap_cnn"
            },
            
        })

    elif model_type == 'mvt_3d_loss':
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview_transformer",
                "losses_to_use": [],
                "head": "heatmap_cnn"
            },
            
        })
    
    elif model_type == 'mvt_heatmap_3d_loss':
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview_transformer",
                "losses_to_use": [],
                "head": "heatmap_cnn"
            },
        })
        
    elif model_type == 'mvt_unsupervised_losses':
        losses_to_use = _model_losses_to_use_from_cfg(
            cfg_lp, default_if_empty=["pca_multiview"]
        )
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview_transformer",
                "losses_to_use": losses_to_use,
                "head": "heatmap_cnn"
            },
        })

    elif model_type == 'mvt_semisupervised':
        losses_to_use = _model_losses_to_use_from_cfg(
            cfg_lp, default_if_empty=["pca_multiview"]
        )
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview_transformer",
                "losses_to_use": losses_to_use,
                "head": "heatmap_cnn"  # Use heatmap_cnn head
            },
            "data": {
                "downsample_factor": 2,  # Set to 2 for heatmap_cnn head
            },
        })
    elif model_type == 'multiview_transformer_mhcrnn':
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview_transformer_mhcrnn",
                "losses_to_use": [],
                "head": "heatmap_mhcrnn"
            },
            "data": {
                "downsample_factor": 2,  # Set to 2 for heatmap_mhcrnn head
            },
        })
    elif model_type == 'mvt_context':
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview_transformer_mhcrnn",
                "losses_to_use": [],
                "head": "heatmap_mhcrnn_multiview"  # MHCRNN head for multiview with context
            },
            "data": {
                "downsample_factor": 2,  # Set to 2 for heatmap_mhcrnn_multiview head
            },
        })
    elif model_type == 'mvt_transformer_mhcrnn_semisupervised':
        losses_to_use = _model_losses_to_use_from_cfg(
            cfg_lp, default_if_empty=["pca_multiview"]
        )
        cfg_overrides.append({
            "model": {
                "model_type": "heatmap_multiview_transformer_mhcrnn",
                "losses_to_use": losses_to_use,
                "head": "heatmap_mhcrnn"
            },
            "data": {
                "downsample_factor": 2,  # Set to 2 for heatmap_mhcrnn head
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
    
    # Parse curriculum learning parameters from training config
    patch_mask_config = cfg_lp.training.get("patch_mask", {})
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
            "patch_mask": patch_mask_config,  # Pass the entire patch_mask config
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
