"""
Distillation Pipeline for cheese-3d — InD + OOD in one run.

Creates a new multiview pseudo-labeled dataset WITHOUT relying on existing
ground-truth labels (cheese-3d only has a single CollectedData.csv, not per-
view CollectedData_BC.csv files).

Two passes share the same output directory:
  InD  (videos/)      → CollectedData_{view}.csv  +  calibrations.csv
  OOD  (videos_new/)  → CollectedData_{view}_new.csv  +  calibrations_new.csv

Uses the cheese-3d-specific pipeline (distillation_cheese3d) which:
  - Extracts ensemble median for triangulation
  - Filters by per-keypoint likelihood across views
  - Generates labels via per-keypoint triangulation + reprojection
"""

import argparse
import logging
import os
from typing import Dict, Optional

from lp3d_analysis.io import load_cfgs
from lp3d_analysis.distillation_cheese3d import (
    Cheese3dDistillationConfig,
    run_cheese3d_distillation_pipeline,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Config construction
# ─────────────────────────────────────────────────────────────────────────────

def _build_config(
    cfg_pipe: dict,
    cfg_lp: dict,
    eks_results_dir: str,
    video_base_dir: str,
    n_clusters: int,
    frames_per_video: int,
    csv_file_suffix: str,
) -> Cheese3dDistillationConfig:
    """Build a Cheese3dDistillationConfig for one pass (InD or OOD)."""

    dist = cfg_pipe.get('distillation', {})
    shared = {k: v for k, v in dist.items() if k not in ('ind', 'ood', 'run')}

    dataset_name = cfg_pipe.get('dataset_name', 'cheese-3d')
    view_names   = cfg_lp['data']['view_names']
    base_data_dir = cfg_pipe.get('base_data_dir', '/teamspace/studios/data')
    pseudo_labeled_base_dir = cfg_pipe.get(
        'pseudo_labeled_base_dir',
        '/teamspace/studios/this_studio/pseudo_labeled_dataset'
    )

    return Cheese3dDistillationConfig(
        dataset_name=dataset_name,
        base_data_dir=base_data_dir,
        pseudo_labeled_base_dir=pseudo_labeled_base_dir,

        eks_results_dir=eks_results_dir,
        camera_params_dir=shared.get('camera_params_dir'),
        # "" (not None) prevents auto-construction and skips existing-data logic
        existing_data_dir="",
        video_base_dir=video_base_dir,

        view_names=view_names,
        frames_per_video=frames_per_video,
        n_clusters=n_clusters,
        n_random_frames=n_clusters,
        train_frames=cfg_pipe.get('train_frames', None),
        min_frames_per_video=shared.get('min_frames_per_video', 0),
        use_posterior_variance=shared.get('use_posterior_variance', True),
        use_random_selection=shared.get('use_random_selection', False),

        train_probability=cfg_lp.get('train_prob', 0.95),
        val_probability=cfg_lp.get('val_prob', 0.05),
        torch_seed=cfg_lp.get('rng_seed_data_pt', 0),

        extract_frames=shared.get('extract_frames', True),
        video_extensions=shared.get('video_extensions', ['.mp4', '.avi', '.mov']),
        copy_existing_data=False,
        copy_calibrations=shared.get('copy_calibrations', True),

        # Cheese-3d specific
        csv_file_suffix=csv_file_suffix,
        likelihood_threshold=shared.get('likelihood_threshold', 0.5),
        min_confident_views=shared.get('min_confident_views', 2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Validation & summary
# ─────────────────────────────────────────────────────────────────────────────

def _validate_paths(config, label: str) -> bool:
    checks = [
        ('required', 'EKS Results',      config.eks_results_dir),
        ('optional', 'Camera Parameters', config.camera_params_dir),
        ('optional', 'Video Base',        config.video_base_dir),
    ]
    missing = []
    for category, name, path in checks:
        if not path:
            if category == 'required':
                logger.error(f"  [{label}] {name}: not configured")
                missing.append(name)
            continue
        if os.path.exists(path):
            logger.info(f"  [{label}] ✅ {name}: {path}")
        else:
            if category == 'required':
                logger.warning(f"  [{label}] ⚠️  {name}: {path} (not found)")
                missing.append(path)
            else:
                logger.warning(f"  [{label}] ⚠️  {name}: {path} (not found)")

    if os.path.exists(config.output_dir):
        logger.info(f"  [{label}] ✅ Output: {config.output_dir}")
    else:
        logger.info(f"  [{label}] 📁 Output: {config.output_dir} (will be created)")

    if missing:
        logger.error(f"  [{label}] Missing required paths: {missing}")
    return not missing


def _print_summary(results: Dict, config, label: str):
    suffix = config.csv_file_suffix or "(none)"
    logger.info(f"[{label}] Summary:")
    logger.info(f"  Output dir:     {config.output_dir}")
    logger.info(f"  CSV suffix:     {suffix}")
    if config.use_random_selection:
        logger.info(f"  Frame selection: Random ({config.n_random_frames} frames)")
    else:
        logger.info(f"  Frame selection: Variance + K-means ({config.n_clusters} clusters)")

    if 'pseudo_data' in results and 'reconstruction_methods' in results['pseudo_data']:
        methods = results['pseudo_data']['reconstruction_methods']
        tri = sum(1 for m in methods.values() if m == 'triangulation')
        pca = sum(1 for m in methods.values() if m == 'pca')
        if tri:
            logger.info(f"  3D: Triangulation ({tri} sessions)")
        if pca:
            logger.info(f"  3D: PCA-based ({pca} sessions)")

    if 'outputs' in results and 'extraction_info' in results['outputs']:
        info = results['outputs']['extraction_info']
        pseudo = sum(1 for v in info.values() if v.get('data_type') == 'pseudo_labeled')
        logger.info(f"  Pseudo-labeled frames: {pseudo} / {len(info)} total")

    if results.get('extraction_results'):
        er = results['extraction_results']
        logger.info(f"  Frames extracted: {er.get('successful_frames', 0)}/{er.get('total_frames', 0)}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def _run_pass(label: str, config: Cheese3dDistillationConfig) -> Optional[Dict]:
    """Run one cheese-3d distillation pass."""
    logger.info(f"{'─'*60}")
    logger.info(f"Starting {label} pass  (suffix='{config.csv_file_suffix}')")
    logger.info(f"  EKS:    {config.eks_results_dir}")
    logger.info(f"  Videos: {config.video_base_dir}")
    logger.info(f"  Target: {config.n_clusters} frames")

    if not _validate_paths(config, label):
        logger.error(f"{label} path validation failed — skipping this pass")
        return None

    try:
        results = run_cheese3d_distillation_pipeline(config)
        logger.info(f"{label} pass completed successfully!")
        _print_summary(results, config, label)
        return results
    except Exception as e:
        logger.error(f"{label} pass failed: {e}")
        raise


def pipeline(config_file: str) -> None:
    cfg_pipe, cfg_lp = load_cfgs(config_file)

    if not cfg_pipe.get('distillation', {}).get('run', False):
        logger.info("Distillation not enabled in config. Skipping.")
        return

    dist = cfg_pipe['distillation']
    ind_cfg_yaml = dist.get('ind', {})
    ood_cfg_yaml = dist.get('ood', {})

    shared_n_clusters = dist.get('n_clusters', 200)
    shared_fpv       = dist.get('frames_per_video', 30000)

    # ── InD pass ──────────────────────────────────────────────────────────────
    ind_config = _build_config(
        cfg_pipe=cfg_pipe,
        cfg_lp=cfg_lp,
        eks_results_dir=ind_cfg_yaml.get('eks_results_dir'),
        video_base_dir=ind_cfg_yaml.get('video_base_dir'),
        n_clusters=ind_cfg_yaml.get('n_clusters', shared_n_clusters),
        frames_per_video=ind_cfg_yaml.get('frames_per_video', shared_fpv),
        csv_file_suffix="",
    )
    _run_pass("InD", ind_config)

    # ── OOD pass ──────────────────────────────────────────────────────────────
    ood_config = _build_config(
        cfg_pipe=cfg_pipe,
        cfg_lp=cfg_lp,
        eks_results_dir=ood_cfg_yaml.get('eks_results_dir'),
        video_base_dir=ood_cfg_yaml.get('video_base_dir'),
        n_clusters=ood_cfg_yaml.get('n_clusters', shared_n_clusters),
        frames_per_video=ood_cfg_yaml.get('frames_per_video', shared_fpv),
        csv_file_suffix="_new",
    )
    ood_config.copy_calibrations = False
    _run_pass("OOD", ood_config)

    logger.info("cheese-3d distillation pipeline complete.")
    logger.info(f"Output: {ind_config.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Distillation Pipeline for cheese-3d (InD + OOD)"
    )
    parser.add_argument("--config", dest="config_file")
    parser.add_argument("config_file_positional", nargs='?')
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    config_file = args.config_file or args.config_file_positional
    if not config_file:
        parser.error("Please provide a config file via --config or as a positional argument")

    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    try:
        pipeline(config_file)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
