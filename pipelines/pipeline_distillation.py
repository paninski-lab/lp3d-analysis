"""
Distillation Pipeline for Multi-View Pose Estimation

This pipeline creates pseudo-labeled datasets by intelligently selecting diverse frames 
from trained model predictions using variance-based filtering and 3D clustering.
"""

import argparse
import logging
import os
from typing import Dict

from lp3d_analysis.io import load_cfgs
from lp3d_analysis.distillation import DistillationConfig, run_distillation_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_distillation_config_from_pipeline_config(cfg_pipe, cfg_lp) -> DistillationConfig:
    """Create distillation configuration from pipeline and lightning-pose configs."""
    
    # Extract configuration from pipeline config
    distillation_config = cfg_pipe.get('distillation', {})
    dataset_name = cfg_pipe.get('dataset_name', distillation_config.get('dataset_name', 'fly-anipose'))
    view_names = cfg_lp['data']['view_names']
    
    # Process paths with dataset_name substitution
    def substitute_variables(path):
        if path and isinstance(path, str):
            path = path.replace('${dataset_name}', dataset_name)
            if '${train_frames}' in path:
                train_frames = distillation_config.get('train_frames', 200)
                path = path.replace('${train_frames}', str(train_frames))
            return path
        return path
    
    # Build configuration with defaults
    config_params = {
        # Dataset configuration
        'dataset_name': dataset_name,
        'base_data_dir': cfg_pipe.get('base_data_dir', "/teamspace/studios/data"),
        'pseudo_labeled_base_dir': cfg_pipe.get('pseudo_labeled_base_dir', "/teamspace/studios/this_studio/pseudo_labeled_dataset"),
        
        # Input paths (will be auto-constructed)
        'eks_results_dir': substitute_variables(distillation_config.get('eks_results_dir')),
        'camera_params_dir': substitute_variables(distillation_config.get('camera_params_dir')),
        'existing_data_dir': substitute_variables(distillation_config.get('existing_data_dir')),
        'video_base_dir': substitute_variables(distillation_config.get('video_base_dir')),
        
        # View and pipeline parameters
        'view_names': view_names,
        'frames_per_video': distillation_config.get('frames_per_video', 400),
        'n_clusters': distillation_config.get('n_clusters', 4000),
        'n_random_frames': distillation_config.get('n_random_frames', 4000),
        'train_frames': cfg_pipe.get('train_frames', distillation_config.get('train_frames', None)),
        'min_frames_per_video': distillation_config.get('min_frames_per_video', 60),
        'use_posterior_variance': distillation_config.get('use_posterior_variance', True),
        'use_random_selection': distillation_config.get('use_random_selection', False),
        
        # Data splitting (from lightning-pose config)
        'train_probability': cfg_lp.get('train_prob', distillation_config.get('train_probability', 0.95)),
        'val_probability': cfg_lp.get('val_prob', distillation_config.get('val_probability', 0.05)),
        'torch_seed': cfg_lp.get('rng_seed_data_pt', distillation_config.get('torch_seed', 0)),
        
        # Frame extraction and file handling 
        'extract_frames': distillation_config.get('extract_frames', False),
        'video_extensions': distillation_config.get('video_extensions', ['.mp4', '.avi', '.mov']),
        'copy_existing_data': distillation_config.get('copy_existing_data', True),
        'copy_calibrations': distillation_config.get('copy_calibrations', True)
    }
    
    return DistillationConfig(**config_params)

def _validate_paths(config: DistillationConfig) -> bool:
    """Validate that all required paths exist."""
    logger.info("Validating paths...")
    
    # Define path categories
    path_categories = {
        'required': [("EKS Results", config.eks_results_dir)],
        'optional': [("Camera Parameters", config.camera_params_dir)],
        'additional': []
    }
    
    # Add additional paths if specified
    if config.existing_data_dir:
        path_categories['additional'].append(("Existing Data", config.existing_data_dir))
    if config.video_base_dir:
        path_categories['additional'].append(("Video Base", config.video_base_dir))
    
    missing_required_paths = []
    
    # Check all path categories
    for category, paths in path_categories.items():
        for name, path in paths:
            if os.path.exists(path):
                logger.info(f"✅ {name}: {path}")
            else:
                if category == 'required':
                    logger.warning(f"⚠️  {name}: {path} (not found)")
                    missing_required_paths.append(path)
                elif category == 'optional':
                    logger.warning(f"⚠️  {name}: {path} (not found - will use PCA-based 3D reconstruction)")
                else:  # additional
                    logger.warning(f"⚠️  {name}: {path} (not found)")
    
    # Check output directory separately - it will be created if it doesn't exist
    if os.path.exists(config.output_dir):
        logger.info(f"✅ Output Directory: {config.output_dir}")
    else:
        logger.info(f"📁 Output Directory: {config.output_dir} (will be created)")
    
    if missing_required_paths:
        logger.error(f"Missing required paths: {missing_required_paths}")
        logger.info("Please update the paths in the configuration to match your setup.")
        return False
    
    return True

def _print_summary(results: Dict, config: DistillationConfig):
    """Print summary of distillation results."""
    logger.info(f"Distillation Summary:")
    logger.info(f"  Output directory: {config.output_dir}")
    if config.use_random_selection:
        logger.info(f"  Frame selection method: Random ({config.n_random_frames} frames)")
    else:
        logger.info(f"  Frame selection method: Variance-based + K-means ({config.n_clusters} clusters)")
    
    # Show reconstruction methods used
    if 'pseudo_data' in results and 'reconstruction_methods' in results['pseudo_data']:
        reconstruction_methods = results['pseudo_data']['reconstruction_methods']
        triangulation_sessions = [s for s, m in reconstruction_methods.items() if m == 'triangulation']
        pca_sessions = [s for s, m in reconstruction_methods.items() if m == 'pca']
        
        if triangulation_sessions:
            logger.info(f"  3D reconstruction method: Triangulation (sessions: {len(triangulation_sessions)})")
        if pca_sessions:
            logger.info(f"  3D reconstruction method: PCA-based (sessions: {len(pca_sessions)})")
    
    if 'outputs' in results and 'extraction_info' in results['outputs']:
        extraction_info = results['outputs']['extraction_info']
        existing_count = sum(1 for v in extraction_info.values() if v.get('data_type') == 'existing')
        new_csv_count = sum(1 for v in extraction_info.values() if v.get('data_type') == 'new_csv')
        pseudo_count = sum(1 for v in extraction_info.values() if v.get('data_type') == 'pseudo_labeled')
        
        logger.info(f"  Existing frames: {existing_count}")
        logger.info(f"  _new.csv frames: {new_csv_count}")
        logger.info(f"  Pseudo-labeled frames: {pseudo_count}")
        logger.info(f"  Total frames: {len(extraction_info)}")
    
    if 'extraction_results' in results and results['extraction_results']:
        extraction_results = results['extraction_results']
        logger.info(f"  Frame extraction: {extraction_results.get('successful_frames', 0)}/{extraction_results.get('total_frames', 0)} frames extracted")

def pipeline(config_file: str) -> None:
    """Run the distillation pipeline."""
    
    # Load configurations using the existing load_cfgs function
    cfg_pipe, cfg_lp = load_cfgs(config_file)
    
    # Check if distillation is enabled
    if not cfg_pipe.get('distillation', {}).get('run', False):
        logger.info("Distillation pipeline not enabled in config. Skipping.")
        return
    
    logger.info("Starting distillation pipeline...")
    
    # Create distillation configuration
    distillation_config = create_distillation_config_from_pipeline_config(cfg_pipe, cfg_lp)
    
    # Log the train/val probabilities being used
    logger.info(f"Using train/val probabilities from lightning-pose config: {distillation_config.train_probability:.1%} train, {distillation_config.val_probability:.1%} val")
    
    # Debug: Log the random selection configuration
    if distillation_config.use_random_selection:
        logger.info(f"Random selection enabled: n_random_frames = {distillation_config.n_random_frames}")
    else:
        logger.info(f"Variance-based selection enabled: n_clusters = {distillation_config.n_clusters}")
    
    # Validate paths
    if not _validate_paths(distillation_config):
        logger.error("Path validation failed. Please check the configuration.")
        return
    
    try:
        # Run distillation pipeline
        results = run_distillation_pipeline(distillation_config)
        
        logger.info("Distillation pipeline completed successfully!")
        
        # Print summary
        _print_summary(results, distillation_config)
        
    except Exception as e:
        logger.error(f"Distillation pipeline failed: {e}")
        raise

def main():
    """Main entry point for the distillation pipeline."""
    parser = argparse.ArgumentParser(description="Distillation Pipeline for Multi-View Pose Estimation")
    parser.add_argument("--config", dest="config_file", help="Path to pipeline configuration file")
    parser.add_argument("config_file_positional", nargs='?', help="Path to pipeline configuration file (positional)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Use --config if provided, otherwise use positional argument
    config_file = args.config_file or args.config_file_positional
    
    if not config_file:
        parser.error("Please provide a configuration file using --config or as a positional argument")
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))
    
    try:
        pipeline(config_file)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
