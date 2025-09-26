# lp3d-analysis

Analysis tools for multiview pose estimation models, including training, inference, and distillation pipelines.

## Installation

**Install `ffmpeg`**

First, check to see if you have `ffmpeg` installed by typing the following in the terminal:

```
ffmpeg -version
```

If not, install:

```
sudo apt install ffmpeg
```

**Set up a `conda` environment**

NOTE: do *not* do this if you are setting up this repo in a Lightning Studio.

```
conda create --yes --name labeler python=3.10
conda activate labeler
```

**Install dependencies**

Lightning Pose:
```
git clone https://github.com/danbider/lightning-pose.git
cd lightning-pose
pip install -e .
cd ..
```

Ensemble Kalman Smoother:
```
git clone https://github.com/paninski-lab/eks.git
cd eks
pip install -e .
cd ..
```

**Install `lp3d-analysis` package locally**

```
git clone https://github.com/paninski-lab/lp3d-analysis.git
cd lp3d-analysis
pip install -e .
cd ..
```

## Usage

### Training and Inference Pipeline

```
python pipelines/pipeline_simple.py --config <path_to_pipeline_config> 
```

### Pseudo-Label Pipeline

```
python pipelines/pipeline_pseudo_label.py --config <path_to_pipeline_config> 
```

### Distillation Pipeline

The distillation pipeline creates pseudo-labeled datasets by intelligently selecting diverse frames from trained model predictions using variance-based filtering and 3D clustering.

```
python pipelines/pipeline_distillation.py --config <path_to_pipeline_config> 
```

Or use the standalone script:

```
python scripts/run_distillation.py --config configs/pipeline_distillation_example.yaml
```

#### Distillation Pipeline Features

- **Variance-based filtering**: Selects frames with low prediction variance for high-confidence predictions
- **3D clustering**: Uses K-means clustering on 3D poses to select diverse representative frames
- **Existing data integration**: Combines with manually labeled data for comprehensive datasets
- **Frame extraction**: Optional automatic extraction of video frames
- **Multi-view support**: Handles multiple camera views with proper triangulation

#### Configuration

Create a configuration file with the distillation section:

```yaml
# Base paths
base_data_dir: "/teamspace/studios/data"
pseudo_labeled_base_dir: "/teamspace/studios/this_studio/pseudo_labeled_dataset"

distillation:
  run: true
  dataset_name: "fly-anipose"
  
  # EKS model info (for auto-constructing paths)
  eks_model_info:
    model_type: "mvt_3d_loss"
    n_hand_labels: "200"
    ensemble_seed: "0-2"
    eks_type: "eks_multiview_varinf"
  
  # Pipeline parameters
  frames_per_video: 400
  n_clusters: 4000
  extract_frames: false
  
  # File handling
  copy_existing_data: true
  copy_calibrations: true
```

The pipeline will automatically:
- Copy the entire `calibrations/` folder from the original dataset
- Copy all `_new.csv` files (CollectedData_{camera_name}_new.csv, calibrations_new.csv)
- Copy corresponding frames from the original dataset
- Generate pseudo-labeled data and combine it with existing data

See `configs/pipeline_distillation_example.yaml` for a complete example.
