# lp3d-analysis


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
conda create --yes --name labeler python=3.8
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
git clone https://github.com/paninski-lab/keypoint-pseudo-labeler.git
cd lp3d-analysis
pip install -e .
cd ..
```

## Use

```
python pipelines/run_pipeline.py --config <path_to_pipeline_config> 
```
