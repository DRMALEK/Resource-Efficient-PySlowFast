# Resource-efficient Deep Learning Framework for Industrial Activity Recognition

This repository is part of a Master's thesis titled "Resource-efficient Deep Learning for Real-time Recognition of Worker Activities During Industrial Assembly". The project extends the PySlowFast framework with advanced capabilities for efficient deep learning in industrial settings.

## Table of Contents

- [About The Project](#about-the-project)
- [Framework Architecture](#framework-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Testing](#testing)
  - [Demo](#demo)
  - [Benchmarking](#benchmarking)
  - [Model Optimization](#model-optimization)
  - [Pruning](#pruning)
  - [Knowledge Distillation](#knowledge-distillation)
  - [End-to-End Pipeline](#end-to-end-pipeline)
- [Acknowledgements](#Acknowledgements)

## About The Project

This repository extends the original [PySlowFast](https://github.com/facebookresearch/SlowFast) framework with comprehensive additions focused on resource-efficient deep learning for industrial applications. Key enhancements include:

- Model compression through structured pruning
- Knowledge distillation capabilities
- Resource-efficient inference benchmarking 
- Integration with the MECCANO dataset for industrial assembly tasks

The framework is designed to strike a balance between high accuracy and computational efficiency, making it suitable for real-world industrial deployments.

## MECCANO Dataset Overview

The [MECCANO dataset](https://github.com/fpv-iplab/MECCANO) is the first egocentric video dataset focused on human-object interactions in industrial-like scenarios. It features:

- **Multimodal data**: RGB, depth, and gaze signals
- **Annotations**: Temporal (action segments) and spatial (object bounding boxes)
- **Tasks**: Action Recognition, Active Object Detection, Egocentric Human-Object Interaction (EHOI) Detection, and more
- **Classes**: 12 verbs, 20 objects, 61 unique actions
- **Acquisition**: 20 subjects, 2 countries, 1920x1080@12fps, over 8,800 video segments and 64,000+ bounding box annotations[^8][^11][^14]

## Framework Architecture

```
                                                     
┌─────────────────┐     ┌──────────────────┐
│  Input Video    │     │  Model Training   │
│    Stream       │────▶│   & Validation   │
└─────────────────┘     └──────────┬───────┘
                                   │
                         ┌─────────▼───────┐
                         │  Optimization    │
                         │    Pipeline     │
                         └─────────┬───────┘
                                  │
                    ┌─────────────┴──────────┐
                    │                        │
          ┌─────────▼─────────┐   ┌─────────▼─────────┐
          │     Pruning       │   │    Knowledge      │
          │    Framework      │   │   Distillation    │
          └─────────┬─────────┘   └─────────┬─────────┘
                    │                       │
                    └──────────┬────────────┘
                              │
                     ┌────────▼─────────┐
                     │   Deployment     │
                     │    Pipeline      │
                     └──────────────────┘

```
## Installation

### Requirements

- Python == 3.12
- GCC >= 4.9
- All dependencies listed in `requirements.txt`

Additionally, you must build **Detectron2** and **PyTorchVideo** from source.

### Install Dependencies

Install the core dependencies:

```bash
pip install -r requirements.txt
```

4. Build required components from source:

```bash
# Install Detectron2
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo

# Install PyTorchVideo
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .
```


---

## Usage

### Configuration

Example configuration file (`configs/MECCANO/SLOWFAST_8x8_R50.yaml`):
```yaml
MODEL:
  NUM_CLASSES: 61
  ARCH: slowfast
  MODEL_NAME: SlowFast
  LOSS_FUNC: cross_entropy
  DROPOUT_RATE: 0.5
SLOWFAST:
  ALPHA: 8
  BETA_INV: 8
  FUSION_CONV_CHANNEL_RATIO: 2
  FUSION_KERNEL_SZ: 7
TRAIN:
  BATCH_SIZE: 16
  EVAL_PERIOD: 10
  CHECKPOINT_PERIOD: 1
  AUTO_RESUME: True
DATA:
  NUM_FRAMES: 32
  SAMPLING_RATE: 2
  TRAIN_JITTER_SCALES: [256, 320]
  TRAIN_CROP_SIZE: 224
  TEST_CROP_SIZE: 256
  INPUT_CHANNEL_NUM: [3, 3]
SOLVER:
  BASE_LR: 0.01
  LR_POLICY: cosine
  MAX_EPOCH: 196
  MOMENTUM: 0.9
  WEIGHT_DECAY: 1e-4
  WARMUP_EPOCHS: 34.0
  WARMUP_START_LR: 0.01
  OPTIMIZING_METHOD: sgd
```

### Training

1. Set environment variables:
```bash
export PYTHONPATH=/path/to/framework:$PYTHONPATH
```

2. Start training:
```bash
python tools/run_net.py \
  --cfg configs/MECCANO/SLOWFAST_8x8_R50.yaml \
  NUM_GPUS 1 \
  TRAIN.BATCH_SIZE 8 \
  SOLVER.BASE_LR 0.01 \
  DATA.PATH_TO_DATA_DIR /path/to/meccano_dataset
```

### Testing

Evaluate a trained model:
```bash
python tools/run_net.py \
  --cfg configs/MECCANO/SLOWFAST_8x8_R50.yaml \
  --eval \
  TEST.CHECKPOINT_FILE_PATH /path/to/checkpoint.pyth
```

### Demo

Run inference on a video:
```bash
python tools/demo.py \
  --cfg configs/MECCANO/SLOWFAST_8x8_R50.yaml \
  --input_video path/to/video.mp4 \
  --checkpoint /path/to/checkpoint.pyth
```

### Benchmarking

Measure model performance:
```bash
python tools/benchmark.py \
  --cfg configs/MECCANO/SLOWFAST_8x8_R50.yaml \
  --model_path /path/to/checkpoint.pyth
```

### Model Optimization

#### Pruning
```bash
python tools/prune.py \
  --cfg configs/MECCANO/SLOWFAST_8x8_R50.yaml \
  --model_path /path/to/checkpoint.pyth \
  --prune_ratio 0.5 \
  --method l1_structured
```

#### Knowledge Distillation
```bash
python tools/distill.py \
  --teacher_cfg configs/MECCANO/SLOWFAST_8x8_R50.yaml \
  --student_cfg configs/MECCANO/MOBILE_8x8.yaml \
  --teacher_model /path/to/teacher.pyth \
  --temperature 4.0
```

### End-to-End Pipeline

Run the complete optimization pipeline:
```bash
bash scripts/run_pipeline.sh \
  --config configs/MECCANO/SLOWFAST_8x8_R50.yaml \
  --data_dir /path/to/meccano_dataset \
  --output_dir /path/to/output
```

## Acknowledgements

- [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)
- [fpv-iplab/MECCANO](https://github.com/fpv-iplab/MECCANO)

---
