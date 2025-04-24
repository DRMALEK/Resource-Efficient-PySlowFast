# SlowFast for MECCANO

A fork of [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast) with custom additions to support the [MECCANO dataset](https://github.com/fpv-iplab/MECCANO): a multimodal egocentric video dataset for understanding human-object interactions in industrial-like settings.

---

## Table of Contents

- [About This Project](#about-this-project)
- [MECCANO Dataset Overview](#meccano-dataset-overview)
- [Installation](#installation)
    - [Requirements](#requirements)
    - [Install Dependencies](#install-dependencies)
    - [Build from Source](#build-from-source)
- [Usage](#usage)
    - [Before Running](#before-running)
    - [Training](#training)
- [Citing](#citing)
- [Acknowledgements](#acknowledgements)

---

## About This Project

This repository extends the original [SlowFast](https://github.com/facebookresearch/SlowFast) video understanding codebase to enable research on the MECCANO dataset. The modifications include:

- Custom dataloaders and configuration files for MECCANO
- Support for egocentric action recognition and human-object interaction tasks
- Integration with Detectron2 and PyTorchVideo for advanced video and object detection pipelines

---

## MECCANO Dataset Overview

The [MECCANO dataset](https://github.com/fpv-iplab/MECCANO) is the first egocentric video dataset focused on human-object interactions in industrial-like scenarios. It features:

- **Multimodal data**: RGB, depth, and gaze signals
- **Annotations**: Temporal (action segments) and spatial (object bounding boxes)
- **Tasks**: Action Recognition, Active Object Detection, Egocentric Human-Object Interaction (EHOI) Detection, and more
- **Classes**: 12 verbs, 20 objects, 61 unique actions
- **Acquisition**: 20 subjects, 2 countries, 1920x1080@12fps, over 8,800 video segments and 64,000+ bounding box annotations[^8][^11][^14]

---

## Installation

### Requirements

- Python == 3.12
- GCC >= 4.9
- All libraries listed in `requirements.txt`

Additionally, you must build **Detectron2** and **PyTorchVideo** from source.

### Install Dependencies

Install the core dependencies:

```bash
pip install -r requirements.txt
```


### Build from Source

#### Detectron2

```bash
git clone https://github.com/facebookresearch/detectron2 detectron2_repo
pip install -e detectron2_repo
```


#### PyTorchVideo

```bash
git clone https://github.com/facebookresearch/pytorchvideo.git
cd pytorchvideo
pip install -e .
```


---

## Usage

### Before Running

Add this directory to your `$PYTHONPATH`:

```bash
export PYTHONPATH=/path/to/SlowFast/slowfast:$PYTHONPATH
```


### Training

Run the training pipeline with:

```bash
python tools/run_net.py --cfg configs/Kinetics/C2D_8x8_R50.yaml NUM_GPUS 1 TRAIN.BATCH_SIZE 8 SOLVER.BASE_LR 0.0125 DATA.PATH_TO_DATA_DIR path_to_your_data_folder
```

Replace `path_to_your_data_folder` with the path to your MECCANO dataset.

---

## Acknowledgements

- [facebookresearch/SlowFast](https://github.com/facebookresearch/SlowFast)
- [fpv-iplab/MECCANO](https://github.com/fpv-iplab/MECCANO)

---
