# Train Guide for Mosquito 2023 by AI-PRIME

## 1. Dataset

### 1.1 Download dataset from following link

Object detection dataset:
```

```
Image classification dataset:
```

```
### 1.2 Place dataset to correct directory

Object detection dataset: `train/detection/dataset`
```
dataset
├── aegypti
├── albopictus
└── yolo_f0_raw
```
Image classification dataset: `train/classification/dataset`
```
dataset
├── aegypti
├── albopictus
└── yolo_f0_raw
```
## 2. Object Detection Model

### 2.1 Modify your dataset path
Modify the train/val path in `train/detection/data/coco_mos_extra_f0.yaml`, make sure the directory is correct


### 2.2 Train
 
```
cd train/detection
sh train.sh
```
get your best model under `train/detection/runs/train` directory

## 3. Image Classification Model

### 3.1 Train
Using following commands, you will get 10 models from 10 train/val splits
```
cd train/classification
sh batch_train.sh
```
### 3.2 Model Soup
using `avg_ckpt.py` to get uniform soup model from 10 models

## 4. Dataset Introduction

### Object Detection

Extra dataset(aegypti, albopictus) is collected in iNaturalist, and bbox of mosquito is labeled manually.

### Image Classification

In original dataset, images with noisy label are deleted.

Extra dataset is collected in Inaturalist, the class of each mosquito is double-checked manually.