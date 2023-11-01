# Train Guide for Mosquito 2023 by AI-PRIME

## 1. Dataset

### 1.1 Download dataset from following link

Object detection dataset:
```
https://www.kaggle.com/datasets/eashenyang/mosquito-alert-2023-detection
```
Image classification dataset:
```
https://www.kaggle.com/datasets/eashenyang/mosquito-alert-2023-classification
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
├── origin_dataset
└── inaturalist
```
## 2. Object Detection Model

### 2.1 Train
 
```
cd train/detection
sh train.sh
```
get your best model under `train/detection/runs/train` directory

## 3. Image Classification Model

### 3.1 Pretrain Model Prepare

Download pretrain model from following link

```
https://drive.google.com/file/d/17sUNST7ivQhonBAfZEiTOLAgtaHa4F3e/view?usp=sharing
```
Place `metafg_2_inat21_384.pth` at `train/classification`

### 3.2 Train
Using following commands, you will get 10 models from 10 train/val splits
```
cd train/classification
sh batch_train.sh
```
### 3.3 Model Soup
Using following commands, you will get uniform soup model from 10 models
```
cd train/classification
python avg_ckpt.py {your_args}
```


## 4. Dataset Introduction

### Object Detection

The original dataset underwent the removal of images containing multiple mosquitoes, and bounding boxes were manually refined. Additionally, an additional dataset was compiled from iNaturalist, featuring Aedes aegypti and Aedes albopictus species, with mosquito bounding boxes also being manually labeled.

### Image Classification

In the original dataset, images with noisy labels were eliminated. An additional dataset was acquired from iNaturalist, and the mosquito class for each entry was meticulously verified through manual inspection.