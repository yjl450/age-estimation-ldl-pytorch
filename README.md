# Age Estimation PyTorch with Label Distribution Learning
PyTorch-based CNN implementation for estimating age from face images.

2021-04-28 Update: Dependency `Pillow==7.2.0` is changed to the latest `Pillow` for security update. The compatibility of the updated package has not been tested.

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets
### Introduction

* Morph<sup>[4]</sup>

This dataset is widely used as a benchmark for age estimation algorithms. MAE (mean absolute error) is used as evaluation metric, which is also the default metric for validation of all datasets.

More infomation and the academic version of this dataset can be purchased [here](https://ebill.uncw.edu/C20231_ustores/web/store_main.jsp?STOREID=4).
* MegaAge and MegaAge-Asian <sup>[5]</sup>

Two large scale datasets featuring unconstrained images from the internet. CA (cummulative accuracy) is used as evaluation metric for this dataset. When training on these two datasets, any improvements in CA, as well as MAE, will result in a new saved model (.pth).

More information and the datasets can be acquired [here](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/).

## Data Preparation & Image Preprocessing
There should be two csv files generated for each dataset, one for training and one for validation, all of which follow the naming convention:
```bash
[dataset_name]_train_align.csv
[dataset_name]_valid_align.csv
```
In each file, 7 fields are required (order is not important):
* photo - the relative path to each image from the dataset folder
* age - the ground truth age
* deg - the angle to rotate the image so that the two eyes are on a horizontal line, computed from the facial landmarks
* box1, box2, box3, box4 - (x1, y1) (x2, y2) coordinates to crop the face from the photo

Two sample scripts are provided in [proprocessing](preprocessing) folder.

## Demo

See `python demo.py -h` for detailed options.

Webcam is required if no image directory is specified.
```bash
python demo.py
```

Using `--img_dir` argument, images in that directory will be used as input:

```bash
python demo.py --img_dir [PATH/TO/IMAGE_DIRECTORY]
```

Further using `--output_dir` argument,
images with predicted ages will be saved in that directory:

```bash
python demo.py --img_dir [PATH/TO/IMAGE_DIRECTORY] --output_dir [PATH/TO/OUTPUT_DIRECTORY]
```
By default, a pretrained model on MegaAge and MegaAge-Asian will be downloaded automatically. Otherwise, you can load a pretrained model using `--resume` argument.

Other arguments:
* `--expand` - a factor by which the cropping area (the face area for estimation) is expanded

## Train

### Train Model with Classification
See `python train.py -h` for detailed options.

```bash
python train.py --data_dir [PATH/TO/morph] --dataset morph
```

### Train Model with Label Distribution Learning (Recommended)


```bash
python train-ldl.py --data_dir [PATH/TO/morph] --dataset morph
```
It is strongly recommended to use data augmentation (`--aug`). Slight expansion of the face area (`--expand`) may also improve accuracy.

```bash
python train-ldl.py --data_dir [PATH/TO/morph] --dataset morph --aug --expand 0.2
```

### Train Multitasking Model with Label Distribution Learning (Morph Only)
```bash
python train-mt.py --data_dir [PATH/TO/morph] --dataset morph --aug --expand 0.2
```
Features after two layers of SE-ResNext bottleneck are fed into one gender predicator and one ethnicity predicator. The results of the two predicators, together with the features from the full network, are fed into the last fully connected layer to form the age distribution prediction.

In experiment, this model improves the MAE but requires the dataset to provide gender and ethnicity labels, which is why this model is only tested on Morph dataset.

### Other arguments:
* `--resume` - the path to a checkpoint
* `--checkpoint` - the folder to save the model
* `--multi_gpu` - use multiple gpu to accelerate training
* `--aug` - apply data augmentation using [albumentations](https://github.com/albumentations-team/albumentations), see [dataset.py](dataset.py) for detail
* `--expand` - same as [demo.py](demo.py), a factor by which the cropping area (the face area for estimation) is expanded

### Training Options
You can change training parameters including model architecture using additional arguments like this:

```bash
python train-ldl.py --data_dir [PATH/TO/dataset] MODEL.ARCH se_resnet50 TRAIN.OPT sgd TRAIN.LR 0.1
```

All default parameters defined in [defaults.py](defaults.py) can be changed using this style.

## Validation
Similar to the validation function used in [train.py](train.py). This script returns the results of MAE and CA of a specified saved model, and the MAEs of each age and gender group for detailed analysis. Furthermore, a csv file containing images whose ages are not correctly estimated is generated.

```bash
python val.py --data_dir [PATH/TO/morph] --dataset morph
```
### Other arguments
* `--ldl` - flag that indicates the result of the network is a distribution and the expectation should be calculated
* `--resume` - path to the saved model that is to be validated
* `--multi_gpu` - use multiple gpu to accelerate validation
* `--expand` - same as [train.py](train.py), the number should be the same as the factor used in training the model

## Result
To be updated

## References
[1] This project is heavily based on https://github.com/yu4u/age-estimation-pytorch

[2] The implementation of DLDL-v2 (ThinAgeNet, TinyAgeNet, Normal Sampling, KL Divergence + L1 Loss) is adopted from https://github.com/PuNeal/DLDL-v2-PyTorch

[3] Bin-Bin Gao, Hong-Yu Zhou, Jianxin Wu, and Xin Geng. 2018. Age Estimation Using Expectation of Label Distribution Learning. In <i>Proc. The 27th International Joint Conference on Artificial Intelligence (IJCAI 2018)</i>.

[4] K. Ricanek and T. Tesafaye, "MORPH: a longitudinal image database of normal adult age-progression," 7th International Conference on Automatic Face and Gesture Recognition (FGR06), Southampton, 2006, pp. 341-345, doi: 10.1109/FGR.2006.78.

[5] Yunxuan Zhang, Li Liu, Cheng Li, and Chen Change Loy. Quantifying Facial Age by Posterior of Age Comparisons, In British Machine Vision Conference (BMVC), 2017.

[6] Rasmus Rothe, Radu Timofte, and Luc Van Gool. 2018. Deep expectation of real and apparent age from a single image without facial landmarks. <i>International Journal of Computer Vision</i>, Vol. 126, No. 2-4, 144–157.