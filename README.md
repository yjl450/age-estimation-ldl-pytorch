# Age Estimation PyTorch with Label Distribution Learning
PyTorch-based CNN implementation for estimating age from face images.

<img src="misc/example.png" width="800px">

## Requirements

```bash
pip install -r requirements.txt
```

## Datasets
### Introduction

* Morph<sup>[CITE]</sup>

This dataset is widely used as a benchmark for age estimation algorithms. MAE (mean absolute error) is used as evaluation metric, which is also the default metric for validation of all datasets.

More infomation and the academic version of this dataset can be purchased [here](https://ebill.uncw.edu/C20231_ustores/web/store_main.jsp?STOREID=4).
* MegaAge and MegaAge-Asian <sup>[CITE]</sup>

Two large scale datasets featuring unconstrained images from the internet. CA (cummulative accuracy) is used as evaluation metric for this dataset. When training on these two datasets, any improvements in CA, as well as MAE, will result in a new saved model (.pth).

More information and the datasets can be acquired [here](http://mmlab.ie.cuhk.edu.hk/projects/MegaAge/).

* IMDB-WIKI <sup>[CITE]</sup>

A huge dataset that is commonly used in pretraining due to its relatively low quality of the images included. Its MAE is not usually used as a evaluation metric of the model. More information [here](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/).

## Preprocessing
There should be two csv files generated for each dataset, one for training and one for validation, all of which follow the naming convention:
```bash
[dataset_name]_train_align.csv
[dataset_name]_valid_align.csv
```
In each file, 7 fields are required (order is not important):
* photo - the relative path to each image from the dataset folder
* age - the ground truth age
* deg - the angle to rotate the image so that the two eyes are on a horizontal line, computed from the facial landmarks
* box1, box2, box3, box4, (x1, y1) (x2, y2) coordinates to crop the face from the photo
  
We use MTCNN<sup>[CITE]</sup> for face detection and alignment. Each image should go through two passes. In the first pass, five points of the facial landmarks are detected and the rotation angle is computed. In the second pass, the rotated image is used to find the bounding box of the face. The rotation angle and the bounding box are saved in the csv files.

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
resulting images will be saved in that directory (no resulting image window is displayed in this case):

```bash
python demo.py --img_dir [PATH/TO/IMAGE_DIRECTORY] --output_dir [PATH/TO/OUTPUT_DIRECTORY]
```
By default, a pretrained model on MegaAge and MegaAge-Asian will be downloaded automatically. Otherwise, you can load a pretrained model using `--resume` argument.

## Train

#### Download Dataset

Download and extract the [APPA-REAL dataset](http://chalearnlap.cvc.uab.es/dataset/26/description/).

> The APPA-REAL database contains 7,591 images with associated real and apparent age labels. The total number of apparent votes is around 250,000. On average we have around 38 votes per each image and this makes the average apparent age very stable (0.3 standard error of the mean).

```bash
wget http://158.109.8.102/AppaRealAge/appa-real-release.zip
unzip appa-real-release.zip
```

#### Train Model
Train a model using the APPA-REAL dataset.
See `python train.py -h` for detailed options.

```bash
python train.py --data_dir [PATH/TO/appa-real-release] --tensorboard tf_log
```

Check training progress:

```bash
tensorboard --logdir=tf_log
```

<img src="misc/tfboard.png" width="400px">

#### Training Options
You can change training parameters including model architecture using additional arguments like this:

```bash
python train.py --data_dir [PATH/TO/appa-real-release] --tensorboard tf_log MODEL.ARCH se_resnet50 TRAIN.OPT sgd TRAIN.LR 0.1
```

All default parameters defined in [defaults.py](defaults.py) can be changed using this style.


#### Test Trained Model
Evaluate the trained model using the APPA-REAL test dataset.

```bash
python test.py --data_dir [PATH/TO/appa-real-release] --resume [PATH/TO/BEST_MODEL.pth]
```

After evaluation, you can see something like this:

```bash
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 16/16 [00:08<00:00,  1.28it/s]
test mae: 4.800
```
