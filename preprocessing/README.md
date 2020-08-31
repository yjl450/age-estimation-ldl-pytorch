## Format
There should be two csv files generated for each dataset, one for training and one for validation, all of which follow the naming convention:
```bash
[dataset_name]_train_align.csv
[dataset_name]_valid_align.csv
```
In each file, 7 fields are required (the order is not important):
* photo - the relative path to each image from the dataset folder
* age - the ground truth age
* deg - the angle to rotate the image so that the two eyes are on a horizontal line, computed from the facial landmarks
* box1, box2, box3, box4, (x1, y1) (x2, y2) coordinates to crop the face from the photo


## Data Preparation
Since each dataset provides the data in different format, the preprocessing script needs to be tailored to each dataset individually. Data preparation helps to extract the information we want from the various formats provided (.txt .mat .xls).

## Image Preprocessing
We use MTCNN for face detection and alignment. Each image should go through two passes. In the first pass, five points of the facial landmarks are detected and the rotation angle is computed. In the second pass, the rotated image is used to find the bounding box of the face. The rotation angle and the bounding box are saved in the csv files.

TODO: face alignment image

## Sample script

Here, we provide two sample scripts for the MegaAge-Asian dataset. Place the two script file in the root folder of the dataset, then run them in the following order.

```bash
python generate_csv.py
```
This piece of script read the txt file provided in the dataset and generate two csv files containing image paths and corresponding ages.

```bash
python face_align.py
```
This piece of script adds the rotation angle and bounding box to the csv files.