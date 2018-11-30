# Human Protein Atlas Image Classification (kaggle)



## Overview

Implementing [human protein atlas image classification from kaggle](https://www.kaggle.com/c/human-protein-atlas-image-classification) using tensorflow.

## Environment

```
conda create -n protein python=3.6 anaconda
```

```
activate protein
# or for unix
source activate protein
```

```
conda install tensorflow-gpu
conda install keras
pip install kaggle
conda update --all
```

Place the API credintials instructed here.
https://github.com/Kaggle/kaggle-api

Go to the working directory you decide.

```
git clone https://github.com/meitetsu3/HumanProtain.git
cd HumanProtein
kaggle competitions download -c human-protein-atlas-image-classification
```

Extract the downloaded zip file.

Create input  folder and move the extracted folders and csv files under input folder.

## Preparing data

go to code folder which is assumed working directory. Open TFRecord.py with your favorite IDE / editor.

In this project, we are converting the image files to TFRecord files.

create input_tf folder, and run the TFRecord.py.

You will get 12 .tfrecord files containing more than 31k training images(each stacked array of 4 gray scale images with different filters) and lables (multi-lables).

Based on your system  and preference, change the number of files you create.

Run CheckTFRecord.py to visualize some of the images from the tfrecord files. It shows 2 images with the first 3 channel (R,G,B) and another image swapping the 3rd channel B with Y.

image here: 

## First training



## Reviewing the model performance



## Predicting

