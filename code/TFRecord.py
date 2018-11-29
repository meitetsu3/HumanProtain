#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creating TF record
"""
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import skimage.io

path_to_train = '../input/train/'
path_to_test = '../input/test/'
traindata = pd.read_csv('../input/train.csv')

####################Image###########################################################
# importing file base name and lables -> path, labels dict object 
# train_dataset_info
###############################################################################
train_dataset_info = []

for name, labels in zip(traindata['Id'], traindata['Target'].str.split(' ')):
    train_dataset_info.append({
        'path':os.path.join(path_to_train, name),
        'labels':np.array([int(label) for label in labels])})

train_dataset_info = np.array(train_dataset_info)

###############################################################################
# saving 4 channel array image as bytes
###############################################################################

def load_image(path):
    R = skimage.io.imread(path+'_red.png')
    Y = skimage.io.imread(path+'_yellow.png')
    G = skimage.io.imread(path+'_green.png')
    B = skimage.io.imread(path+'_blue.png')

    # use yellow somehow?
    image = np.stack((R, G, B,Y), -1)
    #image = np.divide(image, 255) # or standadize?
    return image

def CreateTensorflowReadFile(dataset_info, out_file):
    with tf.python_io.TFRecordWriter( out_file ) as writer:
        for r in dataset_info:
            labels = np.zeros((28),dtype=np.uint8)
            imagearray = load_image(r['path'])    #uint8
            height = imagearray.shape[0]
            width = imagearray.shape[1]
            channel = imagearray.shape[2]
            #image = Image.fromarray(imagearray, 'RGB')
            labels[r['labels']] = 1
            image_raw = imagearray.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                    "height": tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
                    "width": tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
                    "channel": tf.train.Feature(int64_list=tf.train.Int64List(value=[channel])),
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
                    }))
            writer.write(example.SerializeToString())
    writer.close()


bs = 2590 #total 31072, 2590*12 = 31080

for b in range(12):
    CreateTensorflowReadFile(train_dataset_info[b*bs:(b+1)*bs] , "../input_tf/Train-"+str(b)+".tfrecords")

# original about 18GB
# x 4 channel 32.6 GB
# 3 channel 24.4 GB, 4/1 down
# 4 channel 0-1 float, 260GB
# 3 channel as RGB image, 24.4GB

# create test image file as well?


