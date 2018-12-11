#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Creating TF record
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import skimage.io
import os
import glob

path_to_train = '../input/train/'
path_to_test = '../input/test/'

traindata = pd.read_csv('../input/train.csv')
testdata = pd.read_csv('../input/test.csv')



###############################################################################
# importing file base name and lables -> path, labels dict object 
# train_dataset_info
###############################################################################

def createTFRecord(file_path, Test = False):
    path_label_list= []
    if Test:
        for name in file_path['Id']:
            path_label_list.append({
                'path':name,
                'labels':np.array([0])}) # test labels is dummy. not used.
    else:
        for name, labels in zip(file_path['Id'], file_path['Target'].str.split(' ')):
            path_label_list.append({
                'path':name,
                'labels':np.array([int(label) for label in labels])})
            
    return np.array(path_label_list)

train_dataset_info = createTFRecord(traindata)

test_dataset_info = createTFRecord(testdata, Test=True)
    
###############################################################################
# train data. class separation,  count, and repeat to align length to the max
# make one list and shuffle
###############################################################################

def repeat_to_length(string_to_expand, length):
    return (string_to_expand * (int(length/len(string_to_expand))+1))[:length]

img_path_by_class = [[] for i in range(28)]

for e in train_dataset_info:
    for l in e["labels"]:
        img_path_by_class[l].append(e["path"])

class_cnt = [[] for i in range(28)]
for i,c in enumerate(img_path_by_class):
    class_cnt[i] = len(c)

max_cnt = max(class_cnt)
#12885

for i,c in enumerate(img_path_by_class):
    img_path_by_class[i] = repeat_to_length(c,max_cnt)

#img_path_by_class_array = np.array(img_path_by_class)

img_path_balanced = [item for sublist in img_path_by_class for item in sublist]

#len(img_path_by_class) # 360,780
img_path_balanced = random.sample(img_path_balanced,len(img_path_balanced))

###############################################################################
# saving only path and lable
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

def CreateTensorflowReadFile(img_path_balanced, out_file):
    with tf.python_io.TFRecordWriter( out_file ) as writer:
        for r in img_path_balanced:
            labels = np.zeros((28),dtype=np.uint8)
            k = traindata.loc[traindata['Id'] == r]['Target'].to_string(index=False)
            labels[[int(label) for label in k.split(' ')]] = 1
            
            imagearray = load_image(os.path.join(path_to_train, r))    #uint8
    
            example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[labels.tobytes()])),
                    "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[imagearray.tobytes()]))
                    }))
            writer.write(example.SerializeToString())
    writer.close()
    
bs = 3436 # 360780/105

for b in range(105):
    CreateTensorflowReadFile(img_path_balanced[b*bs:(b+1)*bs] , "../input_tf_balanced/Train-"+str(b)+".tfrecords")

for i in range(0,30):
    f = "../input_tf_balanced/Train-"+str(i)+".tfrecords"
    os.rename(f, f.replace("Train","Val")) 