# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:56:35 2018

@author: meite
"""
import tensorflow as tf
import numpy as np
from PIL import Image
###############################################################################
# checking image
###############################################################################


trainf = tf.python_io.tf_record_iterator("../input_tf/Train-1.tfrecords")
testf = tf.python_io.tf_record_iterator("../input_tf/Test-.tfrecords")

def showimg(file_it):
    nxt = next(file_it)
    nxtparsed = tf.train.Example.FromString(nxt)
    height = nxtparsed.features.feature["height"].int64_list.value[0]
    width = nxtparsed.features.feature["width"].int64_list.value[0]
    channel = nxtparsed.features.feature["channel"].int64_list.value[0]
    image = nxtparsed.features.feature["image"].bytes_list.value[0]
    labels = nxtparsed.features.feature["label"].bytes_list.value[0]

    image = np.frombuffer(image, dtype=np.uint8)
    image = image.reshape([height,width,channel])

    labels = np.frombuffer(labels, dtype=np.uint8)
    print(labels)
    RGB = Image.fromarray(image[:,:,0:3])
    RGB.show()
    RGY = Image.fromarray(image[:,:,[0,1,3]])
    RGY.show()

showimg(trainf)

showimg(testf)