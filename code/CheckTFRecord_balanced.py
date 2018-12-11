# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 10:56:35 2018

@author: meite
"""
import tensorflow as tf
import numpy as np
from PIL import Image
import skimage.io
import os

ORI_IMAGE_SHAPE = (512,512,4)

path_to_train = '../input/train/'
path_to_test = '../input/test/'

trainf = tf.python_io.tf_record_iterator("../input_tf_balanced/Train-3.tfrecords")
testf = tf.python_io.tf_record_iterator("../input_tf_balanced/Test-.tfrecords")

def load_image(path):
    R = skimage.io.imread(path+'_red.png')
    Y = skimage.io.imread(path+'_yellow.png')
    G = skimage.io.imread(path+'_green.png')
    B = skimage.io.imread(path+'_blue.png')

    # use yellow somehow?
    image = np.stack((R, G, B,Y), -1)
    #image = np.divide(image, 255) # or standadize?
    return image

###############################################################################
# checking image
###############################################################################



def showimg(file_it):
    nxt = next(trainf)
    nxtparsed = tf.train.Example.FromString(nxt)
    imagepath = nxtparsed.features.feature["imagepath"].bytes_list.value[0].decode("utf-8") 
    labels = nxtparsed.features.feature["label"].bytes_list.value[0]
    
    path = os.path.join(path_to_train, imagepath)
    image = load_image(path)
    image = image.reshape([ORI_IMAGE_SHAPE[0],ORI_IMAGE_SHAPE[1],ORI_IMAGE_SHAPE[2]])

    labels = np.frombuffer(labels, dtype=np.uint8)
    print(labels)
    print(image[:,:,0:3].shape)
    RGB = Image.fromarray(image[:,:,0:3])
    RGB.show()
    #RGY = Image.fromarray(image[:,:,[0,1,3]])
    #RGY.show()

showimg(trainf)

#showimg(testf)