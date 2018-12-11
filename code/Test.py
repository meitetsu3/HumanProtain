#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:26:35 2018

@author: user1
"""

import tensorflow as tf
from PIL import Image
import numpy as np
INPUT_SHAPE = (299,299,3)

trainf = tf.python_io.tf_record_iterator("../input_tf/Train-1.tfrecords")
ite = next(trainf)

def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    nk = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k = nk)
    return image


def parse_fn(example):
  "Parse TFExample records and perform simple data augmentation."
  example_fmt = {
    "height": tf.io.FixedLenFeature((), tf.int64, -1),
    "width": tf.io.FixedLenFeature((), tf.int64, -1),
    "channel": tf.io.FixedLenFeature((), tf.int64, -1),
    "image": tf.io.FixedLenFeature((), tf.string, ""),
    "label": tf.io.FixedLenFeature((), tf.string, ""),
  }
  parsed = tf.parse_single_example(example, example_fmt)
  image = tf.decode_raw(parsed["image"], tf.uint8)
  image = tf.reshape(image,[parsed["height"],parsed["width"],4])
  image = tf.div(tf.cast(image,tf.float32), 255.0)

  image = tf.stack([tf.reshape(image[:,:,0],[parsed["height"],parsed["width"]])
                  ,tf.reshape(image[:,:,1],[parsed["height"],parsed["width"]])
                  ,tf.reshape(image[:,:,2],[parsed["height"],parsed["width"]])
                  +tf.reshape(image[:,:,3],[parsed["height"],parsed["width"]])],axis=2)

  image = tf.slice(image,[0,0,0],[parsed["height"],parsed["width"],3])
  image = tf.image.resize_images(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
  image = augment(image)

  labels = tf.decode_raw(parsed["label"], tf.uint8)
  labels = tf.cast(labels, tf.float32)
  return image, labels

 
image, labels = parse_fn(ite)

with tf.Session() as sess:
    image_tf = sess.run(tf.contrib.image.rotate(image,tf.random.uniform([1])))
    labels_tf = sess.run(labels)
    r = sess.run(tf.random.uniform([1]))
    #test = sess.run(parse_fn(ite))

#test.shape

#RGB = Image.fromarray((test*255).astype(np.uint8))
#RGB.show()


#(image_tf*255).astype(int).shape

RGB = Image.fromarray((image_tf*255).astype(np.uint8))
RGB.show()


file0.tfrecord: [0, 1]
file1.tfrecord: [2, 3]
file2.tfrecord: [4]

with tf.python_io.TFRecordWriter('./file2.tfrecord') as writer:
    for r in range(4,5):
        example = tf.train.Example(features=tf.train.Features(feature={
                "n": tf.train.Feature(int64_list=tf.train.Int64List(value=[r]))}))
        writer.write(example.SerializeToString())
    writer.close()

    
files = ["file{}.tfrecord".format(i) for i in range(3)]
files = tf.data.Dataset.from_tensor_slices(files)
dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x),
                           cycle_length=3, block_length=1)

dataset = dataset.repeat(None)

iterator = dataset.make_one_shot_iterator()
x = iterator.get_next()

with tf.Session() as sess:
    for _ in range(12):
        print(sess.run(x))

