

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os, time
import numpy as np
import skimage.io
import cv2
from imgaug import augmenters as iaa
from sklearn.model_selection import train_test_split

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,Input
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint


ORI_IMAGE_SHAPE = (512,512,4)
INPUT_SHAPE = (299,299,3)
BATCH_SIZE = 10
VAL_BATCH_SIZE=50
EPOCHS = 5
NUM_GPUS = 4

MODEL_DIR = './model'
TFRECORD_NAME = "Train.tfrecords"
path_to_test = '../input/test/'
traindata = pd.read_csv('../input/train.csv')

###############################################################################
# 
###############################################################################



###############################################################################
# model
###############################################################################

def create_model(input_shape, n_out):

    pretrain_model = InceptionResNetV2(
    include_top=False, 
    weights='imagenet', 
    input_shape=input_shape)    
    
    input_tensor = Input(shape=input_shape)
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    #bn = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid')(x)
    model = Model(input_tensor, output)
    return model

def f1(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

K.clear_session()

model = create_model(
    input_shape=INPUT_SHAPE,
    n_out=28)

model.summary()

model.compile(
    loss='categorical_crossentropy', 
    optimizer=tf.train.AdamOptimizer(1e-05),
    metrics=['acc', f1])

###############################################################################
# estimator
###############################################################################

strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)
config = tf.estimator.RunConfig(train_distribute=strategy)
estimator = tf.keras.estimator.model_to_estimator(model,config=config,model_dir=MODEL_DIR)

###############################################################################
# data input pipeline
###############################################################################

def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    nk = tf.random_uniform((1), minval=0, maxval=3, dtype=tf.uint8)
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
  image = tf.slice(image,[0,0,0],[parsed["height"],parsed["width"],3])
  image = tf.image.resize_images(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
  image = augment(image)
  image = tf.div(image, 255)
  labels = tf.decode_raw(parsed["label"], tf.float64)
  return image, labels

def input_fn():
  files = tf.data.Dataset.list_files("../input_tf/Train-*.tfrecords")
  dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=os.cpu_count())
  dataset = dataset.shuffle(buffer_size=100)
  dataset = dataset.map(map_func=parse_fn, num_parallel_calls=os.cpu_count())
  #dataset = dataset.cache()
  dataset = dataset.batch(batch_size=16)
  dataset = dataset.prefetch(buffer_size = None)
  return dataset

class TimeHistory(tf.train.SessionRunHook):
    def begin(self):
        self.times = []
    def before_run(self, run_context):
        self.iter_time_start = time.time()
    def after_run(self, run_context, run_values):
        self.times.append(time.time() - self.iter_time_start)

time_hist = TimeHistory()

checkpointer = ModelCheckpoint(
    '../working/InceptionResNetV2_base.model', 
    verbose=2, 
    save_best_only=True)

epochs = 5; 

# train model
history = estimator.train(input_fn=lambda:input_fn())


#model = load_model(
#    '/kaggle/working/InceptionResNetV2_base.model', 
#    custom_objects={'f1': f1})

submit = pd.read_csv('../input/sample_submission.csv')

predicted = []
for name in tqdm(submit['Id']):
    path = os.path.join('../input/test/', name)
    image = data_generator.load_image(path, INPUT_SHAPE)
    score_predict = model.predict(image[np.newaxis])[0]
    label_predict = np.arange(28)[score_predict>=0.2]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
submit.to_csv('submission_InceptionResNetV2_base_1bnlr5_ep350_.csv', index=False)
