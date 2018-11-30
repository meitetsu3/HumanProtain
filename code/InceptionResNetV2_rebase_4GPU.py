
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os, time
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense,BatchNormalization,Input
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

NUM_GPUS = len(get_available_gpus())
ORI_IMAGE_SHAPE = (512,512,4)
INPUT_SHAPE = (299,299,3)
CHECK_POINT_STEPS = 1000
BATCH_SIZE = 16 # 16 for gtx 1070 laptop, for gtx 1080 ti
VAL_BATCH_SIZE=50
TRAIN_STEPS = 3100*2
lr = 1e-05

TRAIN_FILES = "../input_tf/Train-*.tfrecords"
VAL_FILES = "../input_tf/Val-*.tfrecords"
TEST_FILES = "../input_tf/Test-*.tfrecords"
MODEL_DIR = './model'
TFRECORD_NAME = "Train.tfrecords"
path_to_test = '../input/test/'
traindata = pd.read_csv('../input/train.csv')
exptitle = 'exp'

###############################################################################
#  Util
###############################################################################

def get_model_folder():
    folder_name = "/{0}_{1}".format(datetime.now().strftime("%Y%m%d%H%M"),exptitle)
    tensorboard_path = MODEL_DIR + folder_name
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)
    return tensorboard_path

###############################################################################
# model
###############################################################################

def create_model(input_shape, n_out):

    pretrain_model = InceptionResNetV2(
    include_top=False, 
    weights='imagenet', 
    input_shape=input_shape)    
    
    input_tensor = Input(shape=input_shape, name = 'image_input')
    bn = BatchNormalization()(input_tensor)
    x = pretrain_model(bn)
    #bn = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(n_out, activation='sigmoid',name = 'predictions')(x)
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
    optimizer=tf.train.AdamOptimizer(lr),
    metrics=['acc', f1])

###############################################################################
# strategy, config and estimator
###############################################################################

if NUM_GPUS == 1:
    strategy = tf.contrib.distribute.OneDeviceStrategy('/gpu:0')
elif NUM_GPUS > 1:
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)

config = tf.estimator.RunConfig(train_distribute=strategy
                                ,save_checkpoints_steps = CHECK_POINT_STEPS
                                , model_dir=get_model_folder())

estimator = tf.keras.estimator.model_to_estimator(model,config=config)

###############################################################################
# metrics
###############################################################################


#def my_rmse(labels, predictions):
#    pred_values = predictions['predictions']
#    return {"rmse": tf.metrics.root_mean_squared_error(labels, pred_values)}
#
#estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)

###############################################################################
# data input pipeline
###############################################################################

def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    nk = tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32)
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
  image = tf.div(tf.cast(image,tf.float32), 255.0)
  
  labels = tf.decode_raw(parsed["label"], tf.uint8)
  labels = tf.cast(labels, tf.float32)
  return image, labels

def input_fn(input_files,mode,batch_size=16,repeat_count=1):
    if mode == tf.estimator.ModeKeys.TRAIN:
        repeat_count = None
    shuffle_buffer_size = 10*batch_size
    files = tf.data.Dataset.list_files(input_files)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=os.cpu_count())
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(map_func=parse_fn, num_parallel_calls=os.cpu_count())
      #dataset = dataset.cache()
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size = None)
    return dataset


###############################################################################
# train and evaluate
###############################################################################


def serving_input_fn():
    feature_placeholders = {
        'image_input': tf.placeholder(tf.float64, [None,INPUT_SHAPE[0],INPUT_SHAPE[1],INPUT_SHAPE[2]]),
    }
    features = {
        key: tensor
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

exporter = tf.estimator.BestExporter('best_exporter', serving_input_fn, exports_to_keep=5)

train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(input_files = TRAIN_FILES
                                                   ,batch_size=BATCH_SIZE
                                                   ,mode = tf.estimator.ModeKeys.TRAIN)
                                                    ,max_steps = TRAIN_STEPS)

eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(input_files = VAL_FILES
                                                   ,batch_size=VAL_BATCH_SIZE
                                                   ,mode = tf.estimator.ModeKeys.EVAL)
                                                    ,steps = 100
                                                    ,start_delay_secs = 0
                                                    ,exporters = exporter)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
