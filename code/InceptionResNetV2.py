import tensorflow as tf
import pandas as pd
import os
import glob
import random
from datetime import datetime
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Dropout, Flatten, Dense,BatchNormalization,Input
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet50 import ResNet50
from tensorflow.python.client import device_lib

tf.reset_default_graph()

# having this string i.e. len() > 0 will restore the best model and predict
# ottherwise, it will create new training and save the model.
restore_dir = './model/201812252059_F1Loss_lr1e-05_678QtrHoldOut_RGBY_2layers_freezeto-26'
restore_dir = ''

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

NUM_GPUS = len(get_available_gpus())
ORI_IMAGE_SHAPE = (512,512,4)
INPUT_SHAPE = (224,224,3)
#INPUT_SHAPE = (299,299,3)
CHECK_POINT_STEPS = 1000
BATCH_SIZE = 32 # 16 for gtx 1070 laptop, 32 or more for gtx 1080 ti
VAL_BATCH_SIZE=100
TRAIN_STEPS = 1000*3
lr = 1e-05
VAL_NO = 3
FILE_NO = 12-VAL_NO
if len(restore_dir) > 0:
    FILE_NO = 1

TRAIN_FILES = "../input_tf/Train-*.tfrecords"
VAL_FILES = "../input_tf/Val-*.tfrecords"
TMP_FILES = "../input_tf/temp-*.tfrecords"
TEST_FILE = "../input_tf/Test-.tfrecords"
MODEL_DIR_BASE = './model'
exptitle = 'F1Loss_lr1e-05_678QtrHoldOut_RGBY_2layers_freezeto-26'
traindata = pd.read_csv('../input/train.csv')

def get_model_folder():
    if len(restore_dir)>0:
        tensorboard_path = restore_dir
    else:
        folder_name = "/{0}_{1}".format(datetime.now().strftime("%Y%m%d%H%M"),exptitle)
        tensorboard_path = MODEL_DIR_BASE + folder_name
        if not os.path.exists(tensorboard_path):
            os.makedirs(tensorboard_path)
    return tensorboard_path

MODEL_DIR = get_model_folder()

###############################################################################
# model
###############################################################################

#image_input = tf.feature_column.numeric_column("image_input",shape = INPUT_SHAPE)
#labels = tf.feature_column.numeric_column("labels",shape = [28])
#feature_columns = [image_input, labels]


def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def f1_0(y_true, y_pred):
    return f1(y_true[:,0],y_pred[:,0])
def f1_1(y_true, y_pred):
    return f1(y_true[:,1],y_pred[:,1])
def f1_2(y_true, y_pred):
    return f1(y_true[:,2],y_pred[:,2])
def f1_3(y_true, y_pred):
    return f1(y_true[:,3],y_pred[:,3])
def f1_4(y_true, y_pred):
    return f1(y_true[:,4],y_pred[:,4])
def f1_5(y_true, y_pred):
    return f1(y_true[:,5],y_pred[:,5])
def f1_6(y_true, y_pred):
    return f1(y_true[:,6],y_pred[:,6])
def f1_7(y_true, y_pred):
    return f1(y_true[:,7],y_pred[:,7])
def f1_8(y_true, y_pred):
    return f1(y_true[:,8],y_pred[:,8])
def f1_9(y_true, y_pred):
    return f1(y_true[:,9],y_pred[:,9])
def f1_10(y_true, y_pred):
    return f1(y_true[:,10],y_pred[:,10])
def f1_11(y_true, y_pred):
    return f1(y_true[:,11],y_pred[:,11])
def f1_12(y_true, y_pred):
    return f1(y_true[:,12],y_pred[:,12])
def f1_13(y_true, y_pred):
    return f1(y_true[:,13],y_pred[:,13])
def f1_14(y_true, y_pred):
    return f1(y_true[:,14],y_pred[:,14])
def f1_15(y_true, y_pred):
    return f1(y_true[:,15],y_pred[:,15])
def f1_16(y_true, y_pred):
    return f1(y_true[:,16],y_pred[:,16])
def f1_17(y_true, y_pred):
    return f1(y_true[:,17],y_pred[:,17])
def f1_18(y_true, y_pred):
    return f1(y_true[:,18],y_pred[:,18])
def f1_19(y_true, y_pred):
    return f1(y_true[:,19],y_pred[:,19])
def f1_20(y_true, y_pred):
    return f1(y_true[:,20],y_pred[:,20])
def f1_21(y_true, y_pred):
    return f1(y_true[:,21],y_pred[:,21])
def f1_22(y_true, y_pred):
    return f1(y_true[:,22],y_pred[:,22])
def f1_23(y_true, y_pred):
    return f1(y_true[:,23],y_pred[:,23])
def f1_24(y_true, y_pred):
    return f1(y_true[:,24],y_pred[:,24])
def f1_25(y_true, y_pred):
    return f1(y_true[:,25],y_pred[:,25])
def f1_26(y_true, y_pred):
    return f1(y_true[:,26],y_pred[:,26])
def f1_27(y_true, y_pred):
    return f1(y_true[:,27],y_pred[:,27])

def f1_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    
    return 1-K.mean(f1)

def create_model(input_shape, n_out):

    pretrain_model = ResNet50(
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

def model_fn(features, labels, mode, params):
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        features = features['feature']
    x = Flatten()(features)
    x = Dropout(0.5)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(28, activation='sigmoid',name = 'predictions')(x)

    # Return if it is in prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'predictions': output}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    ### Compute loss.
    loss = f1_loss(y_true=labels, y_pred=output)

    # Return if in evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss)

    ### Create optimizer and trainer
    assert mode == tf.estimator.ModeKeys.TRAIN

    # Create the optimizer
    optimizer = tf.train.AdagradOptimizer(learning_rate=lr)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

###############################################################################
# strategy, config and estimator
###############################################################################

   
if NUM_GPUS == 1:
    strategy = tf.contrib.distribute.OneDeviceStrategy('/gpu:0')
elif NUM_GPUS > 1:
    strategy = tf.contrib.distribute.MirroredStrategy(num_gpus=NUM_GPUS)

config = tf.estimator.RunConfig(train_distribute=strategy
                                ,save_checkpoints_steps = CHECK_POINT_STEPS
                                , model_dir=MODEL_DIR)

estimator = tf.estimator.Estimator(model_fn=model_fn
                                   ,config = config)
#estimator = tf.keras.estimator.model_to_estimator(model,config=config)

###############################################################################
# data input pipeline
###############################################################################

def augment(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    nk = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k = nk)
    return image

example_fmt = {
        "image": tf.FixedLenFeature((), tf.string, ""),
        "label": tf.FixedLenFeature((), tf.string, ""),
    }

def extract_image(parsed, augment_flag=False):
  "extract features from parsed example (dict)"
  image = tf.decode_raw(parsed["image"], tf.uint8)
  image = tf.reshape(image,[ORI_IMAGE_SHAPE[0],ORI_IMAGE_SHAPE[1],ORI_IMAGE_SHAPE[2]])
  image = tf.div(tf.cast(image,tf.float32), 255.0)
  image = tf.stack([
          tf.reshape(image[:,:,0],[ORI_IMAGE_SHAPE[0],ORI_IMAGE_SHAPE[1]])
         ,tf.reshape(image[:,:,1],[ORI_IMAGE_SHAPE[0],ORI_IMAGE_SHAPE[1]])
         ,tf.reshape(image[:,:,2],[ORI_IMAGE_SHAPE[0],ORI_IMAGE_SHAPE[1]])
                     +tf.reshape(image[:,:,3],[ORI_IMAGE_SHAPE[0],ORI_IMAGE_SHAPE[1]])]
         ,axis=2)
  image = tf.image.resize_images(image, (INPUT_SHAPE[0], INPUT_SHAPE[1]))
  if augment_flag:
      image = augment(image)
  return image

def extract_label(parsed):
  "extract features from parsed example (dict)"
  labels = tf.decode_raw(parsed["label"], tf.uint8)
  labels = tf.cast(labels, tf.float32)
  return labels

def parse_fn(example,augment_flag):
  "Parse TFExample records and perform simple data augmentation."
  parsed = tf.parse_single_example(example, example_fmt)
  return extract_image(parsed, augment_flag=augment_flag), extract_label(parsed)

def input_fn(input_files,mode,batch_size=16,repeat_count=1):
    if mode == tf.estimator.ModeKeys.TRAIN:
        repeat_count = None
    shuffle_buffer_size = 10*batch_size
    files = tf.data.Dataset.list_files(input_files)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=FILE_NO)
    if mode != tf.estimator.ModeKeys.PREDICT:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        augment_flag = True
    else:
        augment_flag = False
    dataset = dataset.map(map_func=lambda example:parse_fn(example,augment_flag=augment_flag)
        , num_parallel_calls=os.cpu_count())
    #dataset = dataset.cache()
    dataset = dataset.repeat(repeat_count)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size = None)
    return dataset

def serving_input_fn():
    input_ph = tf.placeholder(tf.string, [1])
    features = tf.parse_example(input_ph,example_fmt)
    features=tf.map_fn(extract_image, features,dtype=tf.float32)
    return tf.estimator.export.ServingInputReceiver(features, input_ph)

###############################################################################
# train and evaluate, or predict
###############################################################################

class evalhook(tf.train.SessionRunHook):
    def end(self, session):
        valfiles =  glob.glob(VAL_FILES)
        trainfiles = random.sample(glob.glob(TRAIN_FILES), VAL_NO)
        for f in valfiles:
            os.rename(f, f.replace("Val","temp")) 
        tempfiles = glob.glob(TMP_FILES)
        for i,f in enumerate(trainfiles):
            os.rename(f, valfiles[i])
        for i,f in enumerate(tempfiles):
            os.rename(f, trainfiles[i])

            
exporter = tf.estimator.BestExporter('best_exporter', serving_input_fn, exports_to_keep=5)

train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(input_files = TRAIN_FILES
                                                   ,batch_size=BATCH_SIZE
                                                   ,mode = tf.estimator.ModeKeys.TRAIN)
                                                    ,max_steps = TRAIN_STEPS)

eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(input_files = VAL_FILES
                                                   ,batch_size=VAL_BATCH_SIZE
                                                   ,mode = tf.estimator.ModeKeys.EVAL)
                                                    ,steps = 200
                                                    ,start_delay_secs = 0
                                                    #,hooks = [evalhook()]
                                                    ,exporters = exporter)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
