
#docker run -p 8500:8500 \
#--mount type=bind,source=/media/user1/MyHDataStor1/kaggle/HumanProtain/code/model/201812222145_F1Loss_lr1e-05_678QtrHoldOut_RGBY/export/best_exporter,target=/models/HumanProtein \
#-e MODEL_NAME=HumanProtein -t tensorflow/serving &

SERVER_URL = 'http://localhost:8501/v1/models/HumanProtein/versions/1545542524:predict'

import tensorflow as tf
import numpy as np
import pandas as pd
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2

VAL_BATCH_SIZE=100
VAL_NO = 3
FILE_NO = 12-VAL_NO
INPUT_SHAPE = (224,224,3)
ORI_IMAGE_SHAPE = (512,512,4)
TEST_FILE = "../input_tf/Test-.tfrecords"

testf = tf.python_io.tf_record_iterator(TEST_FILE)
submit = pd.read_csv('../input/sample_submission.csv')

example_fmt = {
    "image": tf.io.FixedLenFeature((), tf.string, ""),
    "label": tf.io.FixedLenFeature((), tf.string, ""),
 }


###############################################################################
#  Requesting
###############################################################################
channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'HumanProtein'
request.model_spec.version.value = 1545542524
request.model_spec.signature_name = 'serving_default'
request.inputs['image_input'].dtype = types_pb2.DT_STRING

predicted = []
for i in testf:
    request.inputs['image_input'].CopyFrom(tf.contrib.util.make_tensor_proto(i,shape=[1]))
    result = stub.Predict(request, 5.0)
    label_predict = np.arange(28)[np.array(result.outputs['predictions'].float_val)>=0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
submit.to_csv('mysubmission05.csv', index=False)
