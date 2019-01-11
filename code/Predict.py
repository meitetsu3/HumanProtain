
#docker pull tensorflow/serving
#docker run -p 8500:8500 \
#--mount type=bind,source=/media/user1/MyHDataStor1/kaggle/HumanProtain/code/model/201901030733_f1Loss_lr1e-02_678QtrHoldOut_50k/export/best_exporter,target=/models/HumanProtein \
#-e MODEL_NAME=HumanProtein -t tensorflow/serving &

import tensorflow as tf
import numpy as np
import pandas as pd
import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow.core.framework import types_pb2
from tqdm import tqdm

TEST_FILE = "../input_tf/Test-.tfrecords"

testf = tf.python_io.tf_record_iterator(TEST_FILE)
submit = pd.read_csv('../input/sample_submission.csv')

###############################################################################
#  Requesting
###############################################################################

channel = grpc.insecure_channel('localhost:8500')
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = 'HumanProtein'
request.model_spec.version.value = 1546778953
request.model_spec.signature_name = 'serving_default'
request.inputs['input'].dtype = types_pb2.DT_STRING

predicted = []
for i in tqdm(testf):
    request.inputs['input'].CopyFrom(tf.contrib.util.make_tensor_proto(i,shape=[1]))
    result = stub.Predict(request, 5.0)
    result_ar = np.array(result.outputs['predictions'].float_val)
    result_ar = result_ar+max(0,min(0.5-result_ar))
    label_predict = np.arange(28)[result_ar>=0.5]
    str_predict_label = ' '.join(str(l) for l in label_predict)
    predicted.append(str_predict_label)

submit['Predicted'] = predicted
submit.to_csv('mysubmission05.csv', index=False)
