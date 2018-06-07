import sys
import os
import numpy as np
import tensorflow as tf
import preprocessor
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model


os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

MODEL_PATH = str(sys.argv[2])
FEAT_PATH = str(sys.argv[3])
OUT_DIR = str(sys.argv[4]) if str(sys.argv[4])[-1] == '/' else str(sys.argv[4]) + '/'

if not os.path.exists(OUT_DIR):
  os.makedirs(OUT_DIR)

valid_feats, valid_labels = preprocessor.load_feats_and_labels(FEAT_PATH)

classifier = load_model(MODEL_PATH)

predicted = classifier.predict(valid_feats)
predicted = np.argmax(predicted, axis=1)


print('Accuracy: ' + str(np.sum(np.equal(valid_labels, predicted)) / predicted.shape[0]))

preprocessor.write_predict_file(predicted, OUT_DIR + 'p1_valid.txt')