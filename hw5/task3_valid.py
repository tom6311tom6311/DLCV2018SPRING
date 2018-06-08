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
VIDEO_DIR = str(sys.argv[3]) if str(sys.argv[3])[-1] == '/' else str(sys.argv[3]) + '/'
LABEL_PATH = str(sys.argv[4])
OUT_DIR = str(sys.argv[5]) if str(sys.argv[5])[-1] == '/' else str(sys.argv[5]) + '/'

if not os.path.exists(OUT_DIR):
  os.makedirs(OUT_DIR)

valid_feats, valid_labels = preprocessor.extract_full_length_feats(VIDEO_DIR, LABEL_PATH)

classifier = load_model(MODEL_PATH)

sliced_valid_feats, sliced_valid_labels = preprocessor.slice_feats(valid_feats, valid_labels, 10)
predicted = classifier.predict(sliced_valid_feats)
predicted = np.argmax(predicted, axis=1)
print('Accuracy: ' + str(np.sum(np.equal(sliced_valid_labels, predicted)) / predicted.shape[0]))

# preprocessor.write_predict_file(predicted, OUT_DIR + 'p3_valid.txt')