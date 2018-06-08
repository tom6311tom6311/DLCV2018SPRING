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
VIDEOS_DIR = str(sys.argv[3]) if str(sys.argv[3])[-1] == '/' else str(sys.argv[3]) + '/'
OUT_DIR = str(sys.argv[4]) if str(sys.argv[4])[-1] == '/' else str(sys.argv[4]) + '/'

if not os.path.exists(OUT_DIR):
  os.makedirs(OUT_DIR)

classifier = load_model(MODEL_PATH)

videos = os.listdir(VIDEOS_DIR)
videos.sort()

for i, video in enumerate(videos):
  valid_feats = preprocessor.extract_full_length_feats(VIDEOS_DIR + video + '/')
  sliced_valid_feats = preprocessor.slice_feats(valid_feats, 10)
  predicted = classifier.predict(sliced_valid_feats)
  predicted = np.argmax(predicted, axis=1)
  preprocessor.write_predict_file(predicted, OUT_DIR + video + '.txt', append=10)
  preprocessor.progress(i+1, len(videos))