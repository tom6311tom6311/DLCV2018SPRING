import sys
import shutil
import os
import cv2
import numpy as np
import model
import util
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

TRAINED_MODEL_PATH = sys.argv[1]
USE_BASELINE_MODEL = str(sys.argv[2])
VALID_DATA_DIR = str(sys.argv[3]) if str(sys.argv[3])[-1] == '/' else str(sys.argv[3]) + '/'
OUTPUT_DIR = str(sys.argv[4]) if str(sys.argv[4])[-1] == '/' else str(sys.argv[4]) + '/'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


if __name__ == '__main__':
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  else:
    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

  model = model.build_model(USE_BASELINE_MODEL)
  model.load_weights(TRAINED_MODEL_PATH)

  print('loading validation data...')
  test_imgs, test_labels = util.load_data(VALID_DATA_DIR)
  
  print('\npredicting...')
  test_label_predicted = np.around(model.predict(test_imgs))

  for idx,label in enumerate(test_label_predicted):
    cv2.imwrite(OUTPUT_DIR + str(idx).zfill(4) + '_mask.png', util.labelToRgb(label))
  
  print('finished')