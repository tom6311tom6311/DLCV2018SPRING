import sys
import os
import cv2
import time
import datetime
import numpy as np
import model
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping

PRETRAINED_VGG_MODEL_PATH = 'model/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
OUTPUT_MODEL_PATH_PREFIX = 'model/out_'
TRAIN_DATA_DIR = 'data/train/'
MAX_EPOCHS = 100

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def load_data(dir):
  file_names = os.listdir(dir)
  img_list = [[]] * (len(file_names) / 2)
  label_list = [[]] * (len(file_names) / 2)
  for i,file_name in enumerate(file_names):
    [idx, tp] = file_name.split('.')[0].split('_')
    if tp == 'sat':
      img_list[int(idx)] = cv2.imread(dir + file_name)
    elif tp == 'mask':
      label_list[int(idx)] = cv2.imread(dir + file_name) / 255
    progress(i+1, len(file_names))
  return np.array(img_list), np.array(label_list)

if __name__ == '__main__':
  model = model.build_model()
  model.load_weights(PRETRAINED_VGG_MODEL_PATH, by_name=True)

  # Freeze the base layers.
  # for l, layer in enumerate(model.layers):
  #   if l < 19:
  #     layer.trainable = False

  model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['accuracy'])
  model.summary()

  print('loading training data...')
  train_imgs, train_labels = load_data(TRAIN_DATA_DIR)

  print('training...')
  model.fit(train_imgs, train_labels, epochs=MAX_EPOCHS, batch_size=16, callbacks=[EarlyStopping(monitor='loss', patience=3)])
  model.save_weights(OUTPUT_MODEL_PATH_PREFIX + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S') + '.h5')

  print('model saved.')  