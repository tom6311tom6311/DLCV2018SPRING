import os
import sys
import time
import datetime
import numpy as np
import model
import util
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model

PRETRAINED_VGG_MODEL_PATH = 'model/vgg16_weights_tf_dim_ordering_tf_kernels.h5'
OUTPUT_MODEL_PATH_PREFIX = 'model/out_'
TRAIN_DATA_DIR = 'data/train/'
MAX_EPOCHS = 100

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1]) or '0'

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

if __name__ == '__main__':
  model = model.build_model()
  model.load_weights(PRETRAINED_VGG_MODEL_PATH, by_name=True)

  # Freeze the base layers.
  # for l, layer in enumerate(model.layers):
  #   if l < 19:
  #     layer.trainable = False

  model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
  model.summary()
  # plot_model(model, show_shapes=True, to_file='model/model.png')

  print('loading training data...')
  train_imgs, train_labels = util.load_data(TRAIN_DATA_DIR)

  print('training...')
  model.fit(train_imgs, train_labels, epochs=MAX_EPOCHS, batch_size=2, callbacks=[EarlyStopping(monitor='loss', patience=3), ModelCheckpoint(OUTPUT_MODEL_PATH_PREFIX + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S') + '-{epoch:02d}-{val_loss:.2f}.h5', period=5)])
  model.save_weights(OUTPUT_MODEL_PATH_PREFIX + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d%H%M%S') + '.h5')

  print('model saved.')
