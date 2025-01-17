import sys
import os
import preprocessor
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten
from keras.callbacks import EarlyStopping, TensorBoard
from keras import regularizers

ENABLE_EARLY_STOP = True

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

TRAIN_FEAT_PATH = str(sys.argv[2])
VALID_FEAT_PATH = str(sys.argv[3])
TASK1_LOG_DIR = 'log_task1/'
LOG_SUB_DIR = str(sys.argv[4]) if str(sys.argv[4])[-1] == '/' else str(sys.argv[4]) + '/'

if not os.path.exists(TASK1_LOG_DIR):
  os.makedirs(TASK1_LOG_DIR)
if not os.path.exists(LOG_SUB_DIR):
  os.makedirs(LOG_SUB_DIR)


train_feats, train_labels = preprocessor.load_feats_and_labels(TRAIN_FEAT_PATH)
train_labels = np.eye(11)[train_labels]

valid_feats, valid_labels = preprocessor.load_feats_and_labels(VALID_FEAT_PATH)
valid_labels = np.eye(11)[valid_labels]

print(train_feats.shape)
print(train_labels.shape)

classifier = Sequential()
classifier.add(Conv1D(32, 2, padding='same', strides=2, activation='relu', input_shape=train_feats.shape[1:]))
classifier.add(Dropout(0.3))
classifier.add(Conv1D(32, 2, padding='same', strides=2, activation='relu'))
classifier.add(Dropout(0.3))
classifier.add(Flatten())
# classifier.add(Dense(32, input_shape=(train_feats.shape[1],), activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
# classifier.add(Dropout(0.3))
# classifier.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.0005)))
# classifier.add(Dropout(0.3))
classifier.add(Dense(11, activation='softmax'))
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


callbacks = []
if ENABLE_EARLY_STOP:
  callbacks.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto'))
callbacks.append(TensorBoard(log_dir=LOG_SUB_DIR))

classifier.fit(train_feats, train_labels, validation_data=(valid_feats, valid_labels), epochs=100, batch_size=32, callbacks=callbacks)
classifier.save(LOG_SUB_DIR + 'model.hdf5', overwrite=True, include_optimizer=False)
print('Model saved.')