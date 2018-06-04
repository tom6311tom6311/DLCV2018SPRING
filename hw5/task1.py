import sys
import os
import preprocessor
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Flatten
from keras.callbacks import EarlyStopping
from keras import regularizers

ENABLE_EARLY_STOP = False

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

train_feats, train_labels = preprocessor.load_feats_and_labels(True)
train_labels = np.eye(11)[train_labels.astype(np.uint8)]

valid_feats, valid_labels = preprocessor.load_feats_and_labels(False)
valid_labels = np.eye(11)[valid_labels.astype(np.uint8)]

print(train_feats.shape)
print(train_labels.shape)

classifier = Sequential()
classifier.add(Conv1D(32, 2, padding='same', strides=2, activation='relu', input_shape=(4, 1000)))
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


callbacks = [EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')] if ENABLE_EARLY_STOP else []
classifier.fit(train_feats, train_labels, validation_data=(valid_feats, valid_labels), epochs=100, batch_size=32, callbacks=callbacks)