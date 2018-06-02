import sys
import os
import preprocessor
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping

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
classifier.add(Dense(11, input_shape=(train_feats.shape[1],), activation='softmax'))
classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

classifier.fit(train_feats, train_labels, validation_data=(valid_feats, valid_labels), epochs=100, batch_size=32, callbacks=[EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')])