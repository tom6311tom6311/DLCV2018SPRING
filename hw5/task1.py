import sys
import os
import preprocessor
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

all_feats, all_labels = preprocessor.load_feats_and_labels(True)

print(all_feats.shape)
print(all_labels.shape)