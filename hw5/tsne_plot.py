import sys
import os
import preprocessor
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

FEAT_PATH = str(sys.argv[2])
MODEL_PATH = str(sys.argv[3])
OUT_IMG_PATH = str(sys.argv[4])

def plot_embedding(x, y):
  cm = plt.cm.get_cmap('RdYlGn')
  f = plt.figure(figsize=(13, 13))
  ax = plt.subplot(aspect='equal')
  sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=y, cmap=cm)
  plt.xlim(-25, 25)
  plt.ylim(-25, 25)
  ax.axis('off')
  ax.axis('tight')

  plt.savefig(OUT_IMG_PATH)

feats, labels = preprocessor.load_feats_and_labels(FEAT_PATH)
classifier = load_model(MODEL_PATH)

predicted = classifier.predict(feats)

embedded = TSNE(n_components=2).fit_transform(predicted)

plot_embedding(embedded, labels)