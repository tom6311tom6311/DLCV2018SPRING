import sys
import os
from model.SimpleAE import SimpleAE
import util
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


TRAIN_DATA_DIR = '../data/train/'
TEST_DATA_DIR = '../data/test/'
OUTPUT_MODEL_PATH_PREFIX = 'out/simple_'
ENC_DIM = 1024
MAX_EPOCHS = 100
BATCH_SIZE = 64

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

if __name__ == '__main__':
  print('Loading training data...')
  train_data = util.load_data(TRAIN_DATA_DIR)
  print('\nLoading testing data...')
  test_data = util.load_data(TEST_DATA_DIR)

  autoenc = SimpleAE(train_data.shape[1], ENC_DIM)
  autoenc.summary()

  print('\ntraining...')
  autoenc.train(
    train_data,
    test_data,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    out_model_prefix='../' + OUTPUT_MODEL_PATH_PREFIX
  )
  print('\nfinished with models saved.')

