import sys
import os
import shutil
from model.SimpleAE import SimpleAE
from model.DeepAE import DeepAE
from model.ConvAE import ConvAE
from model.VariationalAE import VariationalAE
import util
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

MODEL = 'vae'
TRAIN_DATA_DIR = '../data/train/'
TEST_DATA_DIR = '../data/test/'
OUTPUT_MODEL_DIR = 'out_' + MODEL + '/'
OUTPUT_MODEL_PATH_PREFIX = OUTPUT_MODEL_DIR + MODEL + '_'
# ENC_DIM = [4096, 1024, 512]
ENC_DIM = [(64, 5, 2), (128, 5, 2), (256, 5, 2)]
LATENT_DIM = 1024
KL_LAMBDA = 1e-5
MAX_EPOCHS = 100
BATCH_SIZE = 64

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

if __name__ == '__main__':
  if not os.path.exists(OUTPUT_MODEL_DIR):
    os.makedirs(OUTPUT_MODEL_DIR)
  else:
    shutil.rmtree(OUTPUT_MODEL_DIR)
    os.makedirs(OUTPUT_MODEL_DIR)
  print('Loading training data...')
  train_data, train_file_names = util.load_data(TRAIN_DATA_DIR, flatten=False)
  print('\nLoading testing data...')
  test_data, test_file_names = util.load_data(TEST_DATA_DIR, flatten=False)

  if MODEL == 'simple':
    autoenc = SimpleAE(train_data.shape[1], ENC_DIM[-1])
  elif MODEL == 'deep':
    autoenc = DeepAE(train_data.shape[1], ENC_DIM)
  elif MODEL == 'conv':
    autoenc = ConvAE(train_data.shape[1], ENC_DIM)
  else:
    autoenc = VariationalAE(train_data.shape[1], ENC_DIM, LATENT_DIM, KL_LAMBDA)

  autoenc.summary()

  print('\ntraining...')
  autoenc.train(
    train_data,
    test_data,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    out_model_prefix=OUTPUT_MODEL_PATH_PREFIX
  )
  print('\nfinished with models saved.')
