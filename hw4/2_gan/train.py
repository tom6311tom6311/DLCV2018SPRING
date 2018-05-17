import sys
import os
import shutil
from model.DCGAN import DCGAN
from model.PoolingDCGAN import PoolingDCGAN
from model.ACGAN import ACGAN
import util
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

MODEL = 'acgan'
TRAIN_DATA_DIR = '../data/train/'
TEST_DATA_DIR = '../data/test/'
TRAIN_LABEL_PATH = '../data/train.csv'
TEST_LABEL_PATH = '../data/test.csv'
OUTPUT_MODEL_DIR = 'out_' + MODEL + '/'
OUTPUT_MODEL_PATH_PREFIX = OUTPUT_MODEL_DIR + MODEL + '_'
OUTPUT_IMG_DIR = 'img_' + MODEL + '/'
DISCRIM_DIMS = [(128, 5, 2), (256, 5, 2), (512, 5, 2)]
GEN_DIMS = [(512, 5, 2), (256, 5, 2), (128, 5, 2)]
LATENT_DIM = 100
MAX_EPOCHS = 1000
BATCH_SIZE = 64
CHK_POINT_INTERVAL = 5

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
  if not os.path.exists(OUTPUT_IMG_DIR):
    os.makedirs(OUTPUT_IMG_DIR)
  else:
    shutil.rmtree(OUTPUT_IMG_DIR)
    os.makedirs(OUTPUT_IMG_DIR)
  print('Loading training data...')
  train_data, train_labels, train_file_names = util.load_data(TRAIN_DATA_DIR, TRAIN_LABEL_PATH, flatten=False, val_range=(-1,1))
  # print('\nLoading testing data...')
  # test_data, test_file_names = util.load_data(TEST_DATA_DIR, flatten=False)

  if MODEL == 'dcgan':
    gan = DCGAN(train_data.shape[1], discrim_dims=DISCRIM_DIMS, gen_dims=GEN_DIMS, latent_dim=LATENT_DIM)
  elif MODEL == 'pool_dcgan':
    gan = PoolingDCGAN(train_data.shape[1], DISCRIM_DIMS, LATENT_DIM)
  else:
    gan = ACGAN(train_data.shape[1], discrim_dims=DISCRIM_DIMS, gen_dims=GEN_DIMS, latent_dim=LATENT_DIM, num_class=train_labels.shape[1])

  gan.summary()

  print('\ntraining...')
  gan.train(
    train_data,
    train_labels,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    chk_point_interval=CHK_POINT_INTERVAL,
    out_model_prefix=OUTPUT_MODEL_PATH_PREFIX,
    out_img_dir=OUTPUT_IMG_DIR
  )
  print('\nfinished with models saved.')