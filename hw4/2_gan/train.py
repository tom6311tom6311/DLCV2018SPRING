import sys
import os
import shutil
from model.DCGAN import DCGAN
import util
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

MODEL = 'dcgan'
TRAIN_DATA_DIR = '../data/train/'
TEST_DATA_DIR = '../data/test/'
OUTPUT_MODEL_DIR = 'out_' + MODEL + '/'
OUTPUT_MODEL_PATH_PREFIX = OUTPUT_MODEL_DIR + MODEL + '_'
OUTPUT_IMG_DIR = 'img_' + MODEL + '/'
DISCRIM_DIMS = [(32, 5, 2), (64, 5, 2), (128, 5, 2), (256, 3, 2), (512, 2, 2)]
LATENT_DIM = 512
MAX_EPOCHS = 10000
BATCH_SIZE = 64
CHK_POINT_INTERVAL = 10

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
  train_data, train_file_names = util.load_data(TRAIN_DATA_DIR, flatten=False, val_range=(-1,1))
  # print('\nLoading testing data...')
  # test_data, test_file_names = util.load_data(TEST_DATA_DIR, flatten=False)

  gan = DCGAN(train_data.shape[1], DISCRIM_DIMS, LATENT_DIM)
  gan.summary()

  print('\ntraining...')
  gan.train(
    train_data,
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    chk_point_interval=CHK_POINT_INTERVAL,
    out_model_prefix=OUTPUT_MODEL_PATH_PREFIX,
    out_img_dir=OUTPUT_IMG_DIR
  )
  print('\nfinished with models saved.')