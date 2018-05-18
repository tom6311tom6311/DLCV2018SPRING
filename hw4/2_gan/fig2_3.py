import sys
import os
import shutil
from model.DCGAN import DCGAN
from model.PoolingDCGAN import PoolingDCGAN
from model.ACGAN import ACGAN
import util
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import numpy as np

MODEL = 'dcgan'
TEST_DATA_DIR = str(sys.argv[3]) if str(sys.argv[3])[-1] == '/' else str(sys.argv[3]) + '/'
TEST_LABEL_PATH = TEST_DATA_DIR + 'test.csv'
OUTPUT_IMG_DIR = 'fig2_3' + '/'
OUT_DIR = str(sys.argv[4]) if str(sys.argv[4])[-1] == '/' else str(sys.argv[4]) + '/'
MODEL_PATH = str(sys.argv[2])
DISCRIM_DIMS = [(128, 5, 2), (256, 5, 2), (512, 5, 2)]
GEN_DIMS = [(512, 5, 2), (256, 5, 2), (128, 5, 2)]
LATENT_DIM = 100

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

if __name__ == '__main__':
  if not os.path.exists(OUTPUT_IMG_DIR):
    os.makedirs(OUTPUT_IMG_DIR)
  else:
    shutil.rmtree(OUTPUT_IMG_DIR)
    os.makedirs(OUTPUT_IMG_DIR)

  print('\nLoading testing data...')
  test_data, test_labels, test_file_names = util.load_data(TEST_DATA_DIR, TEST_LABEL_PATH, flatten=False, val_range=(-1,1), num=32)

  print('\nLoading model...')
  if MODEL == 'dcgan':
    gan = DCGAN(64, discrim_dims=DISCRIM_DIMS, gen_dims=GEN_DIMS, latent_dim=LATENT_DIM)
  elif MODEL == 'pool_dcgan':
    gan = PoolingDCGAN(train_data.shape[1], DISCRIM_DIMS, LATENT_DIM)
  else:
    gan = ACGAN(64, discrim_dims=DISCRIM_DIMS, gen_dims=GEN_DIMS, latent_dim=LATENT_DIM, num_class=13)

  gan.generator.load_weights(MODEL_PATH)

  print('\nReconstructing...')
  noise = np.random.normal(0, 1, size=(32, LATENT_DIM))
  generated_images = gan.generator.predict(noise)

  print('\nSaving predictions...')
  img_paths = []
  for i in range(32):
    util.save_image(generated_images[i], OUTPUT_IMG_DIR + 'gen_' + test_file_names[i], isFlattened=False)
    img_paths.append(OUTPUT_IMG_DIR + 'gen_' + test_file_names[i])
    util.progress(i+1, 32)
  util.combine_images(64*8, 64*4, img_paths, OUT_DIR + 'fig2_3.jpg')
  
  print('\nfinished.')

