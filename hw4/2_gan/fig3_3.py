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

MODEL = 'acgan'
TEST_DATA_DIR = '../data/test/'
TEST_LABEL_PATH = '../data/test.csv'
OUTPUT_IMG_DIR = 'fig3_3' + '/'
MODEL_PATH = str(sys.argv[2])
DISCRIM_DIMS = [(128, 5, 2), (256, 5, 2), (512, 5, 2)]
GEN_DIMS = [(512, 5, 2), (256, 5, 2), (128, 5, 2)]
LATENT_DIM = 100
DISENTANGLE_ATTR = 7

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
  test_data, test_labels, test_file_names = util.load_data(TEST_DATA_DIR, TEST_LABEL_PATH, flatten=False, val_range=(-1,1), num=10)

  print('\nLoading model...')
  if MODEL == 'dcgan':
    gan = DCGAN(test_data.shape[1], discrim_dims=DISCRIM_DIMS, gen_dims=GEN_DIMS, latent_dim=LATENT_DIM)
  elif MODEL == 'pool_dcgan':
    gan = PoolingDCGAN(test_data.shape[1], DISCRIM_DIMS, LATENT_DIM)
  else:
    gan = ACGAN(test_data.shape[1], discrim_dims=DISCRIM_DIMS, gen_dims=GEN_DIMS, latent_dim=LATENT_DIM, num_class=test_labels.shape[1])

  gan.generator.load_weights(MODEL_PATH)

  print('\nReconstructing...')
  noise = np.random.normal(0, 1, size=(10, LATENT_DIM))
  noise = np.concatenate(((noise, noise)))
  attrs = np.random.randint(0, 2, size=(10, test_labels.shape[1]+1))
  attrs[:, DISENTANGLE_ATTR] = 0
  attrs[:,-1] = 0
  attrs = np.concatenate((attrs, attrs))
  attrs[10:, DISENTANGLE_ATTR] = 1
  generated_images = gan.generator.predict([noise, attrs])

  print('\nSaving predictions...')
  img_paths = []
  for i in range(20):
    util.save_image(generated_images[i], OUTPUT_IMG_DIR + 'gen_' + str(i) + '.png', isFlattened=False)
    img_paths.append(OUTPUT_IMG_DIR + 'gen_' + str(i) + '.png')
    util.progress(i+1, 20)
  util.combine_images(64*10, 64*2, img_paths, 'fig3_3.jpg')
  
  print('\nfinished.')

