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
TEST_DATA_DIR = '../data/test/'
OUTPUT_IMG_DIR = 'fig1_4' + '/'
MODEL_PATH = str(sys.argv[2])
# ENC_DIM = [4096, 1024, 512]
ENC_DIM = [(64, 5, 2), (128, 5, 2), (256, 5, 2)]
LATENT_DIM = 1024
KL_LAMBDA = 1e-5

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
  test_data, test_file_names = util.load_data(TEST_DATA_DIR, flatten=False, num=32)

  print('\nLoading model...')
  if MODEL == 'simple':
    autoenc = SimpleAE(test_data.shape[1], ENC_DIM[-1])
  elif MODEL == 'deep':
    autoenc = DeepAE(test_data.shape[1], ENC_DIM)
  elif MODEL == 'conv':
    autoenc = ConvAE(test_data.shape[1], ENC_DIM)
  else:
    autoenc = VariationalAE(test_data.shape[1], ENC_DIM, LATENT_DIM, KL_LAMBDA)

  autoenc.load_weights(MODEL_PATH)

  print('\nReconstructing...')
  encoded_imgs = autoenc.encode(test_data)[2]
  decoded_imgs = autoenc.decode(encoded_imgs)

  print('\nSaving predictions...')
  img_paths = []
  for i in range(decoded_imgs.shape[0]):
    util.save_image(decoded_imgs[i,:], OUTPUT_IMG_DIR + 'dec_' + test_file_names[i], isFlattened=False)
    img_paths.append(OUTPUT_IMG_DIR + 'dec_' + test_file_names[i])
    util.progress(i+1, decoded_imgs.shape[0])
  util.combine_images(64*8, 64*4, img_paths, 'fig1_4.jpg')
  
  print('\nfinished.')

