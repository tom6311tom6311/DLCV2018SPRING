import sys
import os
import shutil
from model.SimpleAE import SimpleAE
import util
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


TEST_DATA_DIR = '../data/test/'
OUTPUT_IMG_DIR = 'img_simple/'
MODEL_PATH = str(sys.argv[2])
ENC_DIM = 1024

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
  test_data, test_file_names = util.load_data(TEST_DATA_DIR)

  print('\nLoading model...')
  autoenc = SimpleAE(test_data.shape[1], ENC_DIM)
  autoenc.load_weights(MODEL_PATH)

  print('\nReconstructing...')
  encoded_imgs = autoenc.encode(test_data)
  decoded_imgs = autoenc.decode(encoded_imgs)

  print('\nSaving predictions...')
  for i in range(decoded_imgs.shape[0]):
    util.save_image(decoded_imgs[i,:], OUTPUT_IMG_DIR + test_file_names[i])
    util.progress(i+1, decoded_imgs.shape[0])
  
  print('\nfinished.')

