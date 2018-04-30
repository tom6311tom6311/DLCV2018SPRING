import sys
import shutil
import os
import cv2
import numpy as np
import model


TRAINED_MODEL_PATH = sys.argv[1]
OUTPUT_DIR = 'out/'
VALID_DATA_DIR = 'data/validation/'

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def load_data(dir):
  file_names = os.listdir(dir)
  img_list = [[]] * (len(file_names) / 2)
  label_list = [[]] * (len(file_names) / 2)
  for i,file_name in enumerate(file_names):
    [idx, tp] = file_name.split('.')[0].split('_')
    if tp == 'sat':
      img_list[int(idx)] = cv2.imread(dir + file_name)
    elif tp == 'mask':
      label_list[int(idx)] = cv2.imread(dir + file_name) / 255
    progress(i+1, len(file_names))
  return np.array(img_list), np.array(label_list)

if __name__ == '__main__':
  if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
  else:
    shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

  model = model.build_model()
  model.load_weights(TRAINED_MODEL_PATH, by_name=True)

  print('loading validation data...')
  test_imgs, test_labels = load_data(VALID_DATA_DIR)
  
  print('predicting...')
  test_label_predicted = np.around(model.predict(test_imgs)) * 255

  for idx,label in enumerate(test_label_predicted):
    cv2.imwrite(OUTPUT_DIR + str(idx).zfill(4) + '_mask.png', label)
  
  print('finished')