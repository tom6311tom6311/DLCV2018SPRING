import sys
import os
import cv2
import numpy as np

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def onehot(a):
  out = np.zeros( (a.size,8), dtype=np.uint8)
  out[np.arange(a.size),a.ravel()] = 1
  out.shape = a.shape + (8,)
  return out

def rgbToLabel(img):
  r_layer = img[:,:,2] / 255
  g_layer = img[:,:,1] / 255
  b_layer = img[:,:,0] / 255
  class_labels = r_layer * 4 + g_layer * 2 + b_layer * 1
  return onehot(class_labels.astype(int))

def labelToRgb(label):
  tmp_label = np.argmax(label, axis=2)
  b_layer = tmp_label % 2
  tmp_label -= b_layer
  tmp_label = (tmp_label / 2).astype(int)
  g_layer = tmp_label % 2
  tmp_label -= g_layer
  tmp_label = (tmp_label / 2).astype(int)
  r_layer = tmp_label % 2
  img = np.transpose(np.array([b_layer, g_layer, r_layer]) * 255, (1, 2, 0))
  return img

def load_data(dir):
  file_names = os.listdir(dir)
  file_names.sort()
  img_list = []
  label_list = []
  for i,file_name in enumerate(file_names):
    tp = file_name.split('.')[0].split('_')[1]
    if tp == 'sat':
      img_list.append(cv2.imread(dir + file_name) / 255.0)
    elif tp == 'mask':
      label_list.append(rgbToLabel(cv2.imread(dir + file_name)))
    progress(i+1, len(file_names))
  return np.array(img_list), np.array(label_list)