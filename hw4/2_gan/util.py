import sys
import os
import scipy.misc
import numpy as np
from PIL import Image

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def load_data(dir, label_path, flatten=True, val_range=(0,1), num=-1):
  num_class = 0
  raw_labels = []
  num_data = 0

  with open(label_path, 'r') as label_file:
    lines = label_file.readlines()
    num_class = len(lines[0].split(',')) - 1
    raw_labels = lines[1:]
    num_data = len(raw_labels)

  choices = np.arange(num_data)
  if (num != -1):
    choices = np.random.randint(0, num_data, num)
    raw_labels = [raw_labels[choice] for choice in choices]

  file_names = os.listdir(dir)
  file_names.sort()
  if (num != -1):
    file_names = [file_names[choice] for choice in choices]

  img_list = []
  for i,file_name in enumerate(file_names):
    img = (scipy.misc.imread(dir + file_name) / 255.0) * (val_range[1] - val_range[0]) + val_range[0]
    if flatten:
      img = img.flatten()
    img_list.append(img)
    progress(i+1, len(file_names))

  labels = np.zeros((len(img_list), num_class))
  for i,raw_label in enumerate(raw_labels):
    raw_label = raw_label.split(',')
    label = np.array([float(l) for l in raw_label[1:]])
    labels[i] = label
  return np.array(img_list), labels, file_names

def save_image(arr, path, isFlattened=True, val_range=(0,1)):
  img = arr
  if (isFlattened):
    w = int(np.sqrt(img.shape[-1]/3))
    img = img.reshape((w,w,3))
  img = ((img - val_range[0]) / (val_range[1] - val_range[0])) * 255
  img = np.around(img)
  scipy.misc.imsave(path, img)

def combine_images(total_w, total_h, paths, out_path):
  images = map(Image.open, paths)
  combined_img = Image.new('RGB', (total_w, total_h))
  x = 0
  y = 0
  for img in images:
    if (x + 64) <= total_w:
      combined_img.paste(img, (x,y))
      x += 64
    else:
      y += 64
      x = 0
      combined_img.paste(img, (x,y))
      x += 64
  combined_img.save(out_path)
