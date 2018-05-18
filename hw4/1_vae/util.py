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

def load_data(dir, flatten=True, val_range=(0,1), num=-1):
  file_names = os.listdir(dir)
  file_names.sort()
  if (num != -1):
    file_names = np.random.choice(file_names, num)
  img_list = []
  for i,file_name in enumerate(file_names):
    img = (scipy.misc.imread(dir + file_name) / 255.0) * (val_range[1] - val_range[0]) + val_range[0]
    if flatten:
      img = img.flatten()
    img_list.append(img)
    progress(i+1, len(file_names))
  return np.array(img_list), file_names

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
