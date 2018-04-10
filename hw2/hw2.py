import os
import shutil
import scipy.io as sio
import numpy as np
from skimage import io, color
from sklearn.cluster import KMeans

problem2_dir = 'data/Problem2/'
out_dir = 'results'
filter_bank_file_path = problem2_dir + 'filterBank.mat'
problem2_img_names = ['zebra.jpg', 'mountain.jpg']
kmeans_num_clusters = 10
kmeans_max_iterations = 1000

def load_image(path):
  img = io.imread(path)
  img_lab = color.rgb2lab(img)
  img_shape = img.shape
  img = img.reshape((-1,3))
  img_lab = img_lab.reshape((-1,3))
  return img, img_lab, img_shape

def save_image(img, path, shape):
  io.imsave(path, img.reshape(shape))

def seg_image(img, label_colors):
  kmeans = KMeans(n_clusters=kmeans_num_clusters, max_iter=kmeans_max_iterations, random_state=0).fit(img)
  seg = np.array([label_colors[i] for i in kmeans.labels_])
  return seg

if __name__ == "__main__":
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  else:
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

  for img_name in problem2_img_names:
    label_colors = np.random.randint(256, size=(kmeans_num_clusters, 3))
    img, img_lab, img_shape = load_image(problem2_dir + img_name)
    seg = seg_image(img, label_colors)
    seg_lab = seg_image(img_lab, label_colors)
    save_image(seg, out_dir + '/seg_rgb_' + img_name, img_shape)
    save_image(seg_lab, out_dir + '/seg_lab_' + img_name, img_shape)



# filter_bank = sio.loadmat(filter_bank_file_path)
