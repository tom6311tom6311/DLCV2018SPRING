import os
import shutil
import scipy.io as sio
import scipy.signal as ssig
import numpy as np
from skimage import io, color
from sklearn.cluster import KMeans

problem2_dir = 'data/Problem2/'
out_dir = 'results_p2'
filter_bank_file_path = problem2_dir + 'filterBank.mat'
problem2_img_names = ['zebra.jpg', 'mountain.jpg']
kmeans_num_clusters_color_seg = 10
kmeans_num_clusters_texture_seg = 6
kmeans_max_iterations = 1000

def load_image(path):
  img = io.imread(path)
  img_lab = color.rgb2lab(img)
  img_grey = color.rgb2grey(img)
  img_shape = img.shape
  img = img.reshape((-1,3))
  img_lab = img_lab.reshape((-1,3))
  return img, img_lab, img_grey, img_shape

def save_image(img, path, shape):
  io.imsave(path, img.reshape(shape))

def seg_image(img, num_clusters, label_colors):
  kmeans = KMeans(n_clusters=num_clusters, max_iter=kmeans_max_iterations, random_state=0).fit(img)
  seg = np.array([label_colors[i] for i in kmeans.labels_])
  return seg

def filter_image(img_grey, filter_bank, img_shape):
  return np.array([ssig.convolve2d(img_grey, filter_bank[:,:,i].reshape(49,49), boundary='symm', mode='same') for i in xrange(filter_bank.shape[2])]).transpose((1,2,0)).reshape(-1, filter_bank.shape[2])

if __name__ == "__main__":
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  else:
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

  filter_bank = sio.loadmat(filter_bank_file_path)['F']
  label_colors = np.random.randint(256, size=(kmeans_num_clusters_color_seg, 3))

  for img_name in problem2_img_names:
    print('\n\nStart processing ' + img_name)
    img, img_lab, img_grey, img_shape = load_image(problem2_dir + img_name)
    print('Filtering...')
    img_filtered = filter_image(img_grey, filter_bank, img_shape)
    print('Concating features...')
    img_rgb_filtered = np.concatenate((img, img_filtered), axis=1)
    img_lab_filtered = np.concatenate((img_lab, img_filtered), axis=1)
    print('Doing segmentation...')
    seg = seg_image(img, kmeans_num_clusters_color_seg, label_colors)
    seg_lab = seg_image(img_lab, kmeans_num_clusters_color_seg, label_colors)
    seg_filtered = seg_image(img_filtered, kmeans_num_clusters_texture_seg, label_colors)
    seg_rgb_filtered = seg_image(img_rgb_filtered, kmeans_num_clusters_texture_seg, label_colors)
    seg_lab_filtered = seg_image(img_lab_filtered, kmeans_num_clusters_texture_seg, label_colors)
    print('Saving image...')
    save_image(seg, out_dir + '/seg_rgb_' + img_name, img_shape)
    save_image(seg_lab, out_dir + '/seg_lab_' + img_name, img_shape)
    save_image(seg_filtered, out_dir + '/seg_filtered_' + img_name, img_shape)
    save_image(seg_rgb_filtered, out_dir + '/seg_rgb_filtered_' + img_name, img_shape)
    save_image(seg_lab_filtered, out_dir + '/seg_lab_filtered_' + img_name, img_shape)
