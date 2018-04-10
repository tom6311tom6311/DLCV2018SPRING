import os
import shutil
import numpy as np
from scipy.misc import *
from scipy import linalg

faces_count = 40
faces_dir = 'data'
out_dir = 'results'
num_efaces_to_gen = 3
idx_to_recon = [1,1]
n_efaces_to_recon = [3,50,100,239]
k_candidates = [1,3,5]
n_candidates = [3,50,159]
k_for_test = 1
n_for_test = 159
train_faces_count = 6
test_faces_count = 4
num_cross_validation_folds = 3
cross_validation_split_count = train_faces_count / num_cross_validation_folds
n = train_faces_count * faces_count
img_height = 56
img_width = 46
p = img_height * img_width


# loads a greyscale version of every image in the directory.
# INPUT  : directory
# OUTPUT : imgs - n x p array (n images with p pixels each)
def load_images(idx=range(1, train_faces_count + 1)):
  imgs = np.empty(shape=(len(idx) * faces_count, p))
  curr_img_id = 0
  for face_id in xrange(1, faces_count + 1):
    for training_id in idx:
      imgs[curr_img_id] = imread(faces_dir + '/' + str(face_id) + '_' + str(training_id) + '.png', True).flatten()
      curr_img_id += 1
  return imgs

# Run Principal Component Analysis on the input data.
# INPUT  : data    - an n x p matrix
# OUTPUT : e_faces -
#          weights -
#          mu      -
def pca(data):
  mu = np.mean(data, 0)
  # mean adjust the data
  ma_data = data - mu
  # run SVD
  e_faces, sigma, v = linalg.svd(ma_data.transpose(), full_matrices=False)
  # compute weights for each image
  weights = np.dot(ma_data, e_faces)
  return e_faces, weights, mu

# reconstruct an image using the given number of principal
# components.
def reconstruct(img_idx, e_faces, weights, mu, npcs):
  # dot weights with the eigenfaces and add to mean
  recon = mu + np.dot(weights[img_idx, 0:npcs], e_faces[:, 0:npcs].T)
  return recon

def save_image(file_path, data):
	imsave(file_path, data.reshape((img_height, img_width)))

if __name__ == "__main__":
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  else:
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)
  data = load_images()
  e_faces, weights, mu = pca(data)
  save_image(out_dir + '/mean.png', mu)
  for i in xrange(0,num_efaces_to_gen):
    save_image(out_dir + '/eface_' + str(i) + '.png', e_faces[:,i])
  recon_idx = (idx_to_recon[0] - 1) * train_faces_count + (idx_to_recon[1] - 1)
  for n_efaces in n_efaces_to_recon:
    recon_img = reconstruct(recon_idx, e_faces, weights, mu, n_efaces)
    mse = ((recon_img - data[recon_idx]) ** 2).mean()
    print('Reconstruction MSE: ' + str(mse))
    save_image(out_dir + '/recon_' + str(idx_to_recon[0]) + '_' + str(idx_to_recon[1]) + '_' + str(n_efaces) + '.png', recon_img)

  # Start KNN with cross validation
  for validation_fold_id in xrange(0, num_cross_validation_folds):
    idx_to_train = np.arange((validation_fold_id + 1) * 2, (validation_fold_id + 1) * 2 + 4) % train_faces_count + 1
    idx_to_validate = np.arange((validation_fold_id) * 2, (validation_fold_id) * 2 + 2) + 1
    # print(idx_to_train)
    training_data = load_images(idx_to_train)
    validation_data = load_images(idx_to_validate)
    e_faces, weights, mu = pca(training_data)
    for n in n_candidates:
      for k in k_candidates:
        print('n=' + str(n) + ', k=' + str(k))
        correct_num = 0.0
        for valid_idx in xrange(0, validation_data.shape[0]):
          true_label = valid_idx / cross_validation_split_count + 1
          lower_dim_wieght = np.dot((validation_data[valid_idx] - mu), e_faces[:, 0:n])
          lower_dim_wieght = lower_dim_wieght.reshape(1, lower_dim_wieght.shape[0])
          dist_square = np.sum((weights[:,0:n] - lower_dim_wieght)**2, axis=1)
          closest_img_idx = np.argsort(dist_square)[0:k]
          closest_img_labels = closest_img_idx / (train_faces_count - cross_validation_split_count) + 1
          predict_label = np.argmax(np.bincount(closest_img_labels))
          if (predict_label == true_label):
            correct_num += 1
        print(correct_num / validation_data.shape[0])

  # Start testing
  data = load_images()
  e_faces, weights, mu = pca(data)
  testing_data = load_images(range(train_faces_count+1, train_faces_count+1+test_faces_count))
  correct_num = 0.0
  for test_idx in xrange(0, testing_data.shape[0]):
    true_label = test_idx / test_faces_count + 1
    lower_dim_wieght = np.dot((testing_data[test_idx] - mu), e_faces[:, 0:n])
    lower_dim_wieght = lower_dim_wieght.reshape(1, lower_dim_wieght.shape[0])
    dist_square = np.sum((weights[:,0:n] - lower_dim_wieght)**2, axis=1)
    closest_img_idx = np.argsort(dist_square)[0:k]
    closest_img_labels = closest_img_idx / (train_faces_count) + 1
    predict_label = np.argmax(np.bincount(closest_img_labels))
    if (predict_label == true_label):
      correct_num += 1
  print(correct_num / testing_data.shape[0])

