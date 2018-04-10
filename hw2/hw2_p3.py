import os
import shutil
import numpy as np
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

need_to_plot_des_clusters = False
need_to_plot_bow = False
problem3_dir = 'data/Problem3/'
trainset_path = 'train-100/'
testset_path = 'test-100/'
img_categories = ['Coast', 'Forest', 'Highway', 'Mountain', 'Suburb']
out_dir = 'results_p3/'
sample_img_path = 'train-10/Highway/image_0019.jpg'
num_kp_each_img = 30
num_kp_clusters = 50
kmeans_max_iterations = 5000
num_kp_clusters_to_show_on_plot = 6
plot_colors = ['b', 'g', 'r', 'c', 'm', 'y']
num_visual_words_to_use_for_bow = 50
img_idx_to_plot_in_bow = [1, 11, 21, 31, 41]
num_knn_neighbors = 5

def normalize(v):
  norm = np.linalg.norm(v)
  if norm == 0: 
      return v
  return v / norm

if __name__ == "__main__":
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  else:
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

  print('Comuting key points for sample images...')
  sample_img = cv2.imread(problem3_dir + sample_img_path, cv2.IMREAD_GRAYSCALE)
  surf = cv2.xfeatures2d.SURF_create()
  kp, des = surf.detectAndCompute(sample_img, None)
  sample_img_with_kp = cv2.drawKeypoints(sample_img,kp[:num_kp_each_img],None,(255,0,0),4)
  cv2.imwrite(out_dir + 'sample_img_with_kp.jpg', sample_img_with_kp)

  print('Computing key points for all train images...')
  dess = []
  labels = []
  for idx, category in enumerate(img_categories):
    for img_file_name in os.listdir(problem3_dir + trainset_path + category):
      img = cv2.imread(problem3_dir + trainset_path + category + '/' + img_file_name, cv2.IMREAD_GRAYSCALE)
      surf = cv2.xfeatures2d.SURF_create()
      kp, des = surf.detectAndCompute(img, None)
      new_des = np.zeros((num_kp_each_img, 64))
      for i,j in enumerate(des[:num_kp_each_img]):
        new_des[i][0:len(j)] = j
      dess.extend(new_des)
      labels.append(idx)
  dess = np.array(dess)
  labels = np.array(labels)
  print('Clustering key points...')
  kmeans = KMeans(n_clusters=num_kp_clusters, max_iter=kmeans_max_iterations, random_state=0).fit(dess)
  visual_words = kmeans.cluster_centers_

  if need_to_plot_des_clusters:
    print('Performing PCA dimension reduction to descriptors...')
    pca = PCA(n_components=3)
    pca.fit(dess)
    print('Plotting...')
    visual_words_3d = pca.transform(visual_words)
    dess_3d = pca.transform(dess)
    fig = plt.figure()
    ax = Axes3D(fig)
    for index, visual_word_3d in enumerate(visual_words_3d[:num_kp_clusters_to_show_on_plot]):
      ax.scatter(visual_word_3d[0], visual_word_3d[1], visual_word_3d[2], c=plot_colors[index], marker='^', s=100)
    for index, des_3d in enumerate(dess_3d):
      if kmeans.labels_[index] < num_kp_clusters_to_show_on_plot:
        ax.scatter(des_3d[0], des_3d[1], des_3d[2], c=plot_colors[kmeans.labels_[index]])
    plt.show()

  print('Computing BoWs...')
  hard_sum_bows = []
  soft_sum_bows = []
  soft_max_bows = []
  for img_idx in xrange(dess.shape[0]/num_kp_each_img):
    hard_sum_bow = np.zeros(num_visual_words_to_use_for_bow)
    soft_sum_bow = np.zeros(num_visual_words_to_use_for_bow)
    soft_max_bow = np.zeros(num_visual_words_to_use_for_bow)
    for kp_idx in xrange(num_kp_each_img):
      distance_squares = np.sum(np.square(visual_words[:num_visual_words_to_use_for_bow, :] - dess[img_idx * num_kp_each_img + kp_idx]), axis=1)
      near_to_far = np.argsort(distance_squares)
      hard_sum_bow[near_to_far[0]] += 1
      reciprocal = normalize(np.reciprocal(distance_squares))
      soft_sum_bow += reciprocal
      soft_max_bow = np.maximum(soft_max_bow, reciprocal)
    hard_sum_bows.append(normalize(hard_sum_bow))
    soft_sum_bows.append(normalize(soft_sum_bow))
    soft_max_bows.append(normalize(soft_max_bow))
  hard_sum_bows = np.array(hard_sum_bows)
  soft_sum_bows = np.array(soft_sum_bows)
  soft_max_bows = np.array(soft_max_bows)

  if need_to_plot_bow:
    fig = plt.figure()
    ax = Axes3D(fig)
    for idx, img_idx in enumerate(img_idx_to_plot_in_bow):
      ax.scatter(hard_sum_bows[img_idx][0], hard_sum_bows[img_idx][1], hard_sum_bows[img_idx][2], c=plot_colors[idx], marker='^')
      ax.scatter(soft_sum_bows[img_idx][0], soft_sum_bows[img_idx][1], soft_sum_bows[img_idx][2], c=plot_colors[idx], marker='o')
      ax.scatter(soft_max_bows[img_idx][0], soft_max_bows[img_idx][1], soft_max_bows[img_idx][2], c=plot_colors[idx], marker='D')
    plt.show()

  print('Loading test data...')  
  dess_test = []
  labels_test = []
  for idx, category in enumerate(img_categories):
    for img_file_name in os.listdir(problem3_dir + testset_path + category):
      img = cv2.imread(problem3_dir + testset_path + category + '/' + img_file_name, cv2.IMREAD_GRAYSCALE)
      surf = cv2.xfeatures2d.SURF_create()
      kp, des = surf.detectAndCompute(img, None)
      new_des = np.zeros((num_kp_each_img, dess[0].shape[0]))
      for i,j in enumerate(des[:num_kp_each_img]):
        new_des[i][0:len(j)] = j
      dess_test.extend(new_des)
      labels_test.append(idx)
  dess_test = np.array(dess_test)
  labels_test = np.array(labels_test)

  hard_sum_bows_test = []
  soft_sum_bows_test = []
  soft_max_bows_test = []
  for img_idx in xrange(dess_test.shape[0]/num_kp_each_img):
    hard_sum_bow = np.zeros(num_visual_words_to_use_for_bow)
    soft_sum_bow = np.zeros(num_visual_words_to_use_for_bow)
    soft_max_bow = np.zeros(num_visual_words_to_use_for_bow)
    for kp_idx in xrange(num_kp_each_img):
      distance_squares = np.sum(np.square(visual_words[:num_visual_words_to_use_for_bow, :] - dess_test[img_idx * num_kp_each_img + kp_idx]), axis=1)
      near_to_far = np.argsort(distance_squares)
      hard_sum_bow[near_to_far[0]] += 1
      reciprocal = normalize(np.reciprocal(distance_squares))
      soft_sum_bow += reciprocal
      soft_max_bow = np.maximum(soft_max_bow, reciprocal)
    hard_sum_bows_test.append(normalize(hard_sum_bow))
    soft_sum_bows_test.append(normalize(soft_sum_bow))
    soft_max_bows_test.append(normalize(soft_max_bow))
  hard_sum_bows_test = np.array(hard_sum_bows_test)
  soft_sum_bows_test = np.array(soft_sum_bows_test)
  soft_max_bows_test = np.array(soft_max_bows_test)

  print('Classifying using Bows...')
  knn_classifier = KNeighborsClassifier(n_neighbors=num_knn_neighbors)
  knn_classifier.fit(hard_sum_bows, labels)
  print('Accuracy of Hard-Sum BoW: ' + str(knn_classifier.score(hard_sum_bows_test, labels_test)))
  knn_classifier = KNeighborsClassifier(n_neighbors=num_knn_neighbors)
  knn_classifier.fit(soft_sum_bows, labels)
  print('Accuracy of Soft-Sum BoW: ' + str(knn_classifier.score(soft_sum_bows_test, labels_test)))
  knn_classifier = KNeighborsClassifier(n_neighbors=num_knn_neighbors)
  knn_classifier.fit(soft_max_bows, labels)
  print('Accuracy of Soft-Max BoW: ' + str(knn_classifier.score(soft_max_bows_test, labels_test)))

