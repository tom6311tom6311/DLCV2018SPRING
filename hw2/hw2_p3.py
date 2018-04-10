import os
import shutil
import numpy as np
import cv2

problem3_dir = 'data/Problem3/'
out_dir = 'results_p3/'
sample_img_path = 'train-10/Highway/image_0012.jpg'
num_kp_to_draw = 30

if __name__ == "__main__":
  if not os.path.exists(out_dir):
    os.makedirs(out_dir)
  else:
    shutil.rmtree(out_dir)
    os.makedirs(out_dir)

  sample_img = cv2.imread(problem3_dir + sample_img_path, cv2.IMREAD_GRAYSCALE)
  surf = cv2.xfeatures2d.SURF_create()

  kp, des = surf.detectAndCompute(sample_img, None)
  sample_img_with_kp = cv2.drawKeypoints(sample_img,kp[:num_kp_to_draw],None,(255,0,0),4)

  cv2.imwrite(out_dir + 'sample_img_with_kp.jpg', sample_img_with_kp)