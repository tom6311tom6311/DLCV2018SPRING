import sys
import os
import shutil
import reader
import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

VIDEO_PATH = 'data/TrimmedVideos/'
FEAT_FILE_DIR = VIDEO_PATH + 'feat/'

os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def extract_feats(is_train=True):
  mode = 'train' if is_train else 'valid'
  video_list = reader.getVideoList(VIDEO_PATH + 'label/gt_' + mode + '.csv')
  all_frames = []
  all_labels = []
  for idx in range(len(video_list['Video_index'])):
    frames = reader.readShortVideo(VIDEO_PATH + 'video/' + mode, video_list['Video_category'][idx], video_list['Video_name'][idx])
    all_frames = all_frames + frames
    all_labels = all_labels + [video_list['Action_labels'][idx]] * len(frames)
    progress(idx+1, len(video_list['Video_index']))
  all_frames = np.array(all_frames).astype(np.float64)
  all_frames = preprocess_input(all_frames)
  print(all_frames.shape)
  all_labels = np.array(all_labels)

  feat_extractor = ResNet50(weights='imagenet', input_shape=all_frames.shape[1:])
  all_feats = feat_extractor.predict(all_frames, verbose=1)
  print(all_feats.shape)
  np.savez(FEAT_FILE_DIR + mode + '.npy', feats=all_feats, labels=all_labels)

def main():
  if not os.path.exists(FEAT_FILE_DIR):
    os.makedirs(FEAT_FILE_DIR)
  extract_feats(True)

if __name__ == '__main__':
  main()
