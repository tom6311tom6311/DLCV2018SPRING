import sys
import os
import shutil
import reader
import numpy as np
from keras.models import Model
from keras.applications.resnet50 import ResNet50, preprocess_input
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.layers import Flatten

NUM_FRAMES_EACH_VIDEO = 6
VIDEO_PATH = 'data/TrimmedVideos/'
FEAT_FILE_DIR = VIDEO_PATH + 'feat' + str(NUM_FRAMES_EACH_VIDEO) + '_2048/'

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def extract_feats(is_train=True, concat_frames=True):
  mode = 'train' if is_train else 'valid'
  video_list = reader.getVideoList(VIDEO_PATH + 'label/gt_' + mode + '.csv')
  all_frames = []
  all_labels = []

  feat_extractor = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
  output = feat_extractor.get_layer('avg_pool').output
  output = Flatten()(output)
  feat_extractor = Model(feat_extractor.input, output)

  if (concat_frames):
    print('loading videos...')
    for idx in range(len(video_list['Video_index'])):
      frames = reader.readShortVideo(VIDEO_PATH + 'video/' + mode, video_list['Video_category'][idx], video_list['Video_name'][idx])
      all_frames.append(frames)
      all_labels.append(video_list['Action_labels'][idx]) #+ [video_list['Action_labels'][idx]] * len(frames)
      progress(idx+1, len(video_list['Video_index']))

    print('\nextracting features...')
    all_feats = []
    for idx, frames in enumerate(all_frames):
      selected_frame_idxs = [int((len(frames) - 1) / (NUM_FRAMES_EACH_VIDEO - 1) * i) for i in range(NUM_FRAMES_EACH_VIDEO)]
      selected_frames = preprocess_input(frames[selected_frame_idxs,:])
      selected_feats = feat_extractor.predict(selected_frames) #.reshape((1000 * 4,))
      all_feats.append(selected_feats)
      progress(idx+1, len(all_frames))
    all_feats = np.array(all_feats)
    all_labels = np.array(all_labels)
    print(all_feats.shape)
    print(all_labels.shape)
    np.savez(FEAT_FILE_DIR + mode, feats=all_feats, labels=all_labels)

  else:
    for idx in range(len(video_list['Video_index'])):
      frames = reader.readShortVideo(VIDEO_PATH + 'video/' + mode, video_list['Video_category'][idx], video_list['Video_name'][idx])
      all_frames = all_frames + frames
      all_labels = all_labels + [video_list['Action_labels'][idx]] * len(frames)
      progress(idx+1, len(video_list['Video_index']))

    all_frames = np.array(all_frames).astype(np.float64)
    all_frames = preprocess_input(all_frames)
    all_labels = np.array(all_labels)

    all_feats = feat_extractor.predict(all_frames, verbose=1)
    print(all_feats.shape)
    print(all_labels.shape)
    np.savez(FEAT_FILE_DIR + mode, feats=all_feats, labels=all_labels)

def load_feats_and_labels(is_train=True):
  mode = 'train' if is_train else 'valid'
  raw = np.load(FEAT_FILE_DIR + mode + '.npz')
  all_feats = raw['feats']
  all_labels = raw['labels']
  return all_feats, all_labels

def main():
  os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.8
  set_session(tf.Session(config=config))

  if not os.path.exists(FEAT_FILE_DIR):
    os.makedirs(FEAT_FILE_DIR)
  extract_feats(False)

if __name__ == '__main__':
  main()
