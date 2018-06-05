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

VIDEO_PATH = 'data/TrimmedVideos/'

def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def extract_feats(is_train=True, num_frames_each_video=6, concat_frames=True, zero_padding=False, feat_file_dir=VIDEO_PATH + 'feat/'):
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
      if (zero_padding):
        if (frames.shape[0] < num_frames_each_video):
          frames = np.concatenate([frames, np.zeros((num_frames_each_video - frames.shape[0],) + frames.shape[1:])])
        selected_frame_idxs = list(range(num_frames_each_video))
      else:
        selected_frame_idxs = [int((len(frames) - 1) / (num_frames_each_video - 1) * i) for i in range(num_frames_each_video)]
      selected_frames = preprocess_input(frames[selected_frame_idxs,:])
      selected_feats = feat_extractor.predict(selected_frames) #.reshape((1000 * 4,))
      all_feats.append(selected_feats)
      progress(idx+1, len(all_frames))
    all_feats = np.array(all_feats)
    all_labels = np.array(all_labels)
    print(all_feats.shape)
    print(all_labels.shape)
    np.savez(feat_file_dir + mode, feats=all_feats, labels=all_labels)

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
    np.savez(feat_file_dir + mode, feats=all_feats, labels=all_labels)

def load_feats_and_labels(is_train=True, feat_file_dir=VIDEO_PATH + 'feat/'):
  mode = 'train' if is_train else 'valid'
  raw = np.load(feat_file_dir + mode + '.npz')
  all_feats = raw['feats']
  all_labels = raw['labels']
  return all_feats, all_labels

def main():
  os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.8
  set_session(tf.Session(config=config))

  NUM_FRAMES_EACH_VIDEO = int(sys.argv[2])
  IS_TRAIN = True if str(sys.argv[3]) == 'True' else False
  ZERO_PADDING = True if str(sys.argv[4]) == 'True' else False
  FEAT_FILE_DIR = VIDEO_PATH + 'feat' + str(NUM_FRAMES_EACH_VIDEO) + '_2048_' + str(ZERO_PADDING) + '/'

  print(NUM_FRAMES_EACH_VIDEO)
  print(IS_TRAIN)
  print(ZERO_PADDING)

  if not os.path.exists(FEAT_FILE_DIR):
    os.makedirs(FEAT_FILE_DIR)

  extract_feats(IS_TRAIN, num_frames_each_video=NUM_FRAMES_EACH_VIDEO, zero_padding=ZERO_PADDING, feat_file_dir=FEAT_FILE_DIR)

if __name__ == '__main__':
  main()
