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
import skimage.transform
import skimage.io as io


def progress(count, total, suffix=''):
  bar_len = 60
  filled_len = int(round(bar_len * count / float(total)))
  percents = round(100.0 * count / float(total), 1)
  bar = '#' * filled_len + '-' * (bar_len - filled_len)
  sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
  sys.stdout.flush()

def extract_feats(video_path, label_path, out_path, num_frames_each_video=6, concat_frames=True, zero_padding=False):
  video_list = reader.getVideoList(label_path)
  all_frames = []
  all_labels = []

  feat_extractor = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
  output = feat_extractor.get_layer('avg_pool').output
  output = Flatten()(output)
  feat_extractor = Model(feat_extractor.input, output)

  if (concat_frames):
    print('loading videos...')
    for idx in range(len(video_list['Video_index'])):
      frames = reader.readShortVideo(video_path, video_list['Video_category'][idx], video_list['Video_name'][idx])
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
    np.savez(out_path, feats=all_feats, labels=all_labels)

  else:
    for idx in range(len(video_list['Video_index'])):
      frames = reader.readShortVideo(video_path, video_list['Video_category'][idx], video_list['Video_name'][idx])
      all_frames = all_frames + frames
      all_labels = all_labels + [video_list['Action_labels'][idx]] * len(frames)
      progress(idx+1, len(video_list['Video_index']))

    all_frames = np.array(all_frames).astype(np.float64)
    all_frames = preprocess_input(all_frames)
    all_labels = np.array(all_labels)

    all_feats = feat_extractor.predict(all_frames, verbose=1)
    print(all_feats.shape)
    print(all_labels.shape)
    np.savez(out_path, feats=all_feats, labels=all_labels)

def extract_full_length_feats(video_dir):
  feat_extractor = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
  output = feat_extractor.get_layer('avg_pool').output
  output = Flatten()(output)
  feat_extractor = Model(feat_extractor.input, output)

  print('loading video...')
  images = os.listdir(video_dir)
  images.sort()
  all_frames = [io.imread(video_dir + fn) for fn in images]
  all_frames = [skimage.transform.resize(frame, (224,224,3), mode='constant', preserve_range=True, anti_aliasing=True).astype(np.float64) for frame in all_frames]
  all_frames = np.array(all_frames).astype(np.float64)
  # with open(label_path, 'r') as label_file:
  #   all_labels = label_file.readlines()
  #   label_file.close()
  # all_labels = np.array([int(l) for l in all_labels])

  print('\nextracting features...')
  all_feats = []
  all_frames = preprocess_input(all_frames)
  all_feats = feat_extractor.predict(all_frames, verbose=1)

  print(all_feats.shape)
  return all_feats

def load_feats_and_labels(feat_path):
  raw = np.load(feat_path + '.npz')
  all_feats = raw['feats']
  all_labels = raw['labels']
  return all_feats, all_labels.astype(np.uint8)

def slice_feats(feats, num_frames_each_sample):
  sliced_feats = []
  for idx in range(feats.shape[0] - num_frames_each_sample):
    sliced_feats.append(feats[idx:idx+10, :])
  sliced_feats = np.array(sliced_feats)
  return sliced_feats

def write_predict_file(predicted, file_path, append=0):
  with open(file_path, 'w') as f:
    for p in predicted:
      f.write(str(int(p)) + '\n')
    for i in range(append):
      f.write(str(int(predicted[-1])) + '\n')
    f.close()

def main():
  os.environ["CUDA_VISIBLE_DEVICES"] = str(sys.argv[1])
  config = tf.ConfigProto()
  config.gpu_options.per_process_gpu_memory_fraction = 0.8
  set_session(tf.Session(config=config))

  NUM_FRAMES_EACH_VIDEO = int(sys.argv[2])
  ZERO_PADDING = True if str(sys.argv[3]) == 'True' else False
  VIDEO_DIR = str(sys.argv[4]) if str(sys.argv[4])[-1] == '/' else str(sys.argv[4]) + '/'
  LABEL_PATH = str(sys.argv[5])
  OUT_PATH = str(sys.argv[6])

  print(NUM_FRAMES_EACH_VIDEO)
  print(ZERO_PADDING)

  extract_feats(VIDEO_DIR, LABEL_PATH, OUT_PATH, NUM_FRAMES_EACH_VIDEO)

if __name__ == '__main__':
  main()
