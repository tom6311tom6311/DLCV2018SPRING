#!/bin/bash

if [ $1 ]; then
  video_dir=$1
else
  video_dir="data/TrimmedVideos/video/valid/"
fi

if [ $2 ]; then
  csv_path=$2
else
  csv_path="data/TrimmedVideos/label/gt_valid.csv"
fi

if [ $3 ]; then
  out_dir=$3
else
  out_dir="out/"
fi

# preprocess
python3 preprocessor.py 0 5 False $video_dir $csv_path p1/feat_valid

# validation
python3 task1_valid.py 0 p1/model.hdf5 p1/feat_valid $out_dir
