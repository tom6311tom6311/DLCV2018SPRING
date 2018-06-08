#!/bin/bash

if [ $1 ]; then
  video_dir=$1
else
  video_dir="data/FullLengthVideos/videos/valid/"
fi

if [ $2 ]; then
  out_dir=$2
else
  out_dir="out/"
fi

# validation
python3 task3_valid.py 0 p3/model.hdf5 $video_dir $out_dir
