#!/bin/bash

if [ $1 ]; then
  img_dir=$1
else
  img_dir="data/validation"
fi

if [ $2 ]; then
  out_dir=$2
else
  out_dir="out"
fi

mkdir -p model
wget https://www.dropbox.com/s/n8rfs1dmgebyrsw/best.hdf5?dl=1 -O model/best.hdf5
python test.py 0 False model/best.hdf5 $img_dir $out_dir
