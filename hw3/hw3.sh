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
wget https://www.dropbox.com/s/x1j3wabnp9b3qt4/baseline.hdf5?dl=1 -O model/baseline.hdf5
python3 test.py 0 True model/baseline.hdf5 $img_dir $out_dir