#!/bin/bash

if [ $1 ]; then
  in_dir=$1
else
  img_dir="data/"
fi

if [ $2 ]; then
  out_dir=$2
else
  out_dir="out"
fi

mkdir -p model
wget https://www.dropbox.com/s/3z5qujt6215tt93/vae_30.hdf5?dl=1 -O model/vae.hdf5
wget https://www.dropbox.com/s/55mxr3mvosakvz4/dcgan_24_gen?dl=1 -O model/dcgan.hdf5
wget https://www.dropbox.com/s/0w3ripyhiu1kog0/acgan_34_gen?dl=1 -O model/acgan.hdf5
cd 1_vae
python fig1_3.py 0 ../model/vae.hdf5 $in_dir $out_dir
python fig1_4.py 0 ../model/vae.hdf5 $in_dir $out_dir
cd ../2_gan
python fig2_3.py 0 ../model/dcgan.hdf5 $in_dir $out_dir
python fig3_3.py 0 ../model/acgan.hdf5 $in_dir $out_dir
