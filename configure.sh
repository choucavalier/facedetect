#!/usr/bin/env bash

# TODO : install opencv

sudo apt-get install -y megatools

# clean download
rm -rf data
mkdir data

mkdir checkpoints

# download positive data
wget -O data/lfwcrop_grey.zip 'http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip'
unzip data/lfwcrop_grey.zip -d data/
rm data/lfwcrop_grey.zip

# download negative data
megadl 'https://mega.co.nz/#!1F5WwJIS!_2-YZWSXg3ugxXaRqBaRTrz0rJmdIyydGa8ANJnKrUg' --path='data/'
tar -xf data/non_faces.tar.gz -C data/
rm data/non_faces.tar.gz
mv data/non_faces data/negative
