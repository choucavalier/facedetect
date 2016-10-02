#!/usr/bin/env bash

if [[ "$EUID" -ne 0 ]]; then echo "Please run as root"; exit; fi

# TODO : install opencv

# clean download
rm -rf data
mkdir data

# download positive data
wget -O data/lfwcrop_grey.zip 'http://conradsanderson.id.au/lfwcrop/lfwcrop_grey.zip'
unzip data/lfwcrop_grey.zip -d data/
rm data/lfwcrop_grey.zip
mv data/lfwcrop_grey/faces data/positive
rm -rf data/lfwcrop_grey

# download negative data
sudo apt-get install -y megatools
megadl 'https://mega.co.nz/#!1F5WwJIS!_2-YZWSXg3ugxXaRqBaRTrz0rJmdIyydGa8ANJnKrUg' --path='data/'
tar -xf data/non_faces.tar.gz -C data/
rm data/non_faces.tar.gz
mv data/non_faces data/negative
