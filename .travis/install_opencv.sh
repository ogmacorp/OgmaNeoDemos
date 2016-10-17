#!/bin/sh

# OgmaNeoDemos
# Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
#
# This copy of OgmaNeoDemos is licensed to you under the terms described
# in the OGMANEODEMOS_LICENSE.md file included in this distribution.

set -ex

#----------------------------------------
# Install the OpenCV3
cd $TRAVIS_BUILD_DIR

if [ $TRAVIS_OS_NAME == 'osx' ]; then
    brew tap homebrew/science
    brew install opencv3 --with-ffmpeg --HEAD
    brew linkapps opencv3
else
    sudo apt-get install build-essential
    sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
    # optional
    sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

    wget https://github.com/Itseez/opencv/archive/3.1.0.zip
    unzip 3.1.0.zip
    cd 3.1.0

    mkdir build
    cd build

    CC=gcc-4.8 CXX=g++-4.8 cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local ..

    make
    sudo make install
fi
