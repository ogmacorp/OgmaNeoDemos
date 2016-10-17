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
    sudo brew cask install xquartz
    brew install homebrew/x11/freeglut
    brew install glew
    brew install webp
    brew install sfml
else
    sudo apt-get install sfml
fi
