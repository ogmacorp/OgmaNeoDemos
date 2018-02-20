#!/bin/sh

# OgmaNeoDemos
# Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
#
# This copy of OgmaNeoDemos is licensed to you under the terms described
# in the OGMANEODEMOS_LICENSE.md file included in this distribution.

set -ex

#----------------------------------------
# Install the SFML 2.4.x library
cd $TRAVIS_BUILD_DIR

if [ $TRAVIS_OS_NAME == 'osx' ]; then
    sudo brew cask install xquartz
    brew install homebrew/x11/freeglut
    brew install glew
    brew install webp
    brew install sfml
else
    sudo apt-get install libsfml2

    # Certain Linux distros, such as Mint18, have an older version of SFML
    # The following can be used to install 2.4.0
    #sudo apt purge -qq libsfml-dev
    #sudo apt autoremove -qq
    #wget https://www.sfml-dev.org/files/SFML-2.4.0-linux-gcc-64-bit.tar.gz
    #tar xvf SFML-2.4.0-linux-gcc-64-bit.tar.gz
    #cd SFML-2.4.0
    #sudo cp -r . /usr/local/
fi
