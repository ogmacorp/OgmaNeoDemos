<!---
  OgmaNeoDemos
  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeoDemos is licensed to you under the terms described
  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
--->

# OgmaNeoDemos

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby) 

Examples and experiments using the [OgmaNeo](https://github.com/ogmacorp/OgmaNeo/) C++ library.

## Demos

Currently released demos include:
- Video prediction
- Simple anomaly detection
- Level generation
- Sinusoidal prediction
- Runner
- Ball physics

The Ogma YouTube channel contains videos associated with certain demos, [https://www.youtube.com/ogmaai](https://www.youtube.com/ogmaai).

### Video Prediction

An OgmaNeo Predictor class is used to build an online predictive hierarchy. The video file is 
streamed through the hierarchy `numIter` (currently 32) times. Each video frame is split into 
three colour channels, and an associated corrupted frame (based on a blend of noise and the
previous hierarchy prediction), are passed into the hierarchy. Progress information is shown in 
the main window as this occurs as well as in the terminal, and a graph of the errors for each 
predicted frame is shown in the window.

Once the video has been presented a few times, the original video is ignored and the 
hierarchy is fed with predictions from the hierarchy. These predictions are then shown 
in the main window.

Two sample video files are provided in `resources`, and suggested settings for each are 
shown below. A `useBullFinchMovie` boolean can be toggled to determine which video to use:

| Setting | Bullfinch192.mp4 | Tesseract.wmv | Notes |
|------|------------------|---------------|-------|
| `netScale` | 192 | 128 | Input layer has `netScale * netScale` units |
| `chunkLayers` | 3 |  1 | Number of Chunk Encoder bottom layers | 
| `nLayers` | 6 | 4 | Total number of Encoder Layers |
|  - | 128x128 (x3), 64x64 (x3) | 64x64 (x4) | Units in the Encoder layers |

An optional debug window can be displayed that shows various images from within the hierarchy as it is show each frame of the video. This debug window can be enabled using the `enableDebugWindow` boolean.

This demo uses:  
[OpenCV](http://opencv.org/) (Open Source Computer Vision, version 3.x) library to load a video file.  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x) to create a Window and handle keyboard entry.

Note for OpenCV if you have CUDA 8 or higher (latest version), there is currently an error in 
OpenCV 3.1 as explained [here](http://answers.opencv.org/question/95148/cudalegacy-not-compile-nppigraphcut-missing/).
Essentially, you need to edit opencv-3.1.0/modules/cudalegacy/src/graphcuts.cpp and change the line:

    #if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

to:

    #if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)  || (CUDART_VERSION >= 8000)

Makefile target for build this demo: `make Video_Prediction`

### MNIST Anomaly Detection

The [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) is used as the dataset for this demo.

The `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` training set files must be downloaded from http://yann.lecun.com/exdb/mnist/ and extracted into the `resources` directory.

A predictor learns to predict the pixels of the digits as they move by, and when there is a large enough discrepancy between the prediction at (t) and the input at (t + 1), then an anomaly has been detected.

Initially only a selection of the digit 3 training images are presented to the hierarchy. This can be seen in the left half of the main window.
After a short amount of time (a few minutes), the `R` key can be pressed to change from learning mode into detection mode.
The right half of the window has a graph that shows peaks when a non-3 digit occurs. A red bar will also appear when the system detects an anomaly.
Further, a bar at the top left of the window displays digits as the come, with the center of the bar indicating the current digit.

This demo uses:  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x).

Makefile target for this demo: `make MNIST_Anomaly_Detection`

### Level Gen

An image of a game level is presented to the hierarchy and scrolls to the left. An infinite level is then generated through recall with noise.

This demo uses:  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x).

Makefile target for this demo: `make Level_Gen`

### Sinusoidal Prediction

Multiple sine functions are combined, plotted (red line), and presented to a hierarchy. The predicted next value is plotted to the main window (blue line).
An `ogmaneo::ScalarEncoder` is used to encode the combined sine functions for presenting to the hierarchy, and decoding predictions from the hierarchy.

The `space` key is used to pause the hierarchy and plotting. The `c` key is used to continue from a paused state.

An optional debug window can be displayed that shows various images from within the hierarchy. This debug window can be enabled using the `enableDebugWindow` boolean.

This demo uses:  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x).

Makefile target for this demo: `make Wavy_Test`

### Runner

A running quadruped robot that uses online reinforcement learning to learn to run to the left or to the right. An `ogmaneo::ScalarEncoder` is used to encode limb angles, contact sensors, and body angles into a SDR.

A swarming hierarchical reinforcement learning agent (`ogmaneo::Agent`) optimizes the distance traveled per unit time in the specified direction.

Press the `t` key to toggle speed mode (past real-time simulation) and `k` to toggle the target direction (left to right is the default).

This demo uses:  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x).  
[Box2D](http://box2d.org/) (Box2D, version 2.3.1).

Makefile target for this demo: `make Runner`

### Ball Physics

A test to see how well the hierarchy can approximate the physics of a bouncing 2D ball. A hierarchy is trained on several instances of bouncing balls with random initial starting velocities.

Once trained, the hierarchy receives a few of the starting frames of the ball to determine its trajectory, after which it loops on its own predictions to complete the physical interactions.

Press `g` to switch between train/test modes. Press `k` to show the SDRs.

This demo uses:  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x).  
[Box2D](http://box2d.org/) (Box2D, version 2.3.1).

Makefile target for this demo: `make Ball_Physics`

## Building

All the demos have a reliance on the main [OgmaNeo](https://github.com/ogmacorp/OgmaNeo/) C++ library, and therefore have the same requirements -
- A C++1x compiler,
- [CMake](https://cmake.org/),
- [FlatBuffers](https://google.github.io/flatbuffers/) package (version 1.4.0), 
- An OpenCL 1.2 SDK, and the Khronos Group cl2.hpp file.

Refer to the main OgmaNeo [README.md](https://github.com/ogmacorp/OgmaNeo/blob/master/README.md) file for further details.

Each demo has further dependencies outlined in the previous demo description sections that are required before building the OgmaNeoDemos.

[CMake](https://cmake.org/) is used to build all the demos and discover each demo's additional dependencies. To build all the demos you can use the following (on Linux):
> cmake .  
> make

`make` on its own builds all the demos within the repository. Demos can be built individually. Refer to the previous sections for the appropriate make target name.

The `CMake/FindOgmaNeo.cmake` file tries to find an installation of the OgmaNeo library. If it does not exist on your system, a custom Makefile target will download and build the OgmaNeo library.
Other CMake package discovery modules are used to find additional demo dependencies.

If you do have the OgmaNeo library _installed_ on your system (via a `make install`, or an `INSTALL` project, of the OgmaNeo library), the `FindOgmaNeo.cmake` module will try to find the library.
This discovery process can be overridden by setting the `OGMANEO_INCLUDE_DIR` and `OGMANEO_LIBRARY` options when running cmake with the OgmaNeoDemos `CMakeLists.txt` file.

## Contributions

Refer to the [CONTRIBUTING.md](https://github.com/ogmacorp/OgmaNeoDemos/blob/master/CONTRIBUTING.md) file for information on making contributions to OgmaNeoDemos.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the [OGMANEODEMOS_LICENSE.md](https://github.com/ogmacorp/OgmaNeoDemos/blob/master/OGMANEODEMOS_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/OgmaNeoDemos/blob/master/LICENSE.md) file for further information.

Contact Ogma licenses@ogmacorp.com to discuss commercial use and licensing options.

OgmaNeoDemos Copyright (c) 2016 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
