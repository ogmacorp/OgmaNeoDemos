<!---
  OgmaNeoDemos
  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeoDemos is licensed to you under the terms described
  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
--->

# OgmaNeoDemos

Examples and experiments using the [Ogma Intelligent Systems Corp](https://ogmacorp.com) [OgmaNeo](https://github.com/ogmacorp/OgmaNeo/) C++ library.

## Demos

Currently released demos include:
- Video prediction
- Anomaly detection
- Sequence recall

The Ogma Intelligent Systems Corp YouTube channel contains videos associated with certain demos, [https://www.youtube.com/ogmaai](https://www.youtube.com/ogmaai).

### Video Prediction

An OgmaNeo Predictor class is used to build an online predictive hierarchy. The video file is streamed through the hierarchy eight times. Each video frame is converted to greyscale, and an associated corrupted frame (based on previous hierarchy prediction), and are both passed into the hierarchy. Progress information is shown in the main window as this occurs as well as in the terminal.

Once the video has been presented eight times, the original video is ignored and the hierarchy is fed with predictions from the hierarchy. These predictions are then shown in the main window.

This demo uses:
[OpenCV](http://opencv.org/) (Open Source Computer Vision, version 3.x) library to load a `resources/Tesseract.wmv` video file.  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x) to create a Window and handle keyboard entry.

Note for OpenCV if you have CUDA 8 or higher (latest version), there is currently an error in OpenCV 3.1 as explained [here](http://answers.opencv.org/question/95148/cudalegacy-not-compile-nppigraphcut-missing/). Essentially, you need to edit opencv-3.1.0/modules/cudalegacy/src/graphcuts.cpp and change the line:

    #if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

to:

    #if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)  || (CUDART_VERSION >= 8000)

Makefile target for build this demo: `make Video_Prediction`

### MNIST Anomaly Detection

The [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/) is used as the dataset for this demo.

The `train-images-idx3-ubyte.gz` and `train-labels-idx1-ubyte.gz` training set files must be downloaded from http://yann.lecun.com/exdb/mnist/ and extracted into the `resources` directory.

An OgmaNeo SparseCoder is used to convert individual training images to SDRs and feed the results into an OgmaNeo Predictor online predictive hierarchy. Predictions from the hierarchy are then compared to the SparseCoder's SDR to determine if the video feed is anomalous (not predicted).

Initially only a selection of the digit 3 training images are presented to the hierarchy. This can be seen in the left half of the main window. After a short amount of time (a few minutes), the `R` key can be pressed to change from learning mode into detection mode. The right half of the window has a graph that shows peaks when a non-3 digit occurs. A red bar will also appear when the system detects an anomaly. Further, a bar at the top left of the window displays digits as the come, with the center of the bar indicating the current digit.

This demo uses:
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x).

Makefile target for this demo: `make MNIST_Anomaly_Detection`

### Sequence Recall

A sequence of ten random digits [0-9] are created. Each digit in the sequence is successively shown to a Predictor hierarchy as a bit vector. The hierarchy is repeatable shown the sequence until it is able to recall the entire sequence. Once it is recalled correctly, a new random sequence is created and this processes is repeated.

This demo only depends on the OgmaNeo library.

Makefile target for this demo: `make Sequence_Recall`

## Building

All the demos have a reliance on the main [OgmaNeo](https://github.com/ogmacorp/OgmaNeo/) C++ library, and therefore have the same requirements: A C++1x compiler, [CMake](https://cmake.org/), the [FlatBuffers](https://google.github.io/flatbuffers/) package (version 1.4.0), an OpenCL 1.2 SDK, and the Khronos Group cl2.hpp file. Refer to the main OgmaNeo [README.md](https://github.com/ogmacorp/OgmaNeo/blob/master/README.md) file for further details.

Each demo has further dependencies outlined in the previous demo description sections that are required before building the OgmaNeoDemos.

[CMake](https://cmake.org/) is used to build all the demos and discover each demo's additional dependencies. To build all the demos you can use the following (on Linux):
> cmake .  
> make

`make` on its own builds all the demos within the repository. Demos can be built individually. Refer to the previous sections for the appropriate make target name.

The `CMake/FindOgmaNeo.cmake` file tries to find an installation of the OgmaNeo library. If it does not exist on your system, a custom Makefile target will download and build the OgmaNeo library. Other CMake package discovery modules are used to find additional demo dependencies.

If you do have the OgmaNeo library _installed_ on your system (via a `make install`, or `INSTALL` project, of the OgmaNeo library), the `FindOgmaNeo.cmake` module should find the library. This discovery process can be overridden by setting the `OGMANEO_INCLUDE_DIR` and `OGMANEO_LIBRARY` options when running cmake with the OgmaNeoDemos `CMakeLists.txt` file.

## Contributions

Refer to the [CONTRIBUTING.md](https://github.com/ogmacorp/OgmaNeoDemos/blob/master/CONTRIBUTING.md) file for information on making contributions to OgmaNeoDemos.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the [OGMANEODEMOS_LICENSE.md](https://github.com/ogmacorp/OgmaNeoDemos/blob/master/OGMANEODEMOS_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/OgmaNeoDemos/blob/master/LICENSE.md) file for further information.

Contact Ogma Intelligent Systems Corp licenses@ogmacorp.com to discuss commercial use and licensing options.

OgmaNeoDemos Copyright (c) 2016 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
