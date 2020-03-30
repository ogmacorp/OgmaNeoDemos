<!---
  OgmaNeoDemos
  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.

  This copy of OgmaNeoDemos is licensed to you under the terms described
  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
--->

# OgmaNeoDemos

[![Join the chat at https://gitter.im/ogmaneo/Lobby](https://img.shields.io/gitter/room/nwjs/nw.js.svg)](https://gitter.im/ogmaneo/Lobby) 

Examples and experiments using the [OgmaNeo2](https://github.com/ogmacorp/OgmaNeo2/) C++ library.

## Demos

Current demos are:
- Video prediction
- Wavy Line
- Ball Physics
- Car Racing

### Video Prediction

An OgmaNeo Predictor class is used to build an online predictive hierarchy. The video file is 
streamed through the hierarchy `numIter` times. Progress information is shown in 
the main window as this occurs as well as in the terminal, and a graph of the errors for each 
predicted frame is shown in the window.

Once the video has been presented a few times, the original video is ignored and the 
hierarchy is fed with predictions from itself. These predictions are then shown 
in the main window.

This demo uses:  
[OpenCV](http://opencv.org/) (Open Source Computer Vision, version 3.x) library to load a video file.  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x) to create a Window and handle keyboard entry.

Makefile target for build this demo: `make Video_Prediction`

### Wavy Line

Multiple sine functions are combined, plotted (red line), and presented to a hierarchy. The predicted next value is plotted to the main window (blue line).
The values are encoded into the hierarchy through a simple binning process.

The `space` key is used to toggle pausing the hierarchy and plotting. The right arrow key can also temporarily advance the prediction.

Press and hole the `p` key to run the hierarchy on its own predictions.

This demo uses:  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x).

Makefile target for this demo: `make Wavy_Line`

### Ball Physics

A test to see how well the hierarchy can approximate the physics of a bouncing 2D ball. A hierarchy is trained on several instances of bouncing balls with random initial starting velocities.

Once trained, the hierarchy receives a few of the starting frames of the ball to determine its trajectory, after which it loops on its own predictions to complete the physical interactions.

Press `g` to switch between train/test modes. Press `k` to show the SDRs.

This demo uses:  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x).  
[Box2D](http://box2d.org/) (Box2D, version 2.3.1).

Makefile target for this demo: `make Ball_Physics`

### Car Racing

A race car must navigate a track without hitting the walls. This task requires fine motor control based on sensor "wiskers" the car has.

The `t` key is used to toggle "speed mode" (or "time warp").
Pressed the `c` key to view the checkpoints on the track.

This demo uses:  
[SFML](http://www.sfml-dev.org/) (Simple and Fast Multimedia Library, version 2.4.x).

Makefile target for this demo: `make Car_Racing`

## Building

All the demos have a reliance on the main [OgmaNeo2](https://github.com/ogmacorp/OgmaNeo2/) C++ library, and therefore have the same requirements -
- A C++1x compiler,
- [CMake](https://cmake.org/),
- OpenMP

Refer to the main OgmaNeo2 [README.md](https://github.com/ogmacorp/OgmaNeo2/blob/master/README.md) file for further details.

Each demo has further dependencies outlined in the previous demo description sections that are required before building the OgmaNeoDemos.

[CMake](https://cmake.org/) is used to build all the demos and discover each demo's additional dependencies. To build all the demos you can use the following (on Linux):
> cmake .  
> make

`make` on its own builds all the demos within the repository. Demos can be built individually. Refer to the previous sections for the appropriate make target name.

The `CMake/FindOgmaNeo.cmake` file tries to find an installation of the OgmaNeo library. If it does not exist on your system, a custom Makefile target will download and build the OgmaNeo library.
Other CMake package discovery modules are used to find additional demo dependencies.

If you do have the OgmaNeo2 library _installed_ on your system (via a `make install`, or an `INSTALL` project, of the OgmaNeo2 library), the `FindOgmaNeo.cmake` module will try to find the library.
This discovery process can be overridden by setting the `OGMANEO_INCLUDE_DIR` and `OGMANEO_LIBRARY` options when running cmake with the OgmaNeoDemos `CMakeLists.txt` file.

## Contributions

Refer to the [CONTRIBUTING.md](https://github.com/ogmacorp/OgmaNeoDemos/blob/master/CONTRIBUTING.md) file for information on making contributions to OgmaNeoDemos.

## License and Copyright

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />The work in this repository is licensed under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>. See the [OGMANEODEMOS_LICENSE.md](https://github.com/ogmacorp/OgmaNeoDemos/blob/master/OGMANEODEMOS_LICENSE.md) and [LICENSE.md](https://github.com/ogmacorp/OgmaNeoDemos/blob/master/LICENSE.md) file for further information.

Contact Ogma licenses@ogmacorp.com to discuss commercial use and licensing options.

OgmaNeoDemos Copyright (c) 2016-2020 [Ogma Intelligent Systems Corp](https://ogmacorp.com). All rights reserved.
