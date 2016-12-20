# ----------------------------------------------------------------------------
#  OgmaNeoDemos
#  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeoDemos is licensed to you under the terms described
#  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# CMake version 3.1+ is required to enable C++14 features cleanly
cmake_minimum_required(VERSION 3.1)

project(OgmaNeoDemos)

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BITNESS 64)
else()
    set(BITNESS 32)
endif()
message(STATUS "Bitness: ${BITNESS}")


include(ExternalProject)

set(3RDPARTY_PATH ${CMAKE_CURRENT_BINARY_DIR}/3rdparty)
include_directories("${3RDPARTY_PATH}/include")
link_directories("${3RDPARTY_PATH}/lib")


############################################################################
# Find OpenCL include and libs

#find_package(OpenCL REQUIRED)
#include_directories(${OpenCL_INCLUDE_DIRS})

find_path(OPENCL_INCLUDE_DIRS 
    NAMES OpenCL/cl.h CL/cl.h
    HINTS
        $ENV{AMDAPPSDKROOT}/include/
        $ENV{INTELOCLSDKROOT}/include/
        $ENV{CUDA_PATH}/include/
        $ENV{OPENCL_ROOT}/include/
    PATHS
        /usr/include /usr/local/include/
)

if(BITNESS EQUAL 64)
    find_library(OPENCL_LIBRARIES
        NAMES OpenCL
        HINTS
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{INTELOCLSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            $ENV{OPENCL_ROOT}/lib
        PATH_SUFFIXES x86_64 x64
        PATHS /usr/lib64 /usr/lib /usr/local/lib
    )
elseif(BITNESS EQUAL 32)
    find_library(OPENCL_LIBRARIES
        NAMES OpenCL
        HINTS
            $ENV{AMDAPPSDKROOT}/lib
            $ENV{INTELOCLSDKROOT}/lib
            $ENV{CUDA_PATH}/lib
            $ENV{OPENCL_ROOT}/lib
        PATH_SUFFIXES x86 Win32
        PATHS /usr/lib32 /usr/lib /usr/local/lib
    )
endif()

# Fall-back to find a POCL install
if(OPENCL_LIBRARIES STREQUAL "")
    find_library(OPENCL_LIBRARIES
        NAMES pocl
        HINTS $HOME/.local/pocl/lib /usr/lib64 /usr/lib /usr/local/lib
    )
endif()

if(OPENCL_INCLUDE_DIRS STREQUAL "" OR OPENCL_LIBRARIES STREQUAL "")
    message(FATAL_ERROR "Could not locate OpenCL include & libs")
else()
    message(STATUS "Found existing OpenCL in ${OPENCL_INCLUDE_DIRS}")
    message(STATUS "Found existing OpenCL lib ${OPENCL_LIBRARIES}")
endif()

include_directories(${OPENCL_INCLUDE_DIRS})


############################################################################
# Find OgmaNeo include and libs

find_package(OgmaNeo)

if(OGMANEO_FOUND)
    message(STATUS "Found existing OgmaNeo in ${OGMANEO_INCLUDE_DIRS}")
else()
    message(STATUS "-- OgmaNeo will be downloaded and built from source")

    include(ExternalProject)
    ExternalProject_Add(OgmaNeo
        PREFIX OgmaNeo
        UPDATE_COMMAND ""
        #TODO Change to OgmaNeo repo for release
        #URL "https://github.com/ogmacorp/OgmaNeoInternal/archive/v0.1.0.zip"
        GIT_REPOSITORY "https://github.com/OgmaCorp/OgmaNeoInternal.git"
        CMAKE_ARGS
            "-DBUILD_SHARED_LIBS=OFF"
            "-DCMAKE_INSTALL_PREFIX=${3RDPARTY_PATH}"
    )
    set(OGMANEO_INCLUDE_DIRS "${3RDPARTY_PATH}/include/ogmaneo")
    set(OGMANEO_LIBRARY "${3RDPARTY_PATH}/lib/${CMAKE_STATIC_LIBRARY_PREFIX}OgmaNeo${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()

include_directories(${OGMANEO_INCLUDE_DIRS})


############################################################################
# Find SFML include and libs

#set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML COMPONENTS system window graphics audio)

#if(APPLE)
#    find_package(boost COMPONENTS filesystem-mt REQUIRED)
#    include_directories(${boost_INCLUDE_DIRS})
#endif()

if(SFML_FOUND)
    #message(STATUS "Found existing SFML in ${SFML_INCLUDE_DIR}")
    message(STATUS "SFML dependencies: ${SFML_DEPENDENCIES}")
    message(STATUS "SFML libraries: ${SFML_LIBRARIES}")
    include_directories(${SFML_INCLUDE_DIR})
else()
    message(STATUS "-- SFML will be downloaded and built from source")
    ExternalProject_Add(
        SFML
        PREFIX SFML
        URL "https://github.com/SFML/SFML/archive/2.4.0.zip"
        CMAKE_ARGS
            "-DCMAKE_INSTALL_PREFIX=${3RDPARTY_PATH}"
            "-DBUILD_SHARED_LIBS=${BUILD_SHARED_LIBS}"
            #"-DSFML_STATIC_LIBRARIES=${SFML_STATIC_LIBRARIES}"
            "-DSFML_BUILD_EXAMPLES=OFF"
            "-DSFML_BUILD_DOC=OFF"
    )
    set(SFML_INCLUDE_DIR "${3RDPARTY_PATH}/include")

    if(${SFML_STATIC_LIBRARIES})
        set(SFML_BUILD_SUFFIX "-s")
    else()
        set(SFML_BUILD_SUFFIX "")
    endif()
    set(SFML_LIB_SUFFIX "${CMAKE_STATIC_LIBRARY_SUFFIX}")

    set(SFML_SYSTEM_LIBRARY debug "sfml-system${SFML_BUILD_SUFFIX}-d${SFML_LIB_SUFFIX}" optimized "sfml-system${SFML_BUILD_SUFFIX}${SFML_LIB_SUFFIX}")
    set(SFML_WINDOW_LIBRARY debug "sfml-window${SFML_BUILD_SUFFIX}-d${SFML_LIB_SUFFIX}" optimized "sfml-window${SFML_BUILD_SUFFIX}${SFML_LIB_SUFFIX}")
    set(SFML_GRAPHICS_LIBRARY debug "sfml-graphics${SFML_BUILD_SUFFIX}-d${SFML_LIB_SUFFIX}" optimized "sfml-graphics${SFML_BUILD_SUFFIX}${SFML_LIB_SUFFIX}")
    set(SFML_AUDIO_LIBRARY debug "sfml-audio${SFML_BUILD_SUFFIX}-d${SFML_LIB_SUFFIX}" optimized "sfml-audio${SFML_BUILD_SUFFIX}${SFML_LIB_SUFFIX}")
    set(SFML_NETWORK_LIBRARY debug "sfml-network${SFML_BUILD_SUFFIX}-d${SFML_LIB_SUFFIX}" optimized "sfml-network${SFML_BUILD_SUFFIX}${SFML_LIB_SUFFIX}")

    if(NOT ${SFML_STATIC_LIBRARIES})
        set(SFML_MAIN_LIBRARY debug "sfml-main${SFML_BUILD_SUFFIX}-d${SFML_LIB_SUFFIX}" optimized "sfml-main${SFML_BUILD_SUFFIX}${SFML_LIB_SUFFIX}")
    else()
        set(SFML_MAIN_LIBRARY "")
    endif()

    set(SFML_LIBRARIES ${SFML_MAIN_LIBRARY} ${SFML_WINDOW_LIBRARY} ${SFML_GRAPHICS_LIBRARY} ${SFML_AUDIO_LIBRARY} ${SFML_NETWORK_LIBRARY} ${SFML_SYSTEM_LIBRARY})

#    if(MSVC)
#        set(SFML_DEPENDENCIES flac.lib freetype.lib jpeg.lib ogg.lib openal32.lib vorbis.lib vorbisenc.lib vorbisfile.lib)
#        set(SFML_DEPENDENCIES ${SFML_DEPENDENCIES} winmm.lib ws2_32.lib opengl32.lib gdi32.lib)
#    else()
        #set(SFML_DEPENDENCIES pthread rt GL freetype jpeg X11 xcb X11-xcb xcb-randr xcb-image udev openal ogg vorbis vorbisfile)
#    endif()
endif()


############################################################################
# Find Box2D include and libs

find_package(BOX2D)

if(BOX2D_FOUND)
    include_directories(${BOX2D_INCLUDE_DIRS})
    message(STATUS "Found existing Box2D in ${BOX2D_INCLUDE_DIRS}")
else()
    message(STATUS "-- Box2D will be downloaded and built from source")
    ExternalProject_Add(
        Box2D
        PREFIX Box2D
        URL "https://github.com/erincatto/Box2D/archive/v2.3.0.zip"

        # Box2D's source isn't in the top level directory so add a dummy file to set cmake right
        PATCH_COMMAND ${CMAKE_COMMAND} -E echo ADD_SUBDIRECTORY(Box2D) > <SOURCE_DIR>/CMakeLists.txt

        CMAKE_ARGS
            "-DCMAKE_INSTALL_PREFIX=${3RDPARTY_PATH}"
            "-DBOX2D_INSTALL=ON"
            "-DBOX2D_BUILD_EXAMPLES=OFF"
    )
    set(BOX2D_INCLUDE_DIRS "${3RDPARTY_PATH}/include")
    set(BOX2D_LIBRARIES "box2d${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()


############################################################################
# Find OpenCV

find_package(OpenCV HINTS /usr/local/opt/opencv3)

if(OpenCV_FOUND)
    message(STATUS "Found existing OpenCV in ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()


############################################################################
# Find ALE

if(USE_ALE)
    if(MSVC)
        find_path(ALESDKROOT "Arcade-Learning-Environment")
    endif()

    find_path(ALE_INCLUDES 
        NAMES ale_interface.hpp
        HINTS /usr/local/include/ale ${ALESDKROOT}/src/
    )
    find_library(ALE_LIBRARIES
        NAMES ale
        HINTS /usr/local/lib/
        PATH_SUFFIXES ${PLATFORM}${BITNESS} ${BITNESS_SUFFIX}
    )

    message(STATUS "Found ALE in ${ALE_INCLUDES}")
endif()


############################################################################
# Define experiments to build

include_directories("${PROJECT_SOURCE_DIR}/demos")

list(APPEND WAVY_TEST_SRCS "demos/Wavy_Test.cpp")
list(APPEND WAVY_TEST_SRCS "demos/vis/Plot.cpp")
list(APPEND WAVY_TEST_SRCS "demos/vis/Plot.h")
list(APPEND WAVY_TEST_SRCS "demos/vis/DebugWindow.cpp")
list(APPEND WAVY_TEST_SRCS "demos/vis/DebugWindow.h")
list(APPEND WAVY_TEST_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "Wavy_Test")
list(APPEND DEMO_SOURCES_LIST WAVY_TEST_SRCS)
list(APPEND DEMO_DEPENDS_LIST WAVY_TEST_DEPS)

list(APPEND FB_PARSER_SRCS "demos/FB_Parser.cpp")
list(APPEND FB_PARSER_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "FB_Parser")
list(APPEND DEMO_SOURCES_LIST FB_PARSER_SRCS)
list(APPEND DEMO_DEPENDS_LIST FB_PARSER_DEPS)

list(APPEND LEVEL_GEN_SRCS "demos/Level_Gen.cpp")
list(APPEND LEVEL_GEN_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "Level_Gen")
list(APPEND DEMO_SOURCES_LIST LEVEL_GEN_SRCS)
list(APPEND DEMO_DEPENDS_LIST LEVEL_GEN_DEPS)

list(APPEND VIDEO_PREDICTION_SRCS "demos/Video_Prediction.cpp")
list(APPEND VIDEO_PREDICTION_SRCS "demos/vis/DebugWindow.h")
list(APPEND VIDEO_PREDICTION_SRCS "demos/vis/DebugWindow.cpp")
list(APPEND VIDEO_PREDICTION_DEPS "SFML")
list(APPEND VIDEO_PREDICTION_DEPS "OPENCV")
list(APPEND DEMO_PROJECTS_LIST "Video_Prediction")
list(APPEND DEMO_SOURCES_LIST VIDEO_PREDICTION_SRCS)
list(APPEND DEMO_DEPENDS_LIST VIDEO_PREDICTION_DEPS)

list(APPEND MNIST_ANOMALY_SRCS "demos/MNIST_Anomaly_Detection.cpp")
list(APPEND MNIST_ANOMALY_SRCS "demos/vis/Plot.cpp")
list(APPEND MNIST_ANOMALY_SRCS "demos/vis/Plot.h")
list(APPEND MNIST_ANOMALY_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "MNIST_Anomaly_Detection")
list(APPEND DEMO_SOURCES_LIST MNIST_ANOMALY_SRCS)
list(APPEND DEMO_DEPENDS_LIST MNIST_ANOMALY_DEPS)

list(APPEND BALL_PHYSICS_SRCS "demos/Ball_Physics.cpp")
list(APPEND BALL_PHYSICS_DEPS "SFML")
list(APPEND BALL_PHYSICS_DEPS "BOX2D")
list(APPEND DEMO_PROJECTS_LIST "Ball_Physics")
list(APPEND DEMO_SOURCES_LIST BALL_PHYSICS_SRCS)
list(APPEND DEMO_DEPENDS_LIST BALL_PHYSICS_DEPS)

list(APPEND RUNNER_SRCS "demos/RunnerMain.cpp")
list(APPEND RUNNER_SRCS "demos/runner/Runner.h")
list(APPEND RUNNER_SRCS "demos/runner/Runner.cpp")
list(APPEND RUNNER_DEPS "SFML")
list(APPEND RUNNER_DEPS "BOX2D")
list(APPEND DEMO_PROJECTS_LIST "Runner")
list(APPEND DEMO_SOURCES_LIST RUNNER_SRCS)
list(APPEND DEMO_DEPENDS_LIST RUNNER_DEPS)

list(LENGTH DEMO_PROJECTS_LIST num_demos)
message(STATUS "Demos to build: ${DEMO_PROJECTS_LIST}")

math(EXPR num_demos "${num_demos}-1")
foreach(i RANGE ${num_demos})
    list(GET DEMO_PROJECTS_LIST ${i} DEMO_PROJECT)
    list(GET DEMO_SOURCES_LIST ${i} DEMO_SOURCES)
    list(GET DEMO_DEPENDS_LIST ${i} DEMO_DEPENDS)

    math(EXPR demo_num "${i}+1")
    message(STATUS "Demo #${demo_num} project: ${DEMO_PROJECT}")
    message(STATUS "Demo #${demo_num} sources: ${${DEMO_SOURCES}}")
    message(STATUS "Demo #${demo_num} depends: ${${DEMO_DEPENDS}}")

    add_executable(${DEMO_PROJECT} ${${DEMO_SOURCES}})
    set_property(TARGET ${DEMO_PROJECT} PROPERTY CXX_STANDARD 14)
    set_property(TARGET ${DEMO_PROJECT} PROPERTY CXX_STANDARD_REQUIRED ON)

    if(NOT OGMANEO_FOUND)
        add_dependencies(${DEMO_PROJECT} OgmaNeo)
    endif()
    target_link_libraries(${DEMO_PROJECT} ${OGMANEO_LIBRARY})
    target_link_libraries(${DEMO_PROJECT} ${OPENCL_LIBRARIES})

#    if(APPLE)
#        target_link_libraries(${DEMO_PROJECT} ${boost_LIBRARIES})
#    endif()

    # Does this demo require SFML?
    list(FIND ${DEMO_DEPENDS} "SFML" _sfml_depends_index)
    if (${_sfml_depends_index} GREATER -1)
        if(NOT SFML_FOUND)
            add_dependencies(${DEMO_PROJECT} SFML)
        endif()
        target_link_libraries(${DEMO_PROJECT} ${SFML_DEPENDENCIES})
        target_link_libraries(${DEMO_PROJECT} ${SFML_LIBRARIES})
    endif()

    # Does this demo require Box2D?
    list(FIND ${DEMO_DEPENDS} "BOX2D" _box2d_depends_index)
    if (${_box2d_depends_index} GREATER -1)
        if(NOT BOX2D_FOUND)
            add_dependencies(${DEMO_PROJECT} Box2D)
        endif()
        target_link_libraries(${DEMO_PROJECT} ${BOX2D_LIBRARIES})
    endif()

    # Does this demo require OpenCV?
    list(FIND ${DEMO_DEPENDS} "OPENCV" _opencv_depends_index)
    if (${_opencv_depends_index} GREATER -1)
        target_link_libraries(${DEMO_PROJECT} ${OpenCV_LIBS})
    endif()

    # Does this demo require ALE?
    list(FIND DEMO_DEPENDS "ALE" _ale_depends_index)
    if (${_ale_depends_index} GREATER -1)
        include_directories(${DEMO_PROJECT} ${ALE_INCLUDES})
        target_link_libraries(${DEMO_PROJECT} ${ALE_LIBRARIES})
        if(MSVC)
            file(GLOB_RECURSE ALE_DEPENDENCY_LIBS
                    ${ALESDKROOT}/visual_studio/dependencies/lib_win_x64/*.lib)

            target_link_libraries(${DEMO_PROJECT} ${ALE_DEPENDENCY_LIBS})
        endif()
    endif()
endforeach()
