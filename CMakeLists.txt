# ----------------------------------------------------------------------------
#  AAOgmaNeoDemo
#  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of AOgmaNeoDemo is licensed to you under the terms described
#  in the AOGMANEODEMOS_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

# CMake version 3.1+ is required to enable C++14 features cleanly
cmake_minimum_required(VERSION 3.1)

project(AOgmaNeoDemo)

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
# Find OpenMP

find_package(OpenMP REQUIRED)

include_directories(${OpenMP_CXX_NCLUDE_DIRS})


############################################################################
# Find AOgmaNeo include and libs

find_package(AOgmaNeo)

include_directories(${AOGMANEO_INCLUDE_DIRS})


############################################################################
# Find SFML include and libs

#set(SFML_STATIC_LIBRARIES TRUE)
find_package(SFML 2.5 COMPONENTS system window graphics audio network)

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

    if(${SFML_STATIC_LIBRARIES} OR ${SFML_STATIC})
        set(SFML_BUILD_SUFFIX "-s")
        set(SFML_LIB_SUFFIX "${CMAKE_STATIC_LIBRARY_SUFFIX}")
    else()
        set(SFML_BUILD_SUFFIX "")
        set(SFML_LIB_SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}")
    endif()

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
    set(BOX2D_LIBRARIES "Box2D${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif()


############################################################################
# Find OpenCV

find_package(OpenCV COMPONENTS core imgproc highgui)

if(OpenCV_FOUND)
    message(STATUS "Found existing OpenCV in ${OpenCV_INCLUDE_DIRS}")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

############################################################################
# Find Raylib

find_package(raylib REQUIRED)

if(raylib_FOUND)
    message(STATUS "Found existing Raylib in ${raylib_INCLUDE_DIRS}")
    include_directories(${raylib_INCLUDE_DIRS})
endif()

############################################################################
# Define experiments to build

include_directories("${PROJECT_SOURCE_DIR}/demos")

#list(APPEND VIDEO_PREDICTION_SRCS "demos/Video_Prediction.cpp")
#list(APPEND VIDEO_PREDICTION_SRCS "demos/vis/Vis3D.cpp")
#list(APPEND VIDEO_PREDICTION_DEPS "SFML")
#list(APPEND VIDEO_PREDICTION_DEPS "OPENCV")
#list(APPEND VIDEO_PREDICTION_DEPS "RAYLIB")
#list(APPEND DEMO_PROJECTS_LIST "Video_Prediction")
#list(APPEND DEMO_SOURCES_LIST VIDEO_PREDICTION_SRCS)
#list(APPEND DEMO_DEPENDS_LIST VIDEO_PREDICTION_DEPS)

list(APPEND WAVY_LINE_SRCS "demos/Wavy_Line.cpp")
list(APPEND WAVY_LINE_SRCS "demos/vis/Plot.cpp")
list(APPEND WAVY_LINE_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "Wavy_Line")
list(APPEND DEMO_SOURCES_LIST WAVY_LINE_SRCS)
list(APPEND DEMO_DEPENDS_LIST WAVY_LINE_DEPS)

list(APPEND CAR_RACING_SRCS "demos/Car_Racing.cpp")
list(APPEND CAR_RACING_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "Car_Racing")
list(APPEND DEMO_SOURCES_LIST CAR_RACING_SRCS)
list(APPEND DEMO_DEPENDS_LIST CAR_RACING_DEPS)

#list(APPEND RUNNER_RUN_SRCS "demos/Runner_Run.cpp")
#list(APPEND RUNNER_RUN_SRCS "demos/runner/Runner.cpp")
#list(APPEND RUNNER_RUN_DEPS "SFML")
#list(APPEND RUNNER_RUN_DEPS "BOX2D")
#list(APPEND DEMO_PROJECTS_LIST "Runner_Run")
#list(APPEND DEMO_SOURCES_LIST RUNNER_RUN_SRCS)
#list(APPEND DEMO_DEPENDS_LIST RUNNER_RUN_DEPS)

#list(APPEND STDP_RUN_SRCS "demos/STDP.cpp")
#list(APPEND STDP_RUN_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "STDP")
#list(APPEND DEMO_SOURCES_LIST STDP_RUN_SRCS)
#list(APPEND DEMO_DEPENDS_LIST STDP_RUN_DEPS)

#list(APPEND ARDUINOVIS_SRCS "demos/ArduinoVis.cpp")
#list(APPEND ARDUINOVIS_DEPS "SFML")
#list(APPEND ARDUINOVIS_DEPS "RAYLIB")
#list(APPEND DEMO_PROJECTS_LIST "ArduinoVis")
#list(APPEND DEMO_SOURCES_LIST ARDUINOVIS_SRCS)
#list(APPEND DEMO_DEPENDS_LIST ARDUINOVIS_DEPS)

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

    if(NOT AOGMANEO_FOUND)
        add_dependencies(${DEMO_PROJECT} AOgmaNeo)
    endif()
    find_package(OpenMP REQUIRED)

    target_link_libraries(${DEMO_PROJECT} ${OpenMP_CXX_LIBRARIES})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fPIC") # This links -fopenmp

    target_link_libraries(${DEMO_PROJECT} ${AOGMANEO_LIBRARIES})

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

    # Does this demo require Raylib?
    list(FIND ${DEMO_DEPENDS} "RAYLIB" _raylib_depends_index)
    if (${_raylib_depends_index} GREATER -1)
        target_link_libraries(${DEMO_PROJECT} ${raylib_LIBRARIES})
    endif()
endforeach()
