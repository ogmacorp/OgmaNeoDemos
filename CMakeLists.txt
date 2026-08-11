# ----------------------------------------------------------------------------
#  AAOgmaNeoDemo
#  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of AOgmaNeoDemo is licensed to you under the terms described
#  in the AOGMANEODEMOS_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 4.1)

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
# Find ultrasparse include and libs

find_package(Ultrasparse)

include_directories(${ULTRASPARSE_INCLUDE_DIRS})

############################################################################
# Find OgmaNeo2 include and libs

#find_package(Torch)
#
#include_directories(${TORCH_INCLUDE_DIRS})


############################################################################
# Find SFML include and libs

include(FetchContent)
FetchContent_Declare(SFML
    GIT_REPOSITORY https://github.com/SFML/SFML.git
    GIT_TAG 3.0.0
    GIT_SHALLOW ON
    EXCLUDE_FROM_ALL
    SYSTEM)
FetchContent_MakeAvailable(SFML)


############################################################################
# Find Box2D include and libs

include_directories("/usr/include/box2d/")


############################################################################
# Find OpenCV

find_package(OpenCV COMPONENTS core imgproc highgui features2d)

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

#list(APPEND CROPTOOL_SRCS "demos/CropTool.cpp")
#list(APPEND CROPTOOL_DEPS "SFML")
#list(APPEND CROPTOOL_DEPS "OPENCV")
#list(APPEND DEMO_PROJECTS_LIST "CropTool")
#list(APPEND DEMO_SOURCES_LIST CROPTOOL_SRCS)
#list(APPEND DEMO_DEPENDS_LIST CROPTOOL_DEPS)

#list(APPEND TRACKSOM_SRCS "demos/TrackSOM.cpp")
#list(APPEND TRACKSOM_DEPS "SFML")
#list(APPEND TRACKSOM_DEPS "OPENCV")
#list(APPEND DEMO_PROJECTS_LIST "TrackSOM")
#list(APPEND DEMO_SOURCES_LIST TRACKSOM_SRCS)
#list(APPEND DEMO_DEPENDS_LIST TRACKSOM_DEPS)

#list(APPEND LOOP_MAPPER_SRCS "demos/Loop_Mapper.cpp")
#list(APPEND LOOP_MAPPER_DEPS "SFML")
#list(APPEND LOOP_MAPPER_DEPS "OPENCV")
#list(APPEND DEMO_PROJECTS_LIST "Loop_Mapper")
#list(APPEND DEMO_SOURCES_LIST LOOP_MAPPER_SRCS)
#list(APPEND DEMO_DEPENDS_LIST LOOP_MAPPER_DEPS)

#list(APPEND LOOP_COUNTER_SRCS "demos/Loop_Counter.cpp")
#list(APPEND LOOP_COUNTER_DEPS "SFML")
#list(APPEND LOOP_COUNTER_DEPS "OPENCV")
#list(APPEND DEMO_PROJECTS_LIST "Loop_Counter")
#list(APPEND DEMO_SOURCES_LIST LOOP_COUNTER_SRCS)
#list(APPEND DEMO_DEPENDS_LIST LOOP_COUNTER_DEPS)

list(APPEND CAR_MOCAP_DRAW_SRCS "demos/Car_Mocap_Draw.cpp")
list(APPEND CAR_MOCAP_DRAW_DEPS "SFML")
list(APPEND CAR_MOCAP_DRAW_DEPS "OPENCV")
list(APPEND DEMO_PROJECTS_LIST "Car_Mocap_Draw")
list(APPEND DEMO_SOURCES_LIST CAR_MOCAP_DRAW_SRCS)
list(APPEND DEMO_DEPENDS_LIST CAR_MOCAP_DRAW_DEPS)

#list(APPEND CAR_TRACKER_SRCS "demos/Car_Tracker.cpp")
#list(APPEND CAR_TRACKER_DEPS "SFML")
#list(APPEND CAR_TRACKER_DEPS "OPENCV")
#list(APPEND DEMO_PROJECTS_LIST "Car_Tracker")
#list(APPEND DEMO_SOURCES_LIST CAR_TRACKER_SRCS)
#list(APPEND DEMO_DEPENDS_LIST CAR_TRACKER_DEPS)

#list(APPEND SDC_CONTROLLER_TEST_SRCS "demos/SDC_Controller_Test.cpp")
#list(APPEND SDC_CONTROLLER_TEST_SRCS "demos/sdc/sdc_controller.cpp")
#list(APPEND SDC_CONTROLLER_TEST_SRCS "demos/sdc/config.cpp")
#list(APPEND SDC_CONTROLLER_TEST_SRCS "demos/sdc/helpers.cpp")
#list(APPEND SDC_CONTROLLER_TEST_DEPS "SFML")
#list(APPEND SDC_CONTROLLER_TEST_DEPS "OPENCV")
#list(APPEND DEMO_PROJECTS_LIST "SDC_Controller_Test")
#list(APPEND DEMO_SOURCES_LIST SDC_CONTROLLER_TEST_SRCS)
#list(APPEND DEMO_DEPENDS_LIST SDC_CONTROLLER_TEST_DEPS)

#list(APPEND SINGLE_LAP_MAPPER_SRCS "demos/Single_Lap_Mapper.cpp")
#list(APPEND SINGLE_LAP_MAPPER_DEPS "SFML")
#list(APPEND SINGLE_LAP_MAPPER_DEPS "OPENCV")
#list(APPEND DEMO_PROJECTS_LIST "Single_Lap_Mapper")
#list(APPEND DEMO_SOURCES_LIST SINGLE_LAP_MAPPER_SRCS)
#list(APPEND DEMO_DEPENDS_LIST SINGLE_LAP_MAPPER_DEPS)

#list(APPEND DONKEY_PLAYBACK_SRCS "demos/Donkey_Playback.cpp")
#list(APPEND DONKEY_PLAYBACK_SRCS "demos/vis/Vis3D.cpp")
#list(APPEND DONKEY_PLAYBACK_DEPS "SFML")
#list(APPEND DONKEY_PLAYBACK_DEPS "OPENCV")
#list(APPEND DONKEY_PLAYBACK_DEPS "RAYLIB")
#list(APPEND DEMO_PROJECTS_LIST "Donkey_Playback")
#list(APPEND DEMO_SOURCES_LIST DONKEY_PLAYBACK_SRCS)
#list(APPEND DEMO_DEPENDS_LIST DONKEY_PLAYBACK_DEPS)

list(APPEND FP_PLAYBACK_SRCS "demos/FP_Playback.cpp")
list(APPEND FP_PLAYBACK_DEPS "SFML")
list(APPEND FP_PLAYBACK_DEPS "OPENCV")
list(APPEND DEMO_PROJECTS_LIST "FP_Playback")
list(APPEND DEMO_SOURCES_LIST FP_PLAYBACK_SRCS)
list(APPEND DEMO_DEPENDS_LIST FP_PLAYBACK_DEPS)

#list(APPEND IMAGE_ENCODER_TEST_SRCS "demos/Image_Encoder_Test.cpp")
#list(APPEND IMAGE_ENCODER_TEST_DEPS "SFML")
#list(APPEND IMAGE_ENCODER_TEST_DEPS "BOX2D")
#list(APPEND DEMO_PROJECTS_LIST "Image_Encoder_Test")
#list(APPEND DEMO_SOURCES_LIST IMAGE_ENCODER_TEST_SRCS)
#list(APPEND DEMO_DEPENDS_LIST IMAGE_ENCODER_TEST_DEPS)

#list(APPEND WAVY_LINE_SRCS "demos/Wavy_Line.cpp")
#list(APPEND WAVY_LINE_SRCS "demos/vis/Plot.cpp")
#list(APPEND WAVY_LINE_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Wavy_Line")
#list(APPEND DEMO_SOURCES_LIST WAVY_LINE_SRCS)
#list(APPEND DEMO_DEPENDS_LIST WAVY_LINE_DEPS)

#list(APPEND WAVY_CLASSIFY_SRCS "demos/Wavy_Classify.cpp")
#list(APPEND WAVY_CLASSIFY_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Wavy_Classify")
#list(APPEND DEMO_SOURCES_LIST WAVY_CLASSIFY_SRCS)
#list(APPEND DEMO_DEPENDS_LIST WAVY_CLASSIFY_DEPS)

#list(APPEND SWARM_TEST_SRCS "demos/Swarm_Test.cpp")
#list(APPEND DEMO_PROJECTS_LIST "Swarm_Test")
#list(APPEND DEMO_SOURCES_LIST SWARM_TEST_SRCS)
#list(APPEND DEMO_DEPENDS_LIST SWARM_TEST_DEPS)

#list(APPEND C_TEST_SRCS "demos/C_Test.cpp")
#list(APPEND C_TEST_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "C_Test")
#list(APPEND DEMO_SOURCES_LIST C_TEST_SRCS)
#list(APPEND DEMO_DEPENDS_LIST C_TEST_DEPS)

#list(APPEND CAR_RACING_SRCS "demos/Car_Racing.cpp")
#list(APPEND CAR_RACING_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Car_Racing")
#list(APPEND DEMO_SOURCES_LIST CAR_RACING_SRCS)
#list(APPEND DEMO_DEPENDS_LIST CAR_RACING_DEPS)

#list(APPEND STACKING_RL_SRCS "demos/Stacking_RL.cpp")
#list(APPEND STACKING_RL_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Stacking_RL")
#list(APPEND DEMO_SOURCES_LIST STACKING_RL_SRCS)
#list(APPEND DEMO_DEPENDS_LIST STACKING_RL_DEPS)

#list(APPEND STACKING_PROG_SRCS "demos/Stacking_Prog.cpp")
#list(APPEND STACKING_PROG_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Stacking_Prog")
#list(APPEND DEMO_SOURCES_LIST STACKING_PROG_SRCS)
#list(APPEND DEMO_DEPENDS_LIST STACKING_PROG_DEPS)

list(APPEND PUSHER_SRCS "demos/Pusher.cpp")
list(APPEND PUSHER_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "Pusher")
list(APPEND DEMO_SOURCES_LIST PUSHER_SRCS)
list(APPEND DEMO_DEPENDS_LIST PUSHER_DEPS)

list(APPEND ARTTEST_SRCS "demos/ARTTest.cpp")
list(APPEND ARTTEST_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "ARTTest")
list(APPEND DEMO_SOURCES_LIST ARTTEST_SRCS)
list(APPEND DEMO_DEPENDS_LIST ARTTEST_DEPS)

list(APPEND CAT_MOUSE_SRCS "demos/Cat_Mouse.cpp")
list(APPEND CAT_MOUSE_SRCS "demos/catmouse/CatMouseEnv.cpp")
list(APPEND CAT_MOUSE_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "Cat_Mouse")
list(APPEND DEMO_SOURCES_LIST CAT_MOUSE_SRCS)
list(APPEND DEMO_DEPENDS_LIST CAT_MOUSE_DEPS)

#list(APPEND EXPLORE_SRCS "demos/Explore.cpp")
#list(APPEND EXPLORE_SRCS "demos/explore/ExploreEnv.cpp")
#list(APPEND EXPLORE_SRCS "demos/helpers.cpp")
#list(APPEND EXPLORE_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Explore")
#list(APPEND DEMO_SOURCES_LIST EXPLORE_SRCS)
#list(APPEND DEMO_DEPENDS_LIST EXPLORE_DEPS)

#list(APPEND CAT_MOUSE_POS_SRCS "demos/Cat_Mouse_Pos.cpp")
#list(APPEND CAT_MOUSE_POS_SRCS "demos/catmouse/CatMouseEnv.cpp")
#list(APPEND CAT_MOUSE_POS_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Cat_Mouse_Pos")
#list(APPEND DEMO_SOURCES_LIST CAT_MOUSE_POS_SRCS)
#list(APPEND DEMO_DEPENDS_LIST CAT_MOUSE_POS_DEPS)

#list(APPEND NAVIGRAPH_SRCS "demos/NaviGraph.cpp")
#list(APPEND NAVIGRAPH_SRCS "demos/constructs/Vec2f.cpp")
#list(APPEND NAVIGRAPH_SRCS "demos/constructs/Vec3f.cpp")
#list(APPEND NAVIGRAPH_SRCS "demos/constructs/Vec4f.cpp")
#list(APPEND NAVIGRAPH_SRCS "demos/constructs/Matrix3x3f.cpp")
#list(APPEND NAVIGRAPH_SRCS "demos/constructs/Matrix4x4f.cpp")
#list(APPEND NAVIGRAPH_SRCS "demos/constructs/Quaternion.cpp")
#list(APPEND NAVIGRAPH_SRCS "demos/navigraph/GridCells.cpp")
#list(APPEND NAVIGRAPH_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "NaviGraph")
#list(APPEND DEMO_SOURCES_LIST NAVIGRAPH_SRCS)
#list(APPEND DEMO_DEPENDS_LIST NAVIGRAPH_DEPS)

#list(APPEND PREDICTORVIS_SRCS "demos/PredictorVis.cpp")
#list(APPEND PREDICTORVIS_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "PredictorVis")
#list(APPEND DEMO_SOURCES_LIST PREDICTORVIS_SRCS)
#list(APPEND DEMO_DEPENDS_LIST PREDICTORVIS_DEPS)

#list(APPEND VIT_TEST_SRCS "demos/ViT_Test.cpp")
#list(APPEND VIT_TEST_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "ViT_Test")
#list(APPEND DEMO_SOURCES_LIST VIT_TEST_SRCS)
#list(APPEND DEMO_DEPENDS_LIST VIT_TEST_DEPS)

#list(APPEND CONV_TEST_SRCS "demos/Conv_Test.cpp")
#list(APPEND CONV_TEST_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Conv_Test")
#list(APPEND DEMO_SOURCES_LIST CONV_TEST_SRCS)
#list(APPEND DEMO_DEPENDS_LIST CONV_TEST_DEPS)

#list(APPEND CAT_MOUSE_TORCH_SRCS "demos/Cat_Mouse_Torch.cpp")
#list(APPEND CAT_MOUSE_TORCH_SRCS "demos/catmouse/CatMouseEnv.cpp")
#list(APPEND CAT_MOUSE_TORCH_DEPS "SFML")
#list(APPEND CAT_MOUSE_TORCH_DEPS "TORCH")
#list(APPEND DEMO_PROJECTS_LIST "Cat_Mouse_Torch")
#list(APPEND DEMO_SOURCES_LIST CAT_MOUSE_TORCH_SRCS)
#list(APPEND DEMO_DEPENDS_LIST CAT_MOUSE_TORCH_DEPS)

#list(APPEND INVARIANCE_SRCS "demos/Invariance.cpp")
#list(APPEND INVARIANCE_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Invariance")
#list(APPEND DEMO_SOURCES_LIST INVARIANCE_SRCS)
#list(APPEND DEMO_DEPENDS_LIST INVARIANCE_DEPS)

#list(APPEND GENERATIVE_SRCS "demos/Generative.cpp")
#list(APPEND GENERATIVE_SRCS "demos/generative/Layer.cpp")
#list(APPEND GENERATIVE_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Generative")
#list(APPEND DEMO_SOURCES_LIST GENERATIVE_SRCS)
#list(APPEND DEMO_DEPENDS_LIST GENERATIVE_DEPS)

#list(APPEND BASIC_SRCS "demos/Basic.cpp")
#list(APPEND BASIC_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Basic")
#list(APPEND DEMO_SOURCES_LIST BASIC_SRCS)
#list(APPEND DEMO_DEPENDS_LIST BASIC_DEPS)

#list(APPEND GENERATIVE_SRCS "demos/Diffuser.cpp")
#list(APPEND GENERATIVE_SRCS "demos/generative/diffuser.cpp")
#list(APPEND GENERATIVE_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Generative")
#list(APPEND DEMO_SOURCES_LIST GENERATIVE_SRCS)
#list(APPEND DEMO_DEPENDS_LIST GENERATIVE_DEPS)

#list(APPEND EVO_SRCS "demos/Evo.cpp")
#list(APPEND EVO_SRCS "demos/evo/CPPN.cpp")
#list(APPEND EVO_SRCS "demos/evo/Unit.cpp")
#list(APPEND EVO_SRCS "demos/evo/Lattice.cpp")
#list(APPEND EVO_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Evo")
#list(APPEND DEMO_SOURCES_LIST EVO_SRCS)
#list(APPEND DEMO_DEPENDS_LIST EVO_DEPS)

#list(APPEND ART_VISUALIZER_SRCS "demos/ART_Visualizer.cpp")
#list(APPEND ART_VISUALIZER_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "ART_Visualizer")
#list(APPEND DEMO_SOURCES_LIST ART_VISUALIZER_SRCS)
#list(APPEND DEMO_DEPENDS_LIST ART_VISUALIZER_DEPS)

list(APPEND RUNNER_RUN_SRCS "demos/Runner_Run.cpp")
list(APPEND RUNNER_RUN_SRCS "demos/runner/Runner.cpp")
list(APPEND RUNNER_RUN_DEPS "SFML")
list(APPEND RUNNER_RUN_DEPS "BOX2D")
list(APPEND DEMO_PROJECTS_LIST "Runner_Run")
list(APPEND DEMO_SOURCES_LIST RUNNER_RUN_SRCS)
list(APPEND DEMO_DEPENDS_LIST RUNNER_RUN_DEPS)

#list(APPEND REACHER_RUN_SRCS "demos/Reacher.cpp")
#list(APPEND REACHER_RUN_DEPS "SFML")
#list(APPEND REACHER_RUN_DEPS "BOX2D")
#list(APPEND DEMO_PROJECTS_LIST "Reacher")
#list(APPEND DEMO_SOURCES_LIST REACHER_RUN_SRCS)
#list(APPEND DEMO_DEPENDS_LIST REACHER_RUN_DEPS)

#list(APPEND STDP_RUN_SRCS "demos/STDP.cpp")
#list(APPEND STDP_RUN_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "STDP")
#list(APPEND DEMO_SOURCES_LIST STDP_RUN_SRCS)
#list(APPEND DEMO_DEPENDS_LIST STDP_RUN_DEPS)

#list(APPEND BALL_PHYSICS_SRCS "demos/Ball_Physics.cpp")
#list(APPEND BALL_PHYSICS_DEPS "SFML")
#list(APPEND BALL_PHYSICS_DEPS "BOX2D")
#list(APPEND DEMO_PROJECTS_LIST "Ball_Physics")
#list(APPEND DEMO_SOURCES_LIST BALL_PHYSICS_SRCS)
#list(APPEND DEMO_DEPENDS_LIST BALL_PHYSICS_DEPS)

#list(APPEND BALL_PHYSICS_VEC_RUN_SRCS "demos/Ball_Physics_Vec.cpp")
#list(APPEND BALL_PHYSICS_VEC_RUN_DEPS "SFML")
#list(APPEND BALL_PHYSICS_VEC_RUN_DEPS "BOX2D")
#list(APPEND DEMO_PROJECTS_LIST "Ball_Physics_Vec")
#list(APPEND DEMO_SOURCES_LIST BALL_PHYSICS_VEC_RUN_SRCS)
#list(APPEND DEMO_DEPENDS_LIST BALL_PHYSICS_VEC_RUN_DEPS)

#list(APPEND ARDUINOVIS_SRCS "demos/ArduinoVis.cpp")
#list(APPEND ARDUINOVIS_DEPS "SFML")
#list(APPEND ARDUINOVIS_DEPS "RAYLIB")
#list(APPEND DEMO_PROJECTS_LIST "ArduinoVis")
#list(APPEND DEMO_SOURCES_LIST ARDUINOVIS_SRCS)
#list(APPEND DEMO_DEPENDS_LIST ARDUINOVIS_DEPS)

#list(APPEND ENCVIS_SRCS "demos/EncVis.cpp")
#list(APPEND ENCVIS_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "EncVis")
#list(APPEND DEMO_SOURCES_LIST ENCVIS_SRCS)
#list(APPEND DEMO_DEPENDS_LIST ENCVIS_DEPS)

#list(APPEND TOPO_TEST_AON_SRCS "demos/Topo_Test_AON.cpp")
#list(APPEND TOPO_TEST_AON_SRCS "demos/constructs/Vec2f.cpp")
#list(APPEND TOPO_TEST_AON_SRCS "demos/constructs/Vec3f.cpp")
#list(APPEND TOPO_TEST_AON_SRCS "demos/constructs/Vec4f.cpp")
#list(APPEND TOPO_TEST_AON_SRCS "demos/constructs/Matrix3x3f.cpp")
#list(APPEND TOPO_TEST_AON_SRCS "demos/constructs/Matrix4x4f.cpp")
#list(APPEND TOPO_TEST_AON_SRCS "demos/constructs/Quaternion.cpp")
#list(APPEND TOPO_TEST_AON_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Topo_Test_AON")
#list(APPEND DEMO_SOURCES_LIST TOPO_TEST_AON_SRCS)
#list(APPEND DEMO_DEPENDS_LIST TOPO_TEST_AON_DEPS)

#list(APPEND FLUID_SRCS "demos/Fluid.cpp")
#list(APPEND FLUID_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "Fluid")
#list(APPEND DEMO_SOURCES_LIST FLUID_SRCS)
#list(APPEND DEMO_DEPENDS_LIST FLUID_DEPS)

#list(APPEND CARTPOLE_SRCS "demos/CartPole.cpp")
#list(APPEND CARTPOLE_SRCS "demos/cartpole/cartpole.cpp")
#list(APPEND CARTPOLE_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "CartPole")
#list(APPEND DEMO_SOURCES_LIST CARTPOLE_SRCS)
#list(APPEND DEMO_DEPENDS_LIST CARTPOLE_DEPS)

#list(APPEND SPA_SRCS "demos/SPA.cpp")
#list(APPEND SPA_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "SPA")
#list(APPEND DEMO_SOURCES_LIST SPA_SRCS)
#list(APPEND DEMO_DEPENDS_LIST SPA_DEPS)

#list(APPEND VSA_SRCS "demos/VSA_Tests.cpp")
#list(APPEND VSA_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "VSA")
#list(APPEND DEMO_SOURCES_LIST VSA_SRCS)
#list(APPEND DEMO_DEPENDS_LIST VSA_DEPS)

#list(APPEND VSA_COMP_SRCS "demos/VSA_Tests_Comp.cpp")
#list(APPEND VSA_COMP_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "VSA_Comp")
#list(APPEND DEMO_SOURCES_LIST VSA_COMP_SRCS)
#list(APPEND DEMO_DEPENDS_LIST VSA_COMP_DEPS)

#list(APPEND VSA_COMP2_SRCS "demos/VSA_Tests_Comp2.cpp")
#list(APPEND VSA_COMP2_SRCS "demos/helpers.cpp")
#list(APPEND VSA_COMP2_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "VSA_Comp2")
#list(APPEND DEMO_SOURCES_LIST VSA_COMP2_SRCS)
#list(APPEND DEMO_DEPENDS_LIST VSA_COMP2_DEPS)

#list(APPEND VSA_COMP2_SRCS "demos/VSA_Tests_Video.cpp")
#list(APPEND VSA_COMP2_SRCS "demos/helpers.cpp")
#list(APPEND VSA_COMP2_DEPS "SFML")
#list(APPEND VSA_COMP2_DEPS "OPENCV")
#list(APPEND DEMO_PROJECTS_LIST "VSA_Video")
#list(APPEND DEMO_SOURCES_LIST VSA_COMP2_SRCS)
#list(APPEND DEMO_DEPENDS_LIST VSA_COMP2_DEPS)

#list(APPEND VSA_CHAR_SINGLE_SRCS "demos/VSA_Char_Single.cpp")
#list(APPEND VSA_CHAR_SINGLE_SRCS "demos/helpers.cpp")
#list(APPEND VSA_CHAR_SINGLE_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "VSA_Char_Single")
#list(APPEND DEMO_SOURCES_LIST VSA_CHAR_SINGLE_SRCS)
#list(APPEND DEMO_DEPENDS_LIST VSA_CHAR_SINGLE_DEPS)

#list(APPEND VSA_CHAR_SRCS "demos/VSA_Char.cpp")
#list(APPEND VSA_CHAR_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "VSA_Char")
#list(APPEND DEMO_SOURCES_LIST VSA_CHAR_SRCS)
#list(APPEND DEMO_DEPENDS_LIST VSA_CHAR_DEPS)

#list(APPEND US_CHAR_SRCS "demos/US_Char.cpp")
#list(APPEND DEMO_PROJECTS_LIST "US_Char")
#list(APPEND DEMO_SOURCES_LIST US_CHAR_SRCS)
#list(APPEND DEMO_DEPENDS_LIST US_CHAR_DEPS)

#list(APPEND VSA_CHAR_RECALL_SRCS "demos/VSA_Char_Recall.cpp")
#list(APPEND VSA_CHAR_RECALL_DEPS "SFML")
#list(APPEND DEMO_PROJECTS_LIST "VSA_Char_Recall")
#list(APPEND DEMO_SOURCES_LIST VSA_CHAR_RECALL_SRCS)
#list(APPEND DEMO_DEPENDS_LIST VSA_CHAR_RECALL_DEPS)

list(APPEND FUNC_TESTS_SRCS "demos/FuncTests.cpp")
list(APPEND FUNC_TESTS_DEPS "SFML")
list(APPEND DEMO_PROJECTS_LIST "FuncTests")
list(APPEND DEMO_SOURCES_LIST FUNC_TESTS_SRCS)
list(APPEND DEMO_DEPENDS_LIST FUNC_TESTS_DEPS)

message(STATUS "Demos to build: ${DEMO_PROJECTS_LIST}")

list(LENGTH DEMO_PROJECTS_LIST num_demos)

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
    set_property(TARGET ${DEMO_PROJECT} PROPERTY CXX_STANDARD 23)
    set_property(TARGET ${DEMO_PROJECT} PROPERTY CXX_STANDARD_REQUIRED ON)

    if(NOT AOGMANEO_FOUND)
        add_dependencies(${DEMO_PROJECT} AOgmaNeo)
    endif()
    find_package(OpenMP REQUIRED)

    target_link_libraries(${DEMO_PROJECT} ${OpenMP_CXX_LIBRARIES})

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${TORCH_CXX_FLAGS} -fPIC") # This links -fopenmp

    target_link_libraries(${DEMO_PROJECT} ${AOGMANEO_LIBRARIES})
    target_link_libraries(${DEMO_PROJECT} ${ULTRASPARSE_LIBRARIES})
    target_link_libraries(${DEMO_PROJECT} ${OGMANEO_LIBRARIES})

    # Does this demo require SFML?
    list(FIND ${DEMO_DEPENDS} "SFML" _sfml_depends_index)
    if (${_sfml_depends_index} GREATER -1)
        target_link_libraries(${DEMO_PROJECT} SFML::Window SFML::Graphics)
    endif()

    # Does this demo require Box2D?
    list(FIND ${DEMO_DEPENDS} "BOX2D" _box2d_depends_index)
    if (${_box2d_depends_index} GREATER -1)
        target_link_libraries(${DEMO_PROJECT} "/usr/local/lib/libbox2d.so")
    endif()

    # Does this demo require OpenCV?
    list(FIND ${DEMO_DEPENDS} "OPENCV" _opencv_depends_index)
    if (${_opencv_depends_index} GREATER -1)
        target_link_libraries(${DEMO_PROJECT} ${OpenCV_LIBS})
    endif()

    # Does this demo require Raylib?
    list(FIND ${DEMO_DEPENDS} "RAYLIB" _raylib_depends_index)
    if (${_raylib_depends_index} GREATER -1)
        target_link_libraries(${DEMO_PROJECT} raylib)
    endif()

    list(FIND ${DEMO_DEPENDS} "TORCH" _torch_depends_index)
    if (${_torch_depends_index} GREATER -1)
        target_link_libraries(${DEMO_PROJECT} ${TORCH_LIBRARIES})
    endif()
endforeach()
