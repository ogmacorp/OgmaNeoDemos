#pragma once

#include "helpers.h"

#include <iostream>
#include <fstream>
#include <vector>

using namespace aon;

// config for things that require recompilation
struct Static_Config {
    // sdc_controller
    Int3 image_size = Int3(64, 64, 1);
    Int3 enc_size = Int3(10, 10, 16);
    int enc_radius = 5;

    std::vector<Int3> layer_sizes = {
        Int3(5, 5, 32),
        Int3(5, 5, 32)
    };

    int io_up_radius = 2;
    int io_down_radius = 2;

    int io_num_dendrites_per_cell = 8;
    int io_value_num_dendrites_per_cell = 16;

    int layer_up_radius = 2;
    int layer_down_radius = 2;

    int layer_num_dendrites_per_cell = 4;

    bool rl_enabled = false;
    bool gen_enabled = false;

    float action_smoothing = 0.5f;

    // hardware
    float max_steer = 0.25f;

    // motion detector
    Int2 motion_size = Int2(32, 32);
    float ignore_motion_time = 1.5f; // 1.5 seconds
    float motion_filter_rate = 1.0f;

    // rewards
    float reward_throttle_scale = 1.0f;
    float reward_crash = -100.0f;

    // bc-specific
    float bc_min_throttle = 0.05f;

    // cropping
    float crop_ratio_lower_y = 0.0f;
    float crop_ratio_upper_y = 1.0f;
};

// config for things that can be saved/loaded
class Dynamic_Config {
public:
    // hardware
    float trim;

    // motion detector
    float motion_threshold;

    Dynamic_Config()
    :
    trim(0.0f),
    motion_threshold(0.2f)
    {}

    bool load(const std::string &file_name);
    bool save(const std::string &file_name);
};

// extern from program global instead of per-source global (static)
const Static_Config scfg;
extern Dynamic_Config dcfg;
