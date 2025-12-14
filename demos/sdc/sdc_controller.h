#pragma once

#include "helpers.h"
#include "config.h"

#include <iostream>
#include <string>

using namespace aon;

std::pair<int, int> encode_unorm6(
    float x
);

float decode_unorm6(
    const std::pair<int, int> &x
);

enum class SDC_Mode {
    bc = 0,
    rl = 1,
    inf = 2
};

class SDC_Controller {
private:
    Image_Encoder enc;
    Hierarchy h;

    float smooth_throttle = 0.0f;
    float smooth_steer = 0.0f;

public:
    void init_random();

    bool init_load(
        const std::string &file_name
    );

    void save(
        const std::string &file_name
    );

    std::pair<float, float> step(
        const std::vector<unsigned char> &image,
        SDC_Mode mode,
        float reward,
        const std::pair<float, float> &targets
    );
};
