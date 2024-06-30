#pragma once

#include "../constructs/Matrix4x4f.h"
#include <vector>
#include <random>
#include <tuple>
#include <assert.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

typedef std::vector<int> CSDR;
typedef std::vector<float> QuasiCSDR;

float getSimilarity(
    const CSDR &left,
    const CSDR &right
);

float getSimilarity(
    const Matrix4x4f &left,
    const Matrix4x4f &right
);

void merge(
    const CSDR &left,
    const CSDR &right,
    CSDR &result
);

void toward(
    const CSDR &target,
    QuasiCSDR &qcsdr,
    float rate
);

void inhibit(
    const QuasiCSDR &qcsdr,
    CSDR &csdr
);

void initQuasiCSDR(
    const CSDR &csdr,
    QuasiCSDR &qcsdr
);

class GridCellModule {
private:
    struct Cell {
        std::vector<float> inputProtos;
        std::vector<float> predictionWeights;

        float activation;

        Cell()
        :
        activation(0.0f)
        {}
    };

public:
    float falloff;
    float lr;

    GridCellModule()
    :
    falloff(1.0f),
    lr(0.01f)
    {}

    void initRandom(
        int width,
        int height,
        std::mt19937 &rng
    );

    std::vector<float> step(
        const std::vector<float> &inputs,
        const std::vector<float> &content
    );
};
