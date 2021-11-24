#pragma once

#include <vector>
#include <random>

struct Layer {
    std::mt19937 rng;

    int inputWidth;
    int inputHeight;
    int inputDepth;
    int outputDepth;
    int columnSize;
    int stride;
    int radius;

    std::vector<float> reconstruction;
    std::vector<float> counts;
    std::vector<float> outputActivations;
    std::vector<float> outputs;

    std::vector<float> weights;

    void initRandom(
        int inputWidth,
        int inputHeight,
        int inputDepth,
        int outputDepth,
        int columnSize,
        int stride,
        int radius
    );

    void step(
        const std::vector<float> &inputs,
        float lr
    );

    void reconstruct(
        const std::vector<float> &reconOutputs
    );

    int getOutputWidth() const {
        return inputWidth / stride;
    }

    int getOutputHeight() const {
        return inputHeight / stride;
    }
};
