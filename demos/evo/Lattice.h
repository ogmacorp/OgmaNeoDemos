#pragma once

#include "Unit.h"
#include <tuple>

class Lattice {
public:
    struct Node {
        std::vector<std::vector<float>> ffValues;
        std::vector<std::vector<float>> latValues;
        std::vector<std::vector<float>> fbValues;
    };

private:
    std::pair<int, int> inputSize;
    int numLayers;

    int ffRadius;
    int latRadius;
    int fbRadius;

    std::vector<Node> nodes;

    std::vector<std::vector<std::vector<float>>> states;
    std::vector<std::vector<std::vector<float>>> statesPrev;

public:
    void initRandom(const Unit &unit, const std::pair<int, int> &inputSize, int numLayers, int ffRadius, int latRadius, int fbRadius, std::mt19937 &rng, float initMinValue = -0.001f, float initMaxValue = 0.001f);

    void step(const std::vector<std::vector<float>> &inputStates, const Unit &unit, int subSteps);

    const std::vector<std::vector<float>> &getStates() const {
        return states[0];
    }
};
