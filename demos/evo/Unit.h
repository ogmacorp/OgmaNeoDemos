#pragma once

#include "CPPN.h"

struct Unit {
    int stateSize;
    int stateHidden;
    int connectionValues;
    int connectionHidden;

    CPPN stateUpdater;

    // Feed forward, lateral, feedback
    CPPN ff;
    CPPN lat;
    CPPN fb;

    void initRandom(int stateSize, int stateHidden, int connectionValues, int connectionHidden, std::mt19937 &rng, float initMinWeight = -1.0f, float initMaxWeight = 1.0f);
    void initFromParents(const Unit &parent1, const Unit &parent2, std::mt19937 &rng);
    void mutate(float chance, float pertWeight, std::mt19937 &rng);
};
