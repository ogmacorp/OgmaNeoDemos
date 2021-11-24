#pragma once

#include <vector>
#include <random>

class CPPN {
private:
    std::vector<std::vector<float>> weights;
    std::vector<std::vector<float>> biases;
public:
    void initRandom(const std::vector<int> &layerSizes, std::mt19937 &rng, float initMinWeight = -1.0f, float initMaxWeight = 1.0f);
    void initFromParents(const CPPN &parent1, const CPPN &parent2, std::mt19937 &rng);
    void mutate(float chance, float pertWeight, std::mt19937 &rng);

    int getStateVecSize() const;
    void getStateVec(std::vector<float> &v) const;
    void setStateVec(const std::vector<float> &v);

    std::vector<float> forward(const std::vector<float> &inputs) const;
};
