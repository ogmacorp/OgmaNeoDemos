#include "CPPN.h"

void CPPN::initRandom(const std::vector<int> &layerSizes, std::mt19937 &rng, float initMinWeight, float initMaxWeight) {
    std::uniform_real_distribution<float> weightDist(initMinWeight, initMaxWeight);

    weights.resize(layerSizes.size() - 1);
    biases.resize(weights.size());

    for (int l = 1; l < layerSizes.size(); l++) {
        int wl = l - 1;

        int inDim = layerSizes[wl];
        int outDim = layerSizes[l];

        weights[wl].resize(inDim * outDim);
        biases[wl].resize(outDim);

        for (int w = 0; w < weights[wl].size(); w++)
            weights[wl][w] = weightDist(rng);

        for (int w = 0; w < biases[wl].size(); w++)
            biases[wl][w] = weightDist(rng);
    }
}

void CPPN::initFromParents(const CPPN &parent1, const CPPN &parent2, std::mt19937 &rng) {
    weights = parent1.weights;
    biases = parent1.biases;

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int l = 0; l < weights.size(); l++) {
        for (int w = 0; w < weights[l].size(); w++) {
            if (dist01(rng) < 0.5f)
                weights[l][w] = parent2.weights[l][w];
        }

        for (int w = 0; w < biases[l].size(); w++) {
            if (dist01(rng) < 0.5f)
                biases[l][w] = parent2.biases[l][w];
        }
    }
}

void CPPN::mutate(float chance, float pertWeight, std::mt19937 &rng) {
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::normal_distribution<float> pertWeightDist(0.0f, pertWeight);

    for (int l = 0; l < weights.size(); l++) {
        for (int w = 0; w < weights[l].size(); w++) {
            if (dist01(rng) < chance)
                weights[l][w] += pertWeightDist(rng);
        }

        for (int w = 0; w < biases[l].size(); w++) {
            if (dist01(rng) < chance)
                biases[l][w] += pertWeightDist(rng);
        }
    }
}

std::vector<float> CPPN::forward(const std::vector<float> &inputs) const {
    std::vector<float> inActivations = inputs;

    for (int l = 0; l < weights.size(); l++) {
        bool isLastLayer = l == weights.size() - 1;

        std::vector<float> outActivations(biases[l].size());

        for (int i = 0; i < outActivations.size(); i++) {
            float act = biases[l][i];

            for (int j = 0; j < inActivations.size(); j++) {
                int w = i * inActivations.size() + j;

                act += inActivations[j] * weights[l][w];
            }

            outActivations[i] = isLastLayer ? act : std::sin(act);
        }

        inActivations = outActivations;
    }

    return inActivations;
}

int CPPN::getStateVecSize() const {
    int size = 0;

    for (int l = 0; l < weights.size(); l++) {
        size += weights[l].size();
        size += biases[l].size();
    }

    return size;
}

void CPPN::getStateVec(std::vector<float> &v) const {
    v.resize(getStateVecSize());

    int index = 0;

    for (int l = 0; l < weights.size(); l++) {
        for (int i = 0; i < weights[l].size(); i++)
            v[index++] = weights[l][i];
        
        for (int i = 0; i < biases[l].size(); i++)
            v[index++] = biases[l][i];
    }
}

void CPPN::setStateVec(const std::vector<float> &v) {
    int index = 0;

    for (int l = 0; l < weights.size(); l++) {
        for (int i = 0; i < weights[l].size(); i++)
            weights[l][i] = v[index++];

        for (int i = 0; i < biases[l].size(); i++)
            biases[l][i] = v[index++];
    }
}
