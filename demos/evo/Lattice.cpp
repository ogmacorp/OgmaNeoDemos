#include "Lattice.h"
#include <iostream>

void Lattice::initRandom(const Unit &unit, const std::pair<int, int> &inputSize, int numLayers, int ffRadius, int latRadius, int fbRadius, std::mt19937 &rng, float initMinValue, float initMaxValue) {
    this->inputSize = inputSize;
    this->numLayers = numLayers;

    int numInputs = std::get<0>(inputSize) * std::get<1>(inputSize);

    this->ffRadius = ffRadius;
    this->latRadius = latRadius;
    this->fbRadius = fbRadius;

    nodes.resize(numInputs * numLayers);

    int ffDiam = ffRadius * 2 + 1;
    int ffArea = ffDiam * ffDiam;
    int latDiam = latRadius * 2 + 1;
    int latArea = latDiam * latDiam;
    int fbDiam = fbRadius * 2 + 1;
    int fbArea = fbDiam * fbDiam;

    std::uniform_real_distribution<float> valueDist(initMinValue, initMaxValue);

    states.resize(numLayers);

    for (int l = 0; l < numLayers; l++) {
        states[l].resize(numInputs);

        for (int i = 0; i < numInputs; i++)
            states[l][i].resize(unit.stateSize, 0.0f);
    }

    statesPrev = states;

    for (int i = 0; i < nodes.size(); i++) {
        nodes[i].ffValues.resize(ffArea);

        for (int j = 0; j < ffArea; j++) {
            nodes[i].ffValues[j].resize(unit.connectionValues);

            for (int k = 0; k < unit.connectionValues; k++)
                nodes[i].ffValues[j][k] = valueDist(rng);
        }

        nodes[i].latValues.resize(latArea);

        for (int j = 0; j < latArea; j++) {
            nodes[i].latValues[j].resize(unit.connectionValues);

            for (int k = 0; k < unit.connectionValues; k++)
                nodes[i].latValues[j][k] = valueDist(rng);
        }

        nodes[i].fbValues.resize(fbArea);

        for (int j = 0; j < fbArea; j++) {
            nodes[i].fbValues[j].resize(unit.connectionValues);

            for (int k = 0; k < unit.connectionValues; k++)
                nodes[i].fbValues[j][k] = valueDist(rng);
        }
    }
}

void Lattice::step(const std::vector<std::vector<float>> &inputStates, const Unit &unit, int subSteps) {
    int ffDiam = ffRadius * 2 + 1;
    int latDiam = latRadius * 2 + 1;
    int fbDiam = fbRadius * 2 + 1;

    int numInputs = std::get<0>(inputSize) * std::get<1>(inputSize);

    for (int ss = 0; ss < subSteps; ss++) {
        for (int i = 0; i < numInputs; i++) {
            states[0][i][0] = inputStates[i][0];
            statesPrev[0][i][0] = inputStates[i][0];
        }

        for (int l = 0; l < numLayers; l++) {
            for (int i = 0; i < numInputs; i++) {
                int x = i / std::get<1>(inputSize);
                int y = i % std::get<1>(inputSize);

                int ni = i + numInputs * l;

                std::vector<float> accum(unit.connectionValues * 3, 0.0f);
                
                int accumOffset = 0;

                if (l > 0) {
                    for (int dx = -ffRadius; dx <= ffRadius; dx++)
                        for (int dy = -ffRadius; dy <= ffRadius; dy++) {
                            int ci = (dy + ffRadius) + ffDiam * (dx + ffRadius);

                            int nx = x + dx;
                            int ny = y + dy;

                            if (nx < 0 || ny < 0 || nx >= std::get<0>(inputSize) || ny >= std::get<1>(inputSize))
                                continue;

                            std::vector<float> inputs = statesPrev[l - 1][ny + nx * std::get<1>(inputSize)];

                            inputs.insert(inputs.end(), nodes[ni].ffValues[ci].begin(), nodes[ni].ffValues[ci].end());

                            std::vector<float> outputs = unit.ff.forward(inputs);

                            for (int o = 0; o < unit.connectionValues; o++) {
                                nodes[ni].ffValues[ci][o] += std::min(1.0f, std::max(0.0f, outputs[0 * unit.connectionValues + o])) * (outputs[1 * unit.connectionValues + o] - nodes[ni].ffValues[ci][o]);

                                accum[accumOffset + o] += nodes[ni].ffValues[ci][o];
                            }
                        }
                }

                accumOffset += unit.connectionValues;

                for (int dx = -latRadius; dx <= latRadius; dx++)
                    for (int dy = -latRadius; dy <= latRadius; dy++) {
                        int ci = (dy + latRadius) + latDiam * (dx + latRadius);

                        int nx = x + dx;
                        int ny = y + dy;

                        if (nx < 0 || ny < 0 || nx >= std::get<0>(inputSize) || ny >= std::get<1>(inputSize))
                            continue;

                        std::vector<float> inputs = statesPrev[l][ny + nx * std::get<1>(inputSize)];

                        inputs.insert(inputs.end(), nodes[ni].latValues[ci].begin(), nodes[ni].latValues[ci].end());

                        std::vector<float> outputs = unit.lat.forward(inputs);

                        for (int o = 0; o < unit.connectionValues; o++) {
                            nodes[ni].latValues[ci][o] += std::min(1.0f, std::max(0.0f, outputs[0 * unit.connectionValues + o])) * (outputs[1 * unit.connectionValues + o] - nodes[ni].latValues[ci][o]);

                            accum[accumOffset + o] += nodes[ni].latValues[ci][o];
                        }
                    }

                accumOffset += unit.connectionValues;

                if (l < numLayers - 1) {
                    for (int dx = -fbRadius; dx <= fbRadius; dx++)
                        for (int dy = -fbRadius; dy <= fbRadius; dy++) {
                            int ci = (dy + fbRadius) + fbDiam * (dx + fbRadius);

                            int nx = x + dx;
                            int ny = y + dy;

                            if (nx < 0 || ny < 0 || nx >= std::get<0>(inputSize) || ny >= std::get<1>(inputSize))
                                continue;

                            std::vector<float> inputs = statesPrev[l + 1][ny + nx * std::get<1>(inputSize)];

                            inputs.insert(inputs.end(), nodes[ni].fbValues[ci].begin(), nodes[ni].fbValues[ci].end());

                            std::vector<float> outputs = unit.fb.forward(inputs);

                            for (int o = 0; o < unit.connectionValues; o++) {
                                nodes[ni].fbValues[ci][o] += std::min(1.0f, std::max(0.0f, outputs[0 * unit.connectionValues + o])) * (outputs[1 * unit.connectionValues + o] - nodes[ni].fbValues[ci][o]);

                                accum[accumOffset + o] += nodes[ni].fbValues[ci][o];
                            }
                        }
                }

                std::vector<float> inputs = statesPrev[l][i];
                inputs.insert(inputs.end(), accum.begin(), accum.end());

                std::vector<float> coord = { static_cast<float>(x) / static_cast<float>(std::get<0>(inputSize)) * 2.0f - 1.0f, static_cast<float>(y) / static_cast<float>(std::get<1>(inputSize)) * 2.0f - 1.0f, static_cast<float>(l) };
                inputs.insert(inputs.end(), coord.begin(), coord.end());

                states[l][i] = unit.stateUpdater.forward(inputs);

                // Rectify
                for (int j = 0; j < unit.stateSize; j++)
                    states[l][i][j] = std::max(0.0f, states[l][i][j]);
            }
        }

        statesPrev = states;
    }
}
