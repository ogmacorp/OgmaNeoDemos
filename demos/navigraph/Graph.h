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

class Graph {
public:
    struct Connection {
        Matrix4x4f relTrans;

        int ni;
    };

    struct Node {
        std::vector<Connection> connections;

        QuasiCSDR qcsdr;

        Matrix4x4f trans;
    };

private:
    int columnSize;

    std::vector<Node> nodes;

    int lastN;

    Matrix4x4f accumDelta;

public:
    Matrix4x4f estimated;

    float addThresh;
    float transScale;
    float transWeight;
    float drift;
    float elasticity;

    Graph()
    :
    lastN(-1),
    addThresh(0.98f),
    transScale(32.0f),
    transWeight(0.4f),
    drift(0.001f),
    elasticity(0.01f)
    {}

    void init(
        int columnSize
    );

    void step(
        const Matrix4x4f &delta,
        const CSDR &csdr
    );

    void renderXY(
        sf::RenderTarget &rt,
        float renderScale
    );
};

