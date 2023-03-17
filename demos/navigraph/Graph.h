#pragma once

#include "../constructs/Vec2f.h"
#include "../constructs/Matrix4x4f.h"
#include <vector>
#include <random>
#include <tuple>
#include <unordered_map>
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
    struct Pair {
        int n0, n1;

        bool operator==(const Pair &other) const {
            return n0 == other.n0 && n1 == other.n1;
        }

        size_t operator()(const Pair &pair) const {
            return pair.n0 ^ (pair.n1 << 1);
        }
    };

    struct Connection {
        Matrix4x4f relTrans;
    };

    struct Node {
        QuasiCSDR qcsdr;

        Matrix4x4f trans;

        Matrix4x4f transTemp;
        int countTemp;
    };

    std::unordered_map<Pair, Connection, Pair> connections;

    int columnSize;

    std::vector<Node> nodes;

    int lastN;

    Matrix4x4f estimated;

    float minSim;
    float minDist;
    float drift;
    float elasticity;
    float relElasticity;

    Graph()
    :
    lastN(-1),
    minSim(0.95f),
    minDist(0.05f),
    drift(0.001f),
    elasticity(0.1f),
    relElasticity(0.1f)
    {}

    void init(
        int columnSize
    );

    void step(
        const Matrix4x4f &delta,
        const CSDR &csdr
    );

    void step2D(
        const Vec2f &deltaPos,
        float deltaAngle,
        const CSDR &csdr
    ) {
        step(Matrix4x4f::translateMatrix(Vec3f(deltaPos.x, deltaPos.y, 0.0f)) * Matrix4x4f::rotateMatrixZ(deltaAngle), csdr);
    }

    void findPath(
        int startIndex,
        int endIndex,
        std::vector<int> &path
    );

    void renderXY(
        sf::RenderTarget &rt,
        float renderScale,
        const std::vector<int> &path = {}
    );
};

