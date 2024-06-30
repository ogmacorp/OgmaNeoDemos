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

class Grid {
public:
    sf::Texture attention_texture;

    struct Pair {
        int n0, n1;

        bool operator==(const Pair &other) const {
            return n0 == other.n0 && n1 == other.n1;
        }

        size_t operator()(const Pair &pair) const {
            return pair.n0 ^ (pair.n1 << 1);
        }
    };

    struct Node {
        CSDR qcsdr;
    };

    int width;
    int height;
    int columnSize;

    std::vector<Node> grid;

    Vec3f estimated_pose;

    std::vector<float> attention;

    Grid()
    {}

    void init(
        int width,
        int height,
        int columnSize
    );

    void step(
        const Vec3f &pose_shift,
        const CSDR &csdr
    );

    const sf::Texture &get_attention_texture();
};
