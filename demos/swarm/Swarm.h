#pragma once

#include <vector>
#include <array>
#include <random>

inline float sigmoid(
    float x
) {
    if (x < 0.0f) {
        float z = std::exp(x);

        return z / (1.0f + z);
    }
    
    return 1.0f / (1.0f + std::exp(-x));
}

class Swarm {
public:
    struct Cell {
        std::array<float, 16> values;
        std::array<float, 16> traces;

        bool on;
        float value;

        Cell()
        :
        on(false),
        value(0.0f)
        {}
    };

private:
    int width;
    int height;

    std::vector<Cell> cells;

public:
    float lr;
    float discount;
    float traceDecay;
    float epsilon;

    Swarm()
    :
    lr(0.001f),
    discount(0.99f),
    traceDecay(0.97f),
    epsilon(0.001f)
    {}

    void init(
        int width,
        int height,
        std::mt19937 &rng
    );

    void step(
        const std::vector<bool> &inputs,
        float reward,
        std::mt19937 &rng,
        bool learnEnabled = true
    );

    int getWidth() const {
        return width;
    }

    int getHeight() const {
        return height;
    }

    bool getOn(
        int i
    ) const {
        return cells[i].on;
    }

    bool getOn(
        int x,
        int y
    ) const {
        return cells[y + x * height].on;
    }

    bool getLastOn(
        int x
    ) const {
        return cells[height - 1 + x * height].on;
    }
};
