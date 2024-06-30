#include <vector>
#include <random>
#include <tuple>
#include <unordered_map>
#include <assert.h>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

class GridCells {
private:
    int num_inputs;
    int width;
    int height;
    int filter_radius;

    std::vector<float> inject_weights;
    std::vector<float> states;
    std::vector<float> states_prev;
    std::vector<float> filter;

public:
    // params
    float alpha = 0.1f; // hat outer falloff
    float beta = 0.05f; // hat inner falloff
    float gamma = 0.6f; // hat dip
    float energy = 0.4f; // inject energy
    float scale = 1.0f; // final activation scale
    int iters = 2;

    void init_random(
        int num_inputs,
        int width,
        int height,
        int filter_radius,
        std::mt19937 &rng
    );

    void step(
        const std::vector<float> inputs,
        float shift_x,
        float shift_y
    );

    const std::vector<float> get_states() const {
        return states;
    }

    sf::Image get_states_image() const;
};


