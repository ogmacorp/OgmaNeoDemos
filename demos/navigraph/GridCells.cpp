#include "GridCells.h"

void GridCells::init_random(
    int num_inputs,
    int width,
    int height,
    int filter_radius,
    std::mt19937 &rng
) {
    this->num_inputs = num_inputs;
    this->width = width;
    this->height = height;
    this->filter_radius = filter_radius;

    inject_weights.resize(width * height * num_inputs);

    std::normal_distribution<float> n_dist(0.0f, 1.0f);

    for (int i = 0; i < inject_weights.size(); i++)
        inject_weights[i] = n_dist(rng);

    states = std::vector<float>(width * height, 0.0f);
    states_prev = states;

    int filter_diam = filter_radius * 2 + 1;

    filter.resize(filter_diam * filter_diam);

    injections.resize(width * height);
}

void GridCells::step(
    const std::vector<float> inputs,
    float shift_x,
    float shift_y
) {
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            int state_index = y + height * x;

            float sum = 0.0f;

            for (int i = 0; i < num_inputs; i++)
                sum += inject_weights[i + num_inputs * state_index] * inputs[i];

            injections[state_index] = std::max(0.0f, sum) * energy;
        }

    // determine filter
    int filter_diam = filter_radius * 2 + 1;

    for (int dx = -filter_radius; dx <= filter_radius; dx++)
        for (int dy = -filter_radius; dy <= filter_radius; dy++) {
            float dxf = dx + shift_x / iters;
            float dyf = dy + shift_y / iters;

            float dist2 = dxf * dxf + dyf * dyf;

            filter[(filter_radius + dy) + filter_diam * (filter_radius + dx)] = expf(-alpha * dist2) - gamma * expf(-beta * dist2);
        }

    for (int it = 0; it < iters; it++) {
        // inhibit
        states_prev = states;

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                int state_index = y + height * x;

                float sum = 0.0f;
                float total = 0.0f;

                for (int dx = -filter_radius; dx <= filter_radius; dx++)
                    for (int dy = -filter_radius; dy <= filter_radius; dy++) {
                        int other_x = x + dx;
                        int other_y = y + dy;

                        if (other_x < 0)
                            other_x += width;
                        else if (other_x >= width)
                            other_x -= width;

                        if (other_y < 0)
                            other_y += height;
                        else if (other_y >= height)
                            other_y -= height;

                        int other_state_index = other_y + height * other_x;

                        float w = filter[(filter_radius + dy) + filter_diam * (filter_radius + dx)];

                        sum += w * states_prev[other_state_index];
                        total += w * w;
                    }

                states[state_index] = std::tanh(std::max(0.0f, injections[state_index] + sum / std::max(0.0001f, std::sqrt(total)) * scale));
            }
    }
}

sf::Image GridCells::get_states_image() const {
    sf::Image img;
    img.create(width, height);

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            int state_index = y + height * x;

            float intensity = states[state_index];

            sf::Color color;
            color.r = color.g = color.b = intensity * 255.0f;

            img.setPixel(x, y, color);
        }

    return img;
}
