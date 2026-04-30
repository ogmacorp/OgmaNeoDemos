#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <vector>
#include <time.h>
#include <iostream>
#include <random>

#define USE_MNIST_LOADER
#include "mnist.h"

class DDVFA {
public:
    int num_f1_cells;
    int num_f2_columns;
    int num_f2_cells_per_column;

    std::vector<float> weights;
    std::vector<float> matches;
    std::vector<float> activations;
    std::vector<float> comparisons;
    std::vector<int> cell_indices;
    std::vector<int> learn_column_indices;

    float vigilance_low = 0.8f;
    float vigilance_high = 0.9f;
    float alpha = 0.01f;
    float beta = 0.5f;
    int k = 3;

    void init(
        int num_f1_cells,
        int num_f2_columns,
        int num_f2_cells_per_column,
        std::mt19937 &rng
    ) {
        this->num_f1_cells = num_f1_cells;
        this->num_f2_columns = num_f2_columns;
        this->num_f2_cells_per_column = num_f2_cells_per_column;

        weights.resize(num_f1_cells * num_f2_columns * num_f2_cells_per_column);

        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        for (int i = 0; i < weights.size(); i++)
            weights[i] = 1.0f - dist01(rng) * 0.01f;

        matches.resize(num_f2_columns * num_f2_cells_per_column, 0.0f);
        activations.resize(matches.size(), 0.0f);
        comparisons.resize(num_f2_columns, 0.0f);
        cell_indices.resize(num_f2_columns, -1);
        learn_column_indices.resize(num_f2_columns, -1); // allocate full amount here but actually only goes to k
    }

    void step0(
        const std::vector<float> &inputs
    ) {
        int num_f2_cells = num_f2_columns * num_f2_cells_per_column;

        // matches and activations
        for (int i = 0; i < num_f2_cells; i++) {
            float min_sum = 0.0f;
            float total_input = 0.0f;
            float total_weight = 0.0f;

            for (int j = 0; j < num_f1_cells; j++) {
                int wi = j + num_f1_cells * i;

                min_sum += std::min(weights[wi], inputs[j]);
                total_input += inputs[j];
                total_weight += weights[wi];
            }

            float match = min_sum / total_input;
            float activation = min_sum / (alpha + total_weight);

            matches[i] = match;
            activations[i] = activation;
        }
    }

    void step1() {
        // simplified ART loop, per-column
        for (int c = 0; c < num_f2_columns; c++) {
            int winner_cell_low = -1;
            int winner_cell_high = -1;
            float max_activation_low = 0.0f;
            float max_activation_high = 0.0f;

            for (int d = 0; d < num_f2_cells_per_column; d++) {
                int i = d + num_f2_cells_per_column * c;

                if (matches[i] >= vigilance_low && activations[i] > max_activation_low) {
                    max_activation_low = activations[i];
                    winner_cell_low = i;
                }

                if (matches[i] >= vigilance_high && activations[i] > max_activation_high) {
                    max_activation_high = activations[i];
                    winner_cell_high = i;
                }
            }

            cell_indices[c] = winner_cell_high;

            comparisons[c] = max_activation_low;
        }

    }

    void step2() {
        // find top k in comparison between columns and learn those
        for (int t = 0; t < k; t++) {
            int max_column = -1;
            float max_comparison = 0.0f;

            // find max comparison
            for (int c = 0; c < num_f2_columns; c++) {
                if (comparisons[c] > max_comparison) {
                    max_comparison = comparisons[c];
                    max_column = c;
                }
            }

            if (max_column != -1) {
                // set activation to 0 so don't re-select
                comparisons[max_column] = 0.0f;

                learn_column_indices[t] = max_column;
            }
        }
    }

    void step3(
        const std::vector<float> &inputs
    ) {
        for (int t = 0; t < k; t++) {
            int max_column = learn_column_indices[t];

            // if column found
            if (max_column != -1) {
                // if cell found
                if (cell_indices[max_column] != -1) {
                    // perform learning on cell
                    int i = cell_indices[max_column] + num_f2_cells_per_column * max_column;

                    for (int j = 0; j < num_f1_cells; j++) {
                        int wi = j + num_f1_cells * i;

                        weights[wi] += beta * std::min(0.0f, inputs[j] - weights[wi]);
                    }
                }
            }
        }
    }

    void step(
        const std::vector<float> &inputs,
        bool learn_enabled
    ) {
        step0(inputs);
        step1();

        if (learn_enabled) {
            step2();
            step3(inputs);
        }
    }
};

int main() {
    // RNG
    std::mt19937 rng(time(nullptr));

    // Create window
    sf::ContextSettings glContextSettings;

    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(800, 600)), "Runner Demo", sf::Style::Default, sf::State::Windowed, glContextSettings);

    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    // ---------------------------- Game Loop -----------------------------

    sf::View view = window.getDefaultView();

    bool quit = false;

    sf::Clock clock;

    float dt = 0.017f;

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    DDVFA a;
    a.init(28 * 28, 16, 16, rng);

    const sf::Vector2f cell_size(64.0f, 64.0f);
    const sf::Vector2f grid_start(128.0f, 0.0f);
    const sf::Vector2f grid_spacing(2.0f, 2.0f);
    const sf::Vector2f spaced_cell_size = cell_size + grid_spacing;
    const sf::Vector2f grid_size = spaced_cell_size * sf::Vector2f(a.num_f2_columns, a.num_f2_cells_per_column);

    mnist_data* data;
    unsigned int cnt;
    int ret = mnist_load("resources/train-images-idx3-ubyte", "resources/train-labels-idx1-ubyte", &data, &cnt);

    if (ret) {
        printf("An error occured: %d\n", ret);
        return -1;
    } else {
        printf("image count: %d\n", cnt);
    }

    std::uniform_int_distribution<int> data_dist(0, cnt - 1);

    int current_input_index = data_dist(rng);

    int state = 0;
    bool state_switch = false;

    std::vector<float> inputs(28 * 28, 0.0f);

    sf::Image input_image(sf::Vector2u(28, 28));
    sf::Image weights_image(sf::Vector2u(28, 28));

    do {
        clock.restart();

        // ----------------------------- Input -----------------------------

        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                quit = true;
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape))
                quit = true;
        }

        int label = data[current_input_index].label;

        for (int i = 0; i < inputs.size(); i++) {
            std::uint8_t gray = data[current_input_index].data[i % 28][static_cast<int>(i / 28)];

            inputs[i] = gray / 255.0f;

            input_image.setPixel(sf::Vector2u(i % 28, i / 28), sf::Color(gray, gray, gray));
        }

        if (state_switch) {
            switch(state) {
            case 0: { // match and activation
                a.step0(inputs);
                break;
            }

            case 1: { // resonance in column
                a.step1();
                break;
            }

            case 2: { // resonance between columns
                a.step2();
                break;
            }

            case 3: { // learning
                a.step3(inputs);
                break;
            }
            }

            state_switch = false;
        }

        // render current state
        sf::RectangleShape rs;

        rs.setSize(cell_size);
        rs.setFillColor(sf::Color::Black);
        rs.setOutlineThickness(1.0f);
        rs.setOutlineColor(sf::Color::White);

        for (int c = 0; c < a.num_f2_columns; c++) {
            for (int d = 0; d < a.num_f2_cells_per_column; d++) {
                rs.set_po                
            }
        }

        window.display();
    } while (!quit);

    return 0;
}
