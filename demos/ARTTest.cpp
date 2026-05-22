#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <cmath>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>

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
    std::vector<int> learn_cell_indices;
    std::vector<int> learn_column_indices;
    std::vector<bool> committed;

    float vigilance_low = 0.85f;
    float vigilance_high = 0.9f;
    float alpha = 0.01f;
    float beta = 0.1f;
    float leak = 0.01f;
    float commit_rate = 0.99f;
    int k = 1;

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
            weights[i] = dist01(rng);

        matches.resize(num_f2_columns * num_f2_cells_per_column, 0.0f);
        activations.resize(matches.size(), 0.0f);
        comparisons.resize(num_f2_columns, 0.0f);
        cell_indices.resize(num_f2_columns, -1);
        learn_cell_indices.resize(num_f2_columns, -1);
        learn_column_indices.resize(num_f2_columns, -1); // allocate full amount here but actually only goes to k
        committed.resize(matches.size(), false);
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

                min_sum += (1.0f - leak) * std::min(weights[wi], inputs[j]) + leak * weights[wi] * inputs[j];
                total_input += inputs[j];
                total_weight += weights[wi];
            }

            float match = min_sum / std::max(0.0001f, total_input);
            float activation = min_sum / (alpha + total_weight);

            matches[i] = (!committed[i] ? 1.0f : match);
            activations[i] = activation;
        }
    }

    void step1() {
        // simplified ART loop, per-column
        for (int c = 0; c < num_f2_columns; c++) {
            int winner_cell_low = -1;
            int winner_cell_high = -1;
            int complete_winner_cell = 0;
            float max_activation_low = 0.0f;
            float max_activation_high = 0.0f;
            float max_complete_activation = 0.0f;

            for (int d = 0; d < num_f2_cells_per_column; d++) {
                int i = d + num_f2_cells_per_column * c;

                if (matches[i] >= vigilance_low && activations[i] > max_activation_low) {
                    max_activation_low = activations[i];
                    winner_cell_low = d;
                }

                if (matches[i] >= vigilance_high && activations[i] > max_activation_high) {
                    max_activation_high = activations[i];
                    winner_cell_high = d;
                }

                if (activations[i] > max_complete_activation) {
                    max_complete_activation = activations[i];
                    complete_winner_cell = d;
                }
            }

            cell_indices[c] = (winner_cell_high == -1 ? complete_winner_cell : winner_cell_high);
            learn_cell_indices[c] = winner_cell_high;

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
            }

            learn_column_indices[t] = max_column;
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
                if (learn_cell_indices[max_column] != -1) {
                    // perform learning on cell
                    int i = learn_cell_indices[max_column] + num_f2_cells_per_column * max_column;

                    if (committed[i]) {
                        for (int j = 0; j < num_f1_cells; j++) {
                            int wi = j + num_f1_cells * i;

                            weights[wi] += beta * std::min(0.0f, inputs[j] - weights[wi]);
                        }
                    }
                    else {
                        for (int j = 0; j < num_f1_cells; j++) {
                            int wi = j + num_f1_cells * i;

                            weights[wi] += commit_rate * (inputs[j] - weights[wi]);
                        }

                        committed[i] = true;
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
    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::normal_distribution<float> ndist(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(sf::Vector2u(1024, 1024)), "Pusher", sf::Style::Default);

    window.setFramerateLimit(60);

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    bool speedMode = false;
    bool tPressedPrev = false;

    sf::Clock clock;

    float dt = 0.017f;

    // Used for speed mode to render slower
    int renderCounter = 0;

    std::vector<std::vector<float>> data = {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };

    DDVFA a;
    a.init(4, 2, 2, rng);

    std::vector<sf::Color> palette(a.num_f2_columns * a.num_f2_cells_per_column);

    for (int i = 0; i < palette.size(); i++) {
        palette[i] = sf::Color(dist01(rng) * 255.0f, dist01(rng) * 255.0f, dist01(rng) * 255.0f);
    }

    sf::Image img(sf::Vector2u(256, 256));

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

            bool tPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::T);

            if (tPressed && !tPressedPrev)
                speedMode = !speedMode;

            tPressedPrev = tPressed;
        }

        // train
        std::uniform_int_distribution<int> sampleDist(0, 3);

        int index = sampleDist(rng);

        std::vector<float> complement_coded(4);
        complement_coded[0] = data[index][0];
        complement_coded[1] = data[index][1];
        complement_coded[2] = 1.0f - data[index][0];
        complement_coded[3] = 1.0f - data[index][1];

        a.step(complement_coded, true);

        int full_state = 0;
        int stride = 1;

        for (int i = 0; i < a.cell_indices.size(); i++) {
            full_state += a.cell_indices[i] * stride;
            stride *= a.num_f2_cells_per_column;
        }

        std::cout << index << " " << full_state << std::endl;

        if (!speedMode || renderCounter >= 300) {
            window.clear();

            renderCounter = 0;

            for (int x = 0; x < img.getSize().x; x++)
                for (int y = 0; y < img.getSize().y; y++) {
                    a.step({ static_cast<float>(x) / (img.getSize().x - 1), static_cast<float>(y) / (img.getSize().y - 1), 1.0f - static_cast<float>(x) / (img.getSize().x - 1), 1.0f - static_cast<float>(y) / (img.getSize().y - 1) }, false);

                    int state = a.cell_indices[0];
                    float act = a.activations[state];

                    if (state != -1)
                        //img.setPixel(sf::Vector2u(x, y), sf::Color(palette[state].r * act, palette[state].g * act, palette[state].b * act, 255));
                        img.setPixel(sf::Vector2u(x, y), palette[state]);
                    else
                        img.setPixel(sf::Vector2u(x, y), sf::Color(0, 0, 0, 255));
                }

            sf::Texture tex(img);

            sf::Sprite s(tex);

            s.setScale(sf::Vector2f(2.0f, 2.0f));

            window.draw(s);

            window.display();
        }

        renderCounter++;
    } while (!quit);

    return 0;
}
