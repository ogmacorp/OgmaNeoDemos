#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <cmath>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>

//class MiniART {
//public:
//    int num_inputs;
//    int num_hidden;
//    std::vector<float> weights0;
//    std::vector<float> weights1;
//    std::vector<bool> commits;
//    float max_act;
//    float max_match;
//    int state;
//
//    void init(
//        int num_inputs,
//        int num_hidden,
//        std::mt19937 &rng
//    ) {
//        this->num_inputs = num_inputs;
//        this->num_hidden = num_hidden;
//
//        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
//
//        weights0.resize(num_inputs * num_hidden);
//        weights1.resize(weights0.size());
//
//        for (int i = 0; i < weights0.size(); i++) {
//            weights0[i] = dist01(rng);
//            weights1[i] = dist01(rng);
//        }
//
//        commits.resize(num_hidden, false);
//
//        max_act = 0.0f;
//        max_match = 0.0f;
//        state = -1;
//    }
//
//    void step(
//        const std::vector<float> &inputs,
//        bool learn = true
//    ) {
//        int max_index = -1;
//        max_act = 0.0f;
//        max_match = 0.0f;
//
//        for (int hc = 0; hc < num_hidden; hc++) {
//            float sum = 0.0f;
//            float total = 0.0f;
//
//            for (int vc = 0; vc < num_inputs; vc++) {
//                int wi = vc + num_inputs * hc;
//
//                sum += std::min(inputs[vc], weights0[wi]) + std::min(1.0f - inputs[vc], weights1[wi]);
//                total += weights0[wi] + weights1[wi];
//            }
//
//            float match = sum / num_inputs;
//            float act = sum / (0.01f + total);
//
//            if ((!commits[hc] || match >= 0.95f) && act > max_act) {
//                max_act = act;
//                max_match = match;
//                max_index = hc;
//            }
//        }
//
//        state = max_index;
//
//        if (learn && max_index != -1) {
//            if (commits[max_index]) {
//                float lr = 0.5f;
//
//                for (int vc = 0; vc < num_inputs; vc++) {
//                    int wi = vc + num_inputs * max_index;
//
//                    weights0[wi] += lr * std::min(0.0f, inputs[vc] - weights0[wi]);
//                    weights1[wi] += lr * std::min(0.0f, 1.0f - inputs[vc] - weights1[wi]);
//                }
//            }
//            else {
//                for (int vc = 0; vc < num_inputs; vc++) {
//                    int wi = vc + num_inputs * max_index;
//
//                    weights0[wi] = inputs[vc];
//                    weights1[wi] = 1.0f - inputs[vc];
//                }
//
//                commits[max_index] = true;
//            }
//        }
//    }
//};

class MiniFuzzyMinMax {
public:
    int num_inputs;
    int num_hidden;
    std::vector<float> weights0;
    std::vector<float> weights1;
    std::vector<bool> commits;
    float max_act;
    float max_act_complete;
    float max_criterion;
    int state;

    void init(
        int num_inputs,
        int num_hidden,
        std::mt19937 &rng
    ) {
        this->num_inputs = num_inputs;
        this->num_hidden = num_hidden;

        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        weights0.resize(num_inputs * num_hidden);
        weights1.resize(weights0.size());

        for (int i = 0; i < weights0.size(); i++) {
            weights0[i] = dist01(rng);
            weights1[i] = weights0[i];
        }

        commits.resize(num_hidden, false);

        max_act = 0.0f;
        max_criterion = 0.0f;
        state = -1;
    }

    void step(
        const std::vector<float> &inputs,
        bool learn = true
    ) {
        int max_index = -1;
        int max_index_complete = 0;
        max_act = 0.0f;
        max_act_complete = 0.0f;
        max_criterion = 0.0f;

        float oobf = 1.0f;

        for (int hc = 0; hc < num_hidden; hc++) {
            float sum = 0.0f;
            float total = 0.0f;

            for (int vc = 0; vc < num_inputs; vc++) {
                int wi = vc + num_inputs * hc;

                sum += std::max(0.0f, 1.0f - std::max(0.0f, oobf * (weights0[wi] - inputs[vc]))) + std::max(0.0f, 1.0f - std::max(0.0f, oobf * (inputs[vc] - weights1[wi])));
                total += std::max(weights1[wi], inputs[vc]) - std::min(weights0[wi], inputs[vc]);
            }

            float criterion = total / num_inputs;
            float act = sum / (2.0f * num_inputs);

            if ((!commits[hc] || criterion < 0.05f) && act > max_act) {
                max_act = act;
                max_criterion = criterion;
                max_index = hc;
            }

            if (act > max_act_complete) {
                max_act_complete = act;
                max_index_complete = hc;
            }
        }

        state = max_index_complete;

        if (learn && max_index != -1) {
            if (commits[max_index]) {
                float lr = 0.5f;

                for (int vc = 0; vc < num_inputs; vc++) {
                    int wi = vc + num_inputs * max_index;

                    weights0[wi] += lr * std::min(0.0f, inputs[vc] - weights0[wi]);
                    weights1[wi] += lr * std::max(0.0f, inputs[vc] - weights1[wi]);
                }
            }
            else {
                for (int vc = 0; vc < num_inputs; vc++) {
                    int wi = vc + num_inputs * max_index;

                    weights0[wi] = inputs[vc];
                    weights1[wi] = inputs[vc];
                }

                commits[max_index] = true;
            }
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

    sf::Image density("resources/density_image5.png");

    std::vector<float> densities(density.getSize().x * density.getSize().y);
    float total_density = 0.0f;

    for (int x = 0; x < density.getSize().x; x++)
        for (int y = 0; y < density.getSize().y; y++) {
            sf::Color c = density.getPixel(sf::Vector2u(x, y));

            float gray = (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f) * 0.333f;

            densities[y + x * density.getSize().y] = gray;
            total_density += gray;
        }

    MiniFuzzyMinMax a;
    a.init(2, 24, rng);

    std::vector<sf::Color> palette(a.num_hidden);

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
        // sample 
        int sample_index = 0;
        float cusp = dist01(rng) * total_density;
        float sum_so_far = 0.0f;

        for (int i = 0; i < densities.size(); i++) {
            sum_so_far += densities[i];

            if (sum_so_far >= cusp) {
                sample_index = i;
                break;
            }
        }

        int sample_x = sample_index / density.getSize().y;
        int sample_y = sample_index % density.getSize().y;

        float nx = static_cast<float>(sample_x) / density.getSize().x;
        float ny = static_cast<float>(sample_y) / density.getSize().y;

        a.step({ nx, ny }, true);

        if (!speedMode || renderCounter >= 300) {
            window.clear();

            renderCounter = 0;

            for (int x = 0; x < img.getSize().x; x++)
                for (int y = 0; y < img.getSize().y; y++) {
                    a.step({ static_cast<float>(x) / (img.getSize().x - 1), static_cast<float>(y) / (img.getSize().y - 1) }, false);

                    int state = a.state;
                    float act = a.max_act;

                    if (state != -1)
                        img.setPixel(sf::Vector2u(x, y), sf::Color(palette[state].r * act, palette[state].g * act, palette[state].b * act));
                    else
                        img.setPixel(sf::Vector2u(x, y), sf::Color::Black);
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
