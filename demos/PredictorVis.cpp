#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/predictor.h>

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>
#include <cmath>
#include <random>

const int S = 8;
const int L = 16;
const int N = S * L;

typedef aon::Vec<S, L> Vec1;
typedef aon::Predictor<S, L> Predictor1;

int main() {
    std::mt19937 rng(time(nullptr));

    const float dt = 0.017f;
    const float zoomRate = 0.4f;
    const float viewInterpolateRate = 20.0f;

    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Gen Demo", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    aon::Layer_Params params;

    int hidden_segments = 8;
    int hidden_length = 16;
    int num_hidden = hidden_segments * hidden_length;

    int num_weights = num_hidden * N * 2;

    Predictor1 pred;
    pred.init_random(hidden_segments, hidden_length);

    // sequence
    std::vector<Vec1> vecs(100);

    for (int i = 0; i < vecs.size(); i++)
        vecs[i] = Vec1::randomized();

    sf::View view = window.getDefaultView();
    view.setCenter(0.0f, 0.0f);
    sf::View newView = view;

    sf::Clock clock;

    int t = 0;

    const float step_time = 0.1f;
    float step_timer = 0.0f;

    bool quit = false;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            if (window.hasFocus()) {
                switch (event.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                case sf::Event::MouseWheelMoved:
                    int dWheel = event.mouseWheel.delta;

                    newView = view;

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta0 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    newView.setSize(view.getSize() + newView.getSize() * (-zoomRate * dWheel));

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta1 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    window.setView(view);

                    newView.setCenter(view.getCenter() + mouseZoomDelta0 - mouseZoomDelta1);

                    break;
                }
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            if (step_timer > step_time) {
                step_timer = 0.0f;

                t = (t + 1) % vecs.size();
            }

            Vec1 res = pred.step(vecs[t], vecs[t], true, params);

            std::cout << (int)res[0] << std::endl;

            step_timer += dt;
        }

        view.setCenter(view.getCenter() + (newView.getCenter() - view.getCenter()) * viewInterpolateRate * dt);
        view.setSize(view.getSize() + (newView.getSize() - view.getSize()) * viewInterpolateRate * dt);

        window.setView(view);

        window.clear(sf::Color::Black);

        float radius = 4.0f;
        sf::CircleShape cs;

        cs.setRadius(radius);
        cs.setOutlineThickness(1.0f);
        cs.setOrigin(radius, radius);
        cs.setPointCount(8);

        const aon::Byte_Buffer &weights = pred.get_weights_encode();

        const float gap = 9.0f;

        for (int hs = 0; hs < hidden_segments; hs++) {
            for (int hl = 0; hl < hidden_length; hl++) {
                int hi = hl + hidden_length * hs;

                for (int i = 0; i < 1; i++) {
                    for (int s = 0; s < S; s++) {
                        for (int l = 0; l < L; l++) {
                            int iindex = l + L * s + i * N;

                            int wi = hi + num_hidden * iindex;

                            int byi = wi / 8;
                            int bi = wi % 8;

                            bool w = ((weights[byi] & (1 << bi)) != 0);

                            const Vec1 &v = (i == 0 ? vecs[t] : vecs[(t + 1) % vecs.size()]);

                            bool is_input = v[s] == l;

                            cs.setFillColor(w ? sf::Color::Red : sf::Color::Black);
                            cs.setOutlineColor(is_input ? sf::Color::Green : sf::Color::White);

                            cs.setPosition(gap * sf::Vector2f(s + hs * S * 1.5f, l + L * i * 1.5f + hl * L * 2 * 1.5f));

                            window.draw(cs);
                        }
                    }
                }
            }
        }

        window.display();
    } while (!quit);

    return 0;
}

