#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "constructs/Vec2f.h"
#include "navigraph/GridCells.h"

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <array>
#include <cmath>
#include <random>

std::vector<std::array<unsigned char, 28 * 28>> read_mnist_images(const std::string &full_path) {
    auto rev_int = [](int i) {
        unsigned char c1, c2, c3, c4;

        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    std::vector<std::array<unsigned char, 28 * 28>> dataset;

    if (file.is_open()) {
        int magic_number = 0, n_rows = 0, n_cols = 0;

        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));

        magic_number = rev_int(magic_number);

        std::cout << magic_number << std::endl;

        if (magic_number != 2051)
            throw std::runtime_error("Invalid MNIST image file!");

        int num_images;

        file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        file.read(reinterpret_cast<char*>(&n_rows), sizeof(n_rows));
        file.read(reinterpret_cast<char*>(&n_cols), sizeof(n_cols));
        
        num_images = rev_int(num_images);
        n_rows = rev_int(n_rows);
        n_cols = rev_int(n_cols);

        dataset.resize(num_images);

        for (int i = 0; i < num_images; i++)
            file.read(reinterpret_cast<char*>(&dataset[i][0]), 28 * 28);
    } 
    else
        throw std::runtime_error("Cannot open file `" + full_path + "`!");
    
    return dataset;
}

std::vector<int> read_mnist_labels(const std::string &full_path) {
    auto rev_int = [](int i) {
        unsigned char c1, c2, c3, c4;

        c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;

        return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
    };

    std::ifstream file(full_path, std::ios::binary);

    std::vector<int> dataset;

    if (file.is_open()) {
        int magic_number = 0;

        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));

        magic_number = rev_int(magic_number);

        if (magic_number != 2049)
            throw std::runtime_error("Invalid MNIST label file!");

        int num_labels;

        file.read(reinterpret_cast<char*>(&num_labels), sizeof(num_labels));

        num_labels = rev_int(num_labels);

        dataset.resize(num_labels);

        for (int i = 0; i < num_labels; i++)
            file.read(reinterpret_cast<char*>(&dataset[i]), 1);
    } 
    else
        throw std::runtime_error("Unable to open file `" + full_path + "`!");

    return dataset;
}
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
    //window.setFramerateLimit(60);

    sf::View view = window.getDefaultView();
    view.setCenter(0.0f, 0.0f);
    sf::View newView = view;

    int glimpse_size = 8;

    GridCells gc;
    gc.init_random(glimpse_size * glimpse_size, 32, 32, 6, rng);

    std::vector<float> output_weights(10 * gc.get_states().size());

    std::normal_distribution<float> ndist(0.0f, 1.0f);

    for (int i = 0; i < output_weights.size(); i++)
        output_weights[i] = ndist(rng) * 0.001f;
    
    Vec2f pos(0.0f, 0.0f);
    std::vector<float> features(8, 0.0f);

    std::vector<std::array<unsigned char, 28 * 28>> images = read_mnist_images("resources/train-images-idx3-ubyte");
    std::vector<int> labels = read_mnist_labels("resources/train-labels-idx1-ubyte");

    assert(images.size() == labels.size());

    std::uniform_int_distribution<int> pos_dist(0, 28 - glimpse_size);
    std::uniform_int_distribution<int> image_dist(0, images.size() - 1);

    float avg_acc = 0.0f;

    std::vector<float> inputs(glimpse_size * glimpse_size, 0.0f);

    float avg_time = 0.01f;

    sf::Clock clock;

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

            int x = 14 - glimpse_size / 2;
            int y = 14 - glimpse_size / 2;

            int image_index = image_dist(rng);
            const std::array<unsigned char, 28 * 28> &image = images[image_index];
            int label = labels[image_index];

            gc.clear();

            float speed = 1.0f;

            for (int s = 0; s < 10; s++) {
                int tx = pos_dist(rng);
                int ty = pos_dist(rng);

                int dx = tx - x;
                int dy = ty - y;

                for (int gx = 0; gx < glimpse_size; gx++)
                    for (int gy = 0; gy < glimpse_size; gy++) {
                        inputs[gy + glimpse_size * gx] = image[(y + gy) + 28 * (x + gx)] / 255.0f;
                    }

                gc.step(inputs, dx * speed, dy * speed);

                x = tx;
                y = ty;
            }

            // shift to known location for readout
            for (int gx = 0; gx < glimpse_size; gx++)
                for (int gy = 0; gy < glimpse_size; gy++) {
                    inputs[gy + glimpse_size * gx] = 0.0f;
                }

            gc.step(inputs, ((14 - static_cast<int>(glimpse_size / 2)) - x) * speed, ((14 - static_cast<int>(glimpse_size / 2)) - y) * speed);

            float max_act = -999999.0f;
            int max_index = 0;

            // classify
            for (int c = 0; c < 10; c++) {
                float act = 0.0f;

                for (int i = 0; i < gc.get_states().size(); i++) {
                    act += gc.get_states()[i] * output_weights[i + gc.get_states().size() * c];
                }

                if (act > max_act) {
                    max_act = act;
                    max_index = c;
                }

                act = std::tanh(act * 0.5f) * 0.5f + 0.5f;

                float error = 0.01f * ((c == label) - act);

                for (int i = 0; i < gc.get_states().size(); i++) {
                    output_weights[i + gc.get_states().size() * c] += error * gc.get_states()[i];            
                }
            }

            std::cout << max_index << std::endl;

            avg_acc += 0.01f * ((max_index == label) - avg_acc);

            std::cout << avg_acc << std::endl;
        }

        view.setCenter(view.getCenter() + (newView.getCenter() - view.getCenter()) * viewInterpolateRate * dt);
        view.setSize(view.getSize() + (newView.getSize() - view.getSize()) * viewInterpolateRate * dt);

        window.setView(view);

        window.clear(sf::Color::Black);

        const float render_scale = 4.0f;

        sf::Texture tex;
        tex.loadFromImage(gc.get_states_image());

        sf::Sprite s;
        s.setTexture(tex);
        s.setScale(sf::Vector2f(render_scale, render_scale));

        window.draw(s);

        window.display();
    } while (!quit);

    return 0;
}

