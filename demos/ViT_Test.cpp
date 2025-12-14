#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

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

        }

        view.setCenter(view.getCenter() + (newView.getCenter() - view.getCenter()) * viewInterpolateRate * dt);
        view.setSize(view.getSize() + (newView.getSize() - view.getSize()) * viewInterpolateRate * dt);

        window.setView(view);

        window.clear(sf::Color::Black);

        window.display();
    } while (!quit);

    return 0;
}

