// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Graphics.hpp>

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/image_encoder.h>

#include "vis/Plot.h"

#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <cmath>

int main() {
    aon::set_num_threads(8);

    // Initialize a random number generator
    std::mt19937 rng(time(nullptr));

    const unsigned int windowWidth = 1200;
    const unsigned int windowHeight = 800;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Video Test", sf::Style::Default);

    // Uncap framerate
    window.setFramerateLimit(120);

    sf::Font font;
    font.loadFromFile("resources/Hack-Regular.ttf");

    sf::Texture line_gradient;
    line_gradient.loadFromFile("resources/lineGradient.png");

    const int num_points = 256;
    vis::Plot plt;
    plt.curves.resize(3);
    plt.curves[0].shadow = 0.0f;
    plt.curves[1].shadow = 0.0f;
    plt.curves[2].shadow = 0.0f;

    bool quit = false;

    int t = 0;
    float velocity = 0.0f;
    float command = 0.0f;
    float smooth_command = 0.0f;
    char dir = 1;

    float stopped_thresh = 0.5f;
    bool stopped_prev = false;

    int size = 32;
    int rad = 16;
    float ratio = 0.03f;

    std::vector<float> grid(size * size);

    for (int i = 0; i < grid.size(); i++)
        grid[i] = aon::randf();

    std::vector<float> igrid(size * size, 0.0f);

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            if (window.hasFocus()) {
                switch (event.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                }
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
                for (int i = 0; i < grid.size(); i++)
                    grid[i] = aon::randf();
                
                for (int x = 0; x < size; x++)
                    for (int y = 0; y < size; y++) {
                        int lower_x = std::max(0, x - rad);
                        int lower_y = std::max(0, y - rad);
                        int upper_x = std::min(size - 1, x + rad);
                        int upper_y = std::min(size - 1, y + rad);

                        int num_higher = 0;
                        int count = (upper_x - lower_x + 1) * (upper_y - lower_y + 1);

                        for (int dx = lower_x; dx <= upper_x; dx++)
                            for (int dy = lower_y; dy <= upper_y; dy++) {
                                if (grid[dy + dx * size] > grid[y + x * size])
                                    num_higher++;
                            }

                        float r = (float)num_higher / (float)count;

                        igrid[y + x * size] = (r < ratio);
                    }
            }

            if (plt.curves[0].points.size() < num_points) {
                plt.curves[0].points.push_back(vis::Point());
                plt.curves[1].points.push_back(vis::Point());
                plt.curves[2].points.push_back(vis::Point());
            }
            else {
                // shift old samples
                for (int p = 0; p < num_points - 1; p++) {
                    plt.curves[0].points[p] = plt.curves[0].points[p + 1];
                    plt.curves[1].points[p] = plt.curves[1].points[p + 1];
                    plt.curves[2].points[p] = plt.curves[2].points[p + 1];
                }
            }

            const float dt = 0.1f;

            command = std::sin(t * 0.03f) * 1.0f + aon::rand_normalf() * 1.0f;

            velocity += (command + aon::rand_normalf() * 0.5f) * dt;
            velocity -= 0.1f * velocity * dt;

            float speed = std::abs(velocity);

            bool stopped = speed < stopped_thresh;

            if (stopped)
                std::cout << "Stopped: " << speed << std::endl;

            smooth_command += 1.0f * dt * (command - smooth_command);

            if (!stopped && stopped_prev) {
                // potential sign change
                dir = command > 0.0f;
            }

            stopped_prev = stopped;

            plt.curves[0].points[plt.curves[0].points.size() - 1].position = sf::Vector2f(t, velocity);
            plt.curves[1].points[plt.curves[0].points.size() - 1].position = sf::Vector2f(t, command);;
            plt.curves[2].points[plt.curves[0].points.size() - 1].position = sf::Vector2f(t, dir);

            plt.curves[0].points[plt.curves[0].points.size() - 1].color = sf::Color::Red;
            plt.curves[1].points[plt.curves[0].points.size() - 1].color = sf::Color::Blue;            
            plt.curves[2].points[plt.curves[0].points.size() - 1].color = sf::Color::Green;

            t++;
        }

        window.clear(sf::Color::Black);

        plt.draw(window, line_gradient, font, 0.5f, sf::Vector2f(t - num_points, t), sf::Vector2f(-10.0f, 10.0f), sf::Vector2f(32, 32), sf::Vector2f(10.0f, 5.0f), 2.0f, 2.0f, 2.0f, 6.0f, 1.0f, 2);

        sf::Image img;
        img.create(size, size);

        for (int y = 0; y < img.getSize().y; y++)
            for (int x = 0; x < img.getSize().x; x++) {
                sf::Color c = (igrid[y + x * size] > 0.0f ? sf::Color::White : sf::Color::Black);

                img.setPixel(x, y, c);
            }

        sf::Texture tex;
        tex.loadFromImage(img);

        sf::Sprite s;
        s.setTexture(tex);

        s.setScale(4.0f, 4.0f);

        window.draw(s);

        window.display();
    } while (!quit);

    return 0;
}
