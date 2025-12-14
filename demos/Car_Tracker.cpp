// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <cmath>

using namespace cv;

const float pi = 3.141592f;

const std::vector<std::string> files = {
    "data/laps2.mkv"
};

float min_angle_delta(float delta) {
    return std::fmod(delta + pi, 2.0f * pi) - pi;
}

float length(const sf::Vector2f &v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

int main() {
    // Initialize a random number generator
    std::mt19937 rng(time(nullptr));

    sf::Vector2u window_size(1280, 720);

    sf::RenderWindow window;

    window.create(sf::VideoMode({window_size.x, window_size.y}), "Car Tracking", sf::Style::Default);

    window.setFramerateLimit(0);

    Mat background;
    imread("resources/background.png", background);

    Mat background_gray;

    cvtColor(background, background_gray, COLOR_BGR2GRAY); 

    sf::Vector2i size;

    bool quit = false;

    for (int f = 0; f < files.size() && !quit; f++) {
        VideoCapture capture(files[f]);

        if (!capture.isOpened()) {
            std::cerr << "Could not open capture: " << files[f] << std::endl;
            return 1;
        }
        else
            std::cout << "Loaded " << files[f] << std::endl;

        int cap_len = static_cast<int>(capture.get(CAP_PROP_FRAME_COUNT) * 1.0f);

        size.x = static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH));
        size.y = static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT));

        Mat frame;
        Mat gray;
        Mat delta;

        sf::VertexArray va(sf::PrimitiveType::LineStrip);

        for (int t = 0; t < cap_len && !quit; t++) {
            while (const std::optional event = window.pollEvent()) {
                if (event->is<sf::Event::Closed>())
                    quit = true;
            }

            if (window.hasFocus()) {
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Q))
                    quit = true;
            }

            window.clear();

            capture >> frame;

            if (frame.empty())
                break;

            cvtColor(frame, gray, COLOR_BGR2GRAY);

            delta = gray - background_gray;

            sf::Vector2f pos(0.0f, 0.0f);
            float div = 0.0f;

            sf::Image img;
            img.resize(sf::Vector2u(size.x, size.y));

            for (int x = 0; x < size.x; x++)
                for (int y = 0; y < size.y; y++) {
                    std::uint8_t r = delta.data[x + y * size.x];

                    img.setPixel(sf::Vector2u(x, y), sf::Color(r, r, r));

                    float w = max(0.0f, r / 255.0f - 0.5f); // weighting

                    pos += w * sf::Vector2f(x, y);
                    div += w;
                }

            if (div > 0.0f) {
                pos /= div;

                sf::Vertex v;
                v.position = pos;
                v.color = sf::Color::Green;
            
                va.append(v);
            }

            sf::Texture tex;
            tex.loadFromImage(img);

            sf::Sprite s(tex);
            window.draw(s);

            //GaussianBlur(delta, delta, Size(5, 5), 0, 0);

            if (va.getVertexCount() > 1) {
                window.draw(va);
            }

            sf::CircleShape cs;
            cs.setOrigin(sf::Vector2f(4.0f, 4.0f));
            cs.setPosition(pos);
            cs.setRadius(4.0f);
            cs.setFillColor(sf::Color::Red);

            window.draw(cs);

            std::cout << pos.x << " " << pos.y << std::endl;

            window.display();
        }

        sf::RenderTexture rt;

        rt.resize(sf::Vector2u(size.x, size.y));

        sf::Texture background_tex;

        background_tex.loadFromFile("resources/background.png");

        sf::Sprite s(background_tex);

        rt.draw(s);

        float average_speed_delayed = 0.0f;
        float average_speed = 0.0f;

        for (int i = 1; i < va.getVertexCount(); i++) {
            float speed = length(va[i].position - va[i - 1].position);

            average_speed += 0.1f * (speed - average_speed);
            average_speed_delayed += 0.1f * (average_speed - average_speed_delayed);

            float accel = average_speed - average_speed_delayed;

            float squash = std::tanh(accel * 0.25f) * 0.5f + 0.5f;

            va[i].color.r = (1.0f - squash) * 255.0f;
            va[i].color.g = squash * 255.0f;
            va[i].color.b = 0;
            va[i].color.a = 127;
        }

        rt.draw(va);

        rt.display();

        rt.getTexture().copyToImage().saveToFile("car_tracking_result" + std::to_string(f) + ".png");
    }

    return 0;
}


