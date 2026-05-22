// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>

#include <cmath>

#include "vis/Plot.h"

using namespace cv;

int main() {
    // Initialize a random number generator
    std::mt19937 rng(time(nullptr));

    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(1920, 1080)), "First Person View", sf::Style::Default, sf::State::Windowed);

    sf::Vector2f center(window.getSize().x * 0.5f, window.getSize().y * 0.5f);

    window.setFramerateLimit(100);

    sf::Font font("resources/Hack-Regular.ttf");

    // Open the video file
    VideoCapture capture("resources/fp_view1.mkv");
    Mat frame;

    if (!capture.isOpened()) {
        std::cerr << "Could not open capture!" << std::endl;
        return 1;
    }

    int captureLength = static_cast<int>(capture.get(CAP_PROP_FRAME_COUNT));

    sf::Texture wheel("resources/wheel.png");
    sf::Texture speedometer("resources/throttle.png");
    wheel.setSmooth(true);
    speedometer.setSmooth(true);
    //vis::Plot plot;
    //plot.curves.resize(2);
    //plot.curves[0]

    bool quit = false;

    do {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                quit = true;
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape))
                quit = true;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::R)) {
                // reset
                capture.set(CAP_PROP_POS_FRAMES, 0.0f);
            }
        }

        window.clear();

        capture >> frame;

        if (!frame.empty()) {
            float throttle = frame.data[0 * 3] / 255.0f * 2.0f - 1.0f;
            float steer = frame.data[1 * 3] / 255.0f * 2.0f - 1.0f;

            // clear away the steering and throttle data
            for (int b = 0; b < 6; b++)
                frame.data[b] = 0;

            // Convert to SFML image
            sf::Image img(sf::Vector2u(frame.cols, frame.rows));

            for (int x = 0; x < img.getSize().x; x++)
                for (int y = 0; y < img.getSize().y; y++) {
                    std::uint8_t r = frame.data[2 + x * 3 + y * 3 * img.getSize().x]; // Reverse order so it's BGR -> RGB

                    img.setPixel(sf::Vector2u(x, y), sf::Color(r, r, r));
                }

            // To SFML texture
            sf::Texture tex(img);
            tex.setSmooth(true);

            const float fp_scale = 2.0f;
            sf::Sprite s(tex);
            s.setScale(sf::Vector2f(fp_scale, fp_scale));
            s.setOrigin(sf::Vector2f(tex.getSize().x, tex.getSize().y) * 0.5f);
            s.setRotation(sf::radians(M_PI));

            s.setPosition(center);
            
            window.draw(s);
            
            sf::Sprite ws(wheel);
            ws.setOrigin(sf::Vector2f(wheel.getSize().x * 0.5f, wheel.getSize().y * 0.5f));
            ws.setPosition(center + sf::Vector2f(-170.0f, 300.0f));
            ws.setRotation(sf::radians(steer * M_PI * 0.5f));
            ws.setScale(sf::Vector2f(0.5f, 0.5f));

            window.draw(ws);

            sf::RectangleShape rs;
            rs.setSize(sf::Vector2f(120.0f, 40.0f));
            rs.setFillColor(sf::Color::White);
            rs.setPosition(center + sf::Vector2f(140.0f, 280.0f));
            rs.setScale(sf::Vector2f(throttle, 1.0f));

            window.draw(rs);

            sf::Sprite ts(speedometer);
            ts.setOrigin(sf::Vector2f(speedometer.getSize().x * 0.5f, speedometer.getSize().y * 0.5f));
            ts.setPosition(center + sf::Vector2f(140.0f, 300.0f));
            ts.setScale(sf::Vector2f(0.5f, 0.5f));

            window.draw(ts);
        }

        window.display();
    } while (!quit);

    return 0;
}
