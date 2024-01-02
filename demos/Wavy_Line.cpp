// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "vis/Plot.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

const int numAdditionalStepsAhead = 0;

const float pi = 3.141596f;

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float func(float x) {
    return std::sin(0.025f * pi * x + 0.25f) * std::sin(0.01234f * pi * x - 0.3f) * std::sin(0.0018f * pi * x + 2.0f) * 0.3f;
}

float func_deriv(float x) {
    return std::cos(0.025f * pi * x + 0.25f) * 0.2f * (0.025f * pi);
}

int main(int argc, char *argv[])
{
    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    std::string hFileName = "wavyLine.ohr";

    bool loadHierarchy  = true;
    if (argc > 1) loadHierarchy = atoi(argv[1]);

    bool saveHierarchy  = !loadHierarchy;
    // --------------------------- Create the window(s) ---------------------------

    unsigned int windowWidth = 1000;
    unsigned int windowHeight = 500;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Wavy Test", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(30);

    vis::Plot plot;
    //plot.backgroundColor = sf::Color(64, 64, 64, 255);
    plot.plotXAxisTicks = true;
    plot.curves.resize(1);
    plot.curves[0].shadow = 0.0f; // Input

    float minCurve = -1.25f;
    float maxCurve = 1.25f;

    sf::RenderTexture plotRT;
    plotRT.create(windowWidth, windowHeight, false);
    plotRT.setActive();
    plotRT.clear(sf::Color::White);

    sf::Texture lineGradient;
    lineGradient.loadFromFile("resources/lineGradient.png");

    sf::Font tickFont;
    tickFont.loadFromFile("resources/Hack-Regular.ttf");

    // --------------------------- Create the Hierarchy ---------------------------

    const int maxBufferSize = 10000;

    bool quit = false;
    bool autoplay = true;
    bool spacePressedPrev = false;

    int index = -1;

    int predIndex;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            bool spacePressed = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);

            if (spacePressed && !spacePressedPrev)
                autoplay = !autoplay;

            spacePressedPrev = spacePressed;
        }

        if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            index++;

            if (index % 1000 == 0)
                std::cout << "Step: " << index << std::endl;

            float value = func(index);

            // Plot target data
            vis::Point p;
            p.position.x = index;
            p.position.y = value;
            p.color = sf::Color(192, 32, 32);
            plot.curves[0].points.push_back(p);

            if (plot.curves[0].points.size() > maxBufferSize) {
                plot.curves[0].points.erase(plot.curves[0].points.begin());

                int firstIndex = 0;

                for (std::vector<vis::Point>::iterator it = plot.curves[0].points.begin(); it != plot.curves[0].points.end(); it++, firstIndex++)
                    (*it).position.x = static_cast<float>(firstIndex);
            }

            window.clear();

            plot.draw(
                plotRT, lineGradient, tickFont, 0.5f,
                sf::Vector2f(0.0f, plot.curves[0].points.size()),
                sf::Vector2f(minCurve, maxCurve), sf::Vector2f(48.0f, 48.0f),
                sf::Vector2f(plot.curves[0].points.size() / 10.0f, (maxCurve - minCurve) / 10.0f),
                2.0f, 2.0f, 2.0f, 6.0f, 2.0f, 4
            );

            plotRT.display();

            sf::Sprite plotSprite;
            plotSprite.setTexture(plotRT.getTexture());

            window.draw(plotSprite);

            // Draw segments
            //int numSegments = 100;

            //sf::VertexArray va(sf::Lines, numSegments * 2);
            //
            //for (int i = 0; i < numSegments; i++) {
            //    int j0 = i * 2;
            //    int j1 = j0 + 1;

            //    float x = i / static_cast<float>(numSegments);
            //    int xi = x * plot.curves[0].points.size();
            //    float y = plot.curves[0].points[xi].position.y;

            //    float slope;

            //    if (x == 0)
            //        slope = (plot.curves[0].points[xi + 1].position.y - y) * 32.0f;
            //    else
            //        slope = (y - plot.curves[0].points[xi - 1].position.y) * 32.0f;

            //    sf::Vector2f dir(1.0f, -slope);

            //    float mag = std::sqrt(dir.x * dir.x + dir.y * dir.y);

            //    dir /= mag;

            //    // Perpendicular
            //    dir = sf::Vector2f(dir.y, -dir.x);

            //    va[j0].position.x = x * (1000.0f - 2.0f * 48.0f) + 48.0f;
            //    va[j0].position.y = -y * (500.0f - 2.0f * 48.0f) * 0.4f + 250.0f - 0.0f;
            //    va[j1].position = va[j0].position + dir * 32.0f; 
            //    va[j0].color = sf::Color::Green;
            //    va[j1].color = sf::Color::Green;
            //}

            //window.draw(va);

            window.display();
        }
    } while (!quit);

    return 0;
}

