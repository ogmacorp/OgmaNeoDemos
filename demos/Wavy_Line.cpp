// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <ogmaneo/Hierarchy.h>

#include <vis/Plot.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

const int numAdditionalStepsAhead = 0;

const float pi = 3.141596f;

using namespace ogmaneo;

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
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

    window.setVerticalSyncEnabled(false);
    //window.setFramerateLimit(60);

    vis::Plot plot;
    //plot.backgroundColor = sf::Color(64, 64, 64, 255);
    plot.plotXAxisTicks = false;
    plot.curves.resize(2);
    plot.curves[0].shadow = 0.0f; // Input
    plot.curves[1].shadow = 0.0f; // Prediction

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

    const int inputColumnSize = 32;

    ComputeSystem cs;
    cs.setNumThreads(8);

    std::vector<Hierarchy::LayerDesc> lds(7);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(4, 4, 32);
    }

    Hierarchy h;
    bool learnFlag = true;

    h.initRandom(cs, { Int3(1, 1, inputColumnSize) }, { prediction }, lds);

    const int maxBufferSize = 300;

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

            float value = std::sin(0.0125f * pi * index + 0.25f) * 
            std::sin(0.03f * pi * index + 1.5f) *
            std::sin(0.025f * pi * index - 0.1f);

            if (dist01(rng) < 0.004f) {
                std::uniform_int_distribution<int> indexDist(0, 1000);
                index = indexDist(rng);
            }

            std::vector<int> input = { static_cast<int>((value - minCurve) / (maxCurve - minCurve) * (inputColumnSize - 1) + 0.5f) };

            if (!learnFlag || (window.hasFocus() && sf::Keyboard::isKeyPressed(sf::Keyboard::P)))
            {
                // Prediction mode
                //inputCIs[0] = &h.getPredictionCIs(0);
                h.step(cs, { &input }, false);
            }
            else {
                // training mode
                h.step(cs, { &input }, true);
            }
            
            predIndex = h.getPredictionCs(0)[0];

            // Un-bin
            float predValue = static_cast<float>(predIndex) / static_cast<float>(inputColumnSize - 1) * (maxCurve - minCurve) + minCurve;

            // Plot target data
            vis::Point p;
            p.position.x = index;
            p.position.y = value;
            p.color = sf::Color::Red;
            plot.curves[0].points.push_back(p);

            // Plot predicted data
            vis::Point p1;
            p1.position.x = index;
            p1.position.y = predValue;
            p1.color = sf::Color::Blue;
            plot.curves[1].points.push_back(p1);

            if (plot.curves[0].points.size() > maxBufferSize) {
                plot.curves[0].points.erase(plot.curves[0].points.begin());

                int firstIndex = 0;

                for (std::vector<vis::Point>::iterator it = plot.curves[0].points.begin(); it != plot.curves[0].points.end(); it++, firstIndex++)
                    (*it).position.x = static_cast<float>(firstIndex);

                plot.curves[1].points.erase(plot.curves[1].points.begin());

                firstIndex = 0;

                for (std::vector<vis::Point>::iterator it = plot.curves[1].points.begin(); it != plot.curves[1].points.end(); it++, firstIndex++)
                    (*it).position.x = static_cast<float>(firstIndex);
            }

            window.clear();

            plot.draw(
                plotRT, lineGradient, tickFont, 0.5f,
                sf::Vector2f(0.0f, plot.curves[0].points.size()),
                sf::Vector2f(minCurve, maxCurve), sf::Vector2f(48.0f, 48.0f),
                sf::Vector2f(plot.curves[0].points.size() / 10.0f, (maxCurve - minCurve) / 10.0f),
                2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4
            );

            plotRT.display();

            sf::Sprite plotSprite;
            plotSprite.setTexture(plotRT.getTexture());

            window.draw(plotSprite);

            window.display();
        }
    } while (!quit);

    quit = false;
    autoplay = true;
    spacePressedPrev = false;

    index = -1;

    sf::sleep(sf::seconds(2.0f));

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

            float value = std::sin(0.0125f * pi * index + 0.25f) * 
            std::sin(0.03f * pi * index + 1.5f) *
            std::sin(0.025f * pi * index - 0.1f);

            //if (dist01(rng) < 0.004f) {
            //    std::uniform_int_distribution<int> indexDist(0, 1000);
            //    index = indexDist(rng);
            //}

            std::vector<int> input = { static_cast<int>((value - minCurve) / (maxCurve - minCurve) * (inputColumnSize - 1) + 0.5f) };

            if (!learnFlag || (window.hasFocus() && sf::Keyboard::isKeyPressed(sf::Keyboard::P)))
            {
                // Prediction mode
                //inputCIs[0] = &h.getPredictionCIs(0);
                h.step(cs, { &input }, false);
            }
            else {
                // training mode
                h.step(cs, { &input }, true);
            }
            
            predIndex = h.getPredictionCs(0)[0];

            // Un-bin
            float predValue = static_cast<float>(predIndex) / static_cast<float>(inputColumnSize - 1) * (maxCurve - minCurve) + minCurve;

            // Plot target data
            vis::Point p;
            p.position.x = index;
            p.position.y = value;
            p.color = sf::Color::Red;
            plot.curves[0].points.push_back(p);

            // Plot predicted data
            vis::Point p1;
            p1.position.x = index;
            p1.position.y = predValue;
            p1.color = sf::Color::Blue;
            plot.curves[1].points.push_back(p1);

            if (plot.curves[0].points.size() > maxBufferSize) {
                plot.curves[0].points.erase(plot.curves[0].points.begin());

                int firstIndex = 0;

                for (std::vector<vis::Point>::iterator it = plot.curves[0].points.begin(); it != plot.curves[0].points.end(); it++, firstIndex++)
                    (*it).position.x = static_cast<float>(firstIndex);

                plot.curves[1].points.erase(plot.curves[1].points.begin());

                firstIndex = 0;

                for (std::vector<vis::Point>::iterator it = plot.curves[1].points.begin(); it != plot.curves[1].points.end(); it++, firstIndex++)
                    (*it).position.x = static_cast<float>(firstIndex);
            }

            window.clear();

            plot.draw(
                plotRT, lineGradient, tickFont, 0.5f,
                sf::Vector2f(0.0f, plot.curves[0].points.size()),
                sf::Vector2f(minCurve, maxCurve), sf::Vector2f(48.0f, 48.0f),
                sf::Vector2f(plot.curves[0].points.size() / 10.0f, (maxCurve - minCurve) / 10.0f),
                2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4
            );

            plotRT.display();

            sf::Sprite plotSprite;
            plotSprite.setTexture(plotRT.getTexture());

            window.draw(plotSprite);

            window.display();
        }
    } while (!quit);
    return 0;
}

