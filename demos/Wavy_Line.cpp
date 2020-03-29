// ----------------------------------------------------------// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
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

#if !defined(M_PI)
#define M_PI 3.141596f
#endif

using namespace ogmaneo;

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int main() {
    std::string hFileName = "wavyLine.ohr";

    // --------------------------- Create the window(s) ---------------------------

    unsigned int windowWidth = 1000;
    unsigned int windowHeight = 500;

    sf::RenderWindow renderWindow;

    renderWindow.create(sf::VideoMode(windowWidth, windowHeight), "Wavy Test", sf::Style::Default);

    renderWindow.setVerticalSyncEnabled(false);
    //renderWindow.setFramerateLimit(60);

    vis::Plot plot;
    //plot.backgroundColor = sf::Color(64, 64, 64, 255);
    plot.plotXAxisTicks = false;
    plot.curves.resize(2);
    plot.curves[0].shadow = 0.1f; // Input
    plot.curves[1].shadow = 0.1f; // Predict

    float minCurve = -1.25f;
    float maxCurve = 1.25f;

    sf::RenderTexture plotRT;
    plotRT.create(windowWidth, windowHeight, false);
    plotRT.setActive();
    plotRT.clear(sf::Color::White);

    sf::Texture lineGradient;
    lineGradient.loadFromFile("resources/lineGradient.png");

    sf::Font tickFont;

#if defined(_WINDOWS)
    tickFont.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#elif defined(__APPLE__)
    tickFont.loadFromFile("/Library/Fonts/Courier New.ttf");
#else
    tickFont.loadFromFile("/usr/share/fonts/TTF/VeraMono.ttf");
#endif

    // --------------------------- Create the Hierarchy ---------------------------

    const int inputColumnSize = 32;

    // Create hierarchy
    ComputeSystem::setNumThreads(4);
    ComputeSystem cs;

    std::vector<Hierarchy::LayerDesc> lds(6);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(4, 4, 32);

        lds[i].ffRadius = lds[i].pRadius = 4;

        lds[i].ticksPerUpdate = 2;
        lds[i].temporalHorizon = 4;
    }

    Hierarchy h;
    
    const int maxBufferSize = 300;

    bool quit = false;
    bool autoplay = true;
    bool sPrev = false;
    bool spacePressedPrev = false;

    int index = -1;

    bool loadHierarchy = false;
    bool saveHierarchy = false;

    if (loadHierarchy) {
        std::ifstream fromFile(hFileName);

        h.readFromStream(fromFile);
    }
    else // Random init
        h.initRandom(cs, { Int3(1, 1, inputColumnSize) }, { prediction }, lds);

    do {
        sf::Event event;

        while (renderWindow.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            quit = true;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space) && !spacePressedPrev)
            autoplay = !autoplay;

        spacePressedPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);

        if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            index++;

            if (index % 1000 == 0)
                std::cout << "Step: " << index << std::endl;

            float value = std::sin(0.0125f * M_PI * index + 0.25f) * 
                std::sin(0.03f * M_PI * index + 1.5f) *
                std::sin(0.025f * M_PI * index - 0.1f);

            int predIndex = h.getPredictionCs(0)[0];

            // Un-bin
            float predValue = static_cast<float>(predIndex) / static_cast<float>(inputColumnSize - 1) * 2.0f - 1.0f;

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
                    (*it).position.x = (float)firstIndex;

                plot.curves[1].points.erase(plot.curves[1].points.begin());

                firstIndex = 0;

                for (std::vector<vis::Point>::iterator it = plot.curves[1].points.begin(); it != plot.curves[1].points.end(); it++, firstIndex++)
                    (*it).position.x = (float)firstIndex;
            }

            renderWindow.clear();

            plot.draw(plotRT, lineGradient, tickFont, 0.5f,
                sf::Vector2f(0.0f, plot.curves[0].points.size()),
                sf::Vector2f(minCurve, maxCurve), sf::Vector2f(48.0f, 48.0f),
                sf::Vector2f(plot.curves[0].points.size() / 10.0f, (maxCurve - minCurve) / 10.0f),
                2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);

            plotRT.display();

            sf::Sprite plotSprite;
            plotSprite.setTexture(plotRT.getTexture());

            renderWindow.draw(plotSprite);

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
                h.step(cs, { &h.getPredictionCs(0) }, false);
            }
            else {
                // Bin
                std::vector<int> input = { static_cast<int>((value * 0.5f + 0.5f) * (inputColumnSize - 1) + 0.5f) };

                h.step(cs, { &input }, true);
            }

            renderWindow.display();
        }
    } while (!quit);

    if (saveHierarchy) {
        std::cout << "Saving hierarachy as " << hFileName << std::endl;
        
        std::ofstream toFile(hFileName);

        h.writeToStream(toFile);
    }

    return 0;
}