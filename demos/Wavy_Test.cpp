// ----------------------------------------------------------// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <neo/Architect.h>
#include <neo/Hierarchy.h>
#include <neo/ScalarEncoder.h>

#include <vis/Plot.h>
#include <vis/DebugWindow.h>

#include <fstream>
#include <sstream>
#include <iostream>

#if !defined(M_PI)
#define M_PI 3.141596f
#endif

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int main() {
    std::mt19937 generator(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    // --------------------------- Create the window(s) ---------------------------

    unsigned int windowWidth = 1000;
    unsigned int windowHeight = 500;

    sf::RenderWindow renderWindow;

    renderWindow.create(sf::VideoMode(windowWidth, windowHeight), "Wavy Test", sf::Style::Default);

    renderWindow.setVerticalSyncEnabled(false);
    //renderWindow.setFramerateLimit(60);

    bool enableDebugWindow = false;
    vis::DebugWindow debugWindow;

    if (enableDebugWindow) {
        renderWindow.setPosition(sf::Vector2i(64, 64));

        debugWindow.create(
            sf::String("Debug"),
            sf::Vector2i(renderWindow.getPosition().x + windowWidth + 8, renderWindow.getPosition().y),
            sf::Vector2u(windowHeight, windowHeight));
    }

    vis::Plot plot;
    //plot._backgroundColor = sf::Color(64, 64, 64, 255);
    plot._plotXAxisTicks = false;
    plot._curves.resize(2);
    plot._curves[0]._shadow = 0.1f;	// input
    plot._curves[1]._shadow = 0.1f;	// predict

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
    tickFont.loadFromFile("/usr/share/fonts/truetype/freefont/FreeMono.ttf");
#endif


    // --------------------------- Create the Hierarchy ---------------------------

    std::shared_ptr<ogmaneo::Resources> res = std::make_shared<ogmaneo::Resources>();

    res->create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::Architect arch;
    arch.initialize(1234, res);

    ogmaneo::ScalarEncoder se;
    int encoderOutputWidth = 16;
    se.createRandom(1, encoderOutputWidth*encoderOutputWidth, -0.999f, 0.999f, 1234);

    ogmaneo::ValueField2D inputField(ogmaneo::Vec2i(encoderOutputWidth, encoderOutputWidth), 0.0f);
    ogmaneo::ValueField2D inputNoisedField(ogmaneo::Vec2i(encoderOutputWidth, encoderOutputWidth), 0.0f);

    arch.addInputLayer(ogmaneo::Vec2i(encoderOutputWidth, encoderOutputWidth));

    int chunkLayers = 3;
    int delayLayers = 3;

    for (int l = 0; l < chunkLayers; l++)
        arch.addHigherLayer(ogmaneo::Vec2i(60, 60), ogmaneo::_chunk);

    for (int l = 0; l < delayLayers; l++)
        arch.addHigherLayer(ogmaneo::Vec2i(60, 60), ogmaneo::_delay);

    ogmaneo::ValueField2D prediction(ogmaneo::Vec2i(encoderOutputWidth, encoderOutputWidth), 0.0f);

    // Generate the hierarchy
    std::shared_ptr<ogmaneo::Hierarchy> h = arch.generateHierarchy();

    if (enableDebugWindow) {
        //debugWindow.catchUnitRanges(false);
        debugWindow.registerHierarchy(res, h);
    }

    std::vector<sf::Texture> layerTextures(h->getPredictor().getNumPredLayers());

    const int maxBufferSize = 200;

    bool quit = false;
    bool autoplay = true;
    bool sPrev = false;

    int index = -1;

    bool saveHierarchy = false;

    bool reloadHierarchy = false;
    if (reloadHierarchy)
        h->load(*res->getComputeSystem(), "Wavy_Test.ohr");

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

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
            autoplay = false;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::C))
            autoplay = true;

        sPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::S);

        if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            index++;

            if (index % 1000 == 0)
                std::cout << "Step: " << index << std::endl;

            float value =
                0.5f * (std::sin(0.164f * M_PI * index + 0.25f) +
                    0.7f * std::sin(0.12352f * M_PI * index * 1.5f + 0.2154f) +
                    0.5f * std::sin(0.0612f * M_PI * index * 3.0f - 0.2112f));

            se.encode(std::vector<float>(1, value), 0.1f, 0.0f, 0.0f);

            std::vector<float> outputNoised = se.getEncoderOutputs();

            for (int i = 0; i < outputNoised.size(); i++)
                if (dist01(generator) < 0.001f)
                    outputNoised[i] = 1.0f - outputNoised[i];

            prediction = h->getPredictions()[0];

            std::vector<float> result(encoderOutputWidth*encoderOutputWidth);

            for (size_t i = 0; i < se.getEncoderOutputs().size(); i++) {
                result[i] = std::min(1.0f, std::max(0.0f, prediction.getData()[i]));
            }

            se.decode(result);

            float v = se.getDecoderOutputs()[0];

            // Plot target data
            vis::Point p;
            p._position.x = (float)index;
            p._position.y = (float)value;
            p._color = sf::Color::Red;
            plot._curves[0]._points.push_back(p);

            // Plot predicted data
            vis::Point p1;
            p1._position.x = (float)index;
            p1._position.y = (float)v;
            p1._color = sf::Color::Blue;
            plot._curves[1]._points.push_back(p1);

            if (plot._curves[0]._points.size() > maxBufferSize) {
                plot._curves[0]._points.erase(plot._curves[0]._points.begin());

                int firstIndex = 0;

                for (std::vector<vis::Point>::iterator it = plot._curves[0]._points.begin(); it != plot._curves[0]._points.end(); ++it, ++firstIndex)
                    (*it)._position.x = (float)firstIndex;

                plot._curves[1]._points.erase(plot._curves[1]._points.begin());

                firstIndex = 0;

                for (std::vector<vis::Point>::iterator it = plot._curves[1]._points.begin(); it != plot._curves[1]._points.end(); ++it, ++firstIndex)
                    (*it)._position.x = (float)firstIndex;
            }

            renderWindow.clear();

            plot.draw(plotRT, lineGradient, tickFont, 0.5f,
                sf::Vector2f(0.0f, plot._curves[0]._points.size()),
                sf::Vector2f(minCurve, maxCurve), sf::Vector2f(48.0f, 48.0f),
                sf::Vector2f(plot._curves[0]._points.size() / 10.0f, (maxCurve - minCurve) / 10.0f),
                2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);

            plotRT.display();

            sf::Sprite plotSprite;
            plotSprite.setTexture(plotRT.getTexture());

            renderWindow.draw(plotSprite);

            if (!sf::Keyboard::isKeyPressed(sf::Keyboard::T)) {
                for (size_t i = 0; i < se.getEncoderOutputs().size(); i++) {
                    inputField.getData()[i] = se.getEncoderOutputs().data()[i];
                    inputNoisedField.getData()[i] = outputNoised[i];
                }

                std::vector<ogmaneo::ValueField2D> inputVector = { inputField };
                std::vector<ogmaneo::ValueField2D> inputNoisedVector = { inputNoisedField };
                h->simStep(inputVector, inputNoisedVector, true);
            }
            else {
                prediction = h->getPredictions()[0];

                for (size_t i = 0; i < se.getEncoderOutputs().size(); i++) {
                    inputField.getData()[i] = prediction.getData()[i];
                }

                std::vector<ogmaneo::ValueField2D> inputVector = { inputField };
                h->simStep(inputVector, false);
            }

#if 0
            float xOffset = 0.0f;
            float scale = 2.0f;

            for (int l = 0; l < layerTextures.size(); l++) {
                std::vector<float> data(h->getPredictor().getPredLayer(l).getHiddenSize().x * h->getPredictor().getPredLayer(l).getHiddenSize().y);

                res->getComputeSystem()->getQueue().enqueueReadImage(
                    h->getPredictor().getPredLayer(l).getHiddenStates()[ogmaneo::_back],
                    CL_TRUE, { 0, 0, 0 },
                    { static_cast<cl::size_type>(h->getPredictor().getPredLayer(l).getHiddenSize().x), static_cast<cl::size_type>(h->getPredictor().getPredLayer(l).getHiddenSize().y), 1 },
                    0, 0, data.data());

                sf::Image img;

                img.create(h->getPredictor().getPredLayer(l).getHiddenSize().x, h->getPredictor().getPredLayer(l).getHiddenSize().y);

                for (unsigned int x = 0; x < img.getSize().x; x++)
                    for (unsigned int y = 0; y < img.getSize().y; y++) {
                        sf::Color c = sf::Color::White;

                        c.r = c.b = c.g = static_cast<sf::Uint8>(255.0f * sigmoid(2.0f * (data[(x + y * img.getSize().x)])));

                        img.setPixel(x, y, c);
                    }

                layerTextures[l].loadFromImage(img);

                sf::Sprite s;

                s.setTexture(layerTextures[l]);

                s.setPosition(xOffset, renderWindow.getSize().y - img.getSize().y * scale);

                s.setScale(scale, scale);

                renderWindow.draw(s);

                xOffset += img.getSize().x * scale + 4.f;
            }
#endif

            renderWindow.display();

            if (enableDebugWindow) {
                debugWindow.update();
                debugWindow.display();
            }
        }
    } while (!quit);

    if (saveHierarchy) {
        std::cout << "Saving hierarachy" << std::endl;
        h->save(*res->getComputeSystem(), "Wavy_Test.ohr");

        // Optionally save the Architect (for use with FB_Parser)
        //arch.save("Wavy_Test.oar");
    }

    return 0;
}