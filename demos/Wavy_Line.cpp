// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/Helpers.h>

#include <vis/Plot.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

const int numAdditionalStepsAhead = 3;

const float pi = 3.141596f;

using namespace aon;

class BufferReader : public aon::StreamReader {
public:
    int start;
    const std::vector<unsigned char>* buffer;

    BufferReader()
    :
    start(0),
    buffer(nullptr)
    {}

    void read(
        void* data,
        int len
    ) override;
};

class BufferWriter : public aon::StreamWriter {
public:
    int start;

    std::vector<unsigned char> buffer;

    BufferWriter(
        int size
    )
    :
    start(0)
    {
        buffer.resize(size);
    }

    void write(
        const void* data,
        int len
    ) override;
};

void BufferReader::read(void* data, int len) {
    for (int i = 0; i < len; i++)
        static_cast<unsigned char*>(data)[i] = (*buffer)[start + i];

    start += len;
}

void BufferWriter::write(const void* data, int len) {
    assert(buffer.size() >= start + len);

    for (int i = 0; i < len; i++)
        buffer[start + i] = static_cast<const unsigned char*>(data)[i];

    start += len;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int main() {
    std::string hFileName = "wavyLine.ohr";

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

    setNumThreads(8);

    Array<Hierarchy::LayerDesc> lds(8);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(4, 4, 32);
        lds[i].errorSize = Int3(4, 4, 32);

        lds[i].hRadius = 2;
        lds[i].eRadius = 2;
        lds[i].dRadius = 2;
        lds[i].bRadius = 2;
    }

    Array<Hierarchy::IODesc> ioDescs(1);
    ioDescs[0] = Hierarchy::IODesc(Int3(1, 1, inputColumnSize), IOType::prediction, 2, 2, 2, 2, 32);

    Hierarchy h;
    h.initRandom(ioDescs, lds);

    int hStateSize = h.stateSize();

    const int maxBufferSize = 300;

    bool quit = false;
    bool autoplay = true;
    bool spacePressedPrev = false;

    int index = -1;

    bool loadHierarchy = false;
    bool saveHierarchy = false;

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

            Array<const IntBuffer*> inputCIs(1);

            IntBuffer input = IntBuffer(1, static_cast<int>((value - minCurve) / (maxCurve - minCurve) * (inputColumnSize - 1) + 0.5f));

            inputCIs[0] = &input;
            
            if (window.hasFocus() && sf::Keyboard::isKeyPressed(sf::Keyboard::P))
            {
                // Prediction mode
                inputCIs[0] = &h.getPredictionCIs(0);
                h.step(inputCIs, false);
            }
            else {
                // training mode
                h.step(inputCIs, true);
            }
            
            inputCIs[0] = &h.getPredictionCIs(0);

            BufferWriter writer(hStateSize);
            h.writeState(writer);

            for (int step = 0; step < numAdditionalStepsAhead; step++)
                h.step(inputCIs, false);

            predIndex = h.getPredictionCIs(0)[0];

            BufferReader reader;
            reader.buffer = &writer.buffer;
            h.readState(reader);

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
