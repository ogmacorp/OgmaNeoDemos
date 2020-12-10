// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <Box2D/Box2D.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "constructs/Vec3f.h"
#include "constructs/Matrix4x4f.h"

#include <time.h>
#include <iostream>
#include <random>

#include <aogmaneo/ImageEncoder.h>
#include <aogmaneo/Hierarchy.h>

const int subImgWidth = 8;
const int subImgHeight = 8;
const int gridRes = 64;
const int moveRes = 64;

const bool hasVision = true;

const int showScale = 4;

const float pixelsPerUnit = 4.0f;
const float unitsPerPixel = 1.0f / pixelsPerUnit;

const float predMix = 0.7f;

const float imuNoise = 0.04f;

using namespace aon;

class GridCells {
public:
    struct Layer {
        Vec3f pos;

    Layer()
            : pos(0.0f, 0.0f, 0.0f)
        {}
    };

    float scaleFactor;

    std::vector<Layer> layers;

    GridCells()
        : scaleFactor(0.25f)
    {}

    void init(int numLayers) {
        layers.resize(numLayers);
    }

    Vec3f getFinalPos() const {
        Vec3f pos(0.0f, 0.0f, 0.0f);

        float scale = 1.0f;

        for (int i = 0; i < layers.size(); i++) {
            pos += layers[i].pos / scale;
            scale *= scaleFactor;
        }

        return pos;
    }

    aon::IntBuffer getCIs(int res) const {
        aon::IntBuffer cis(layers.size() * 3);

        for (int i = 0; i < layers.size(); i++) {
            int x = (layers[i].pos.x * 0.5f + 0.5f) * (res - 1) + 0.5f; 
            int y = (layers[i].pos.y * 0.5f + 0.5f) * (res - 1) + 0.5f; 
            int z = (layers[i].pos.z * 0.5f + 0.5f) * (res - 1) + 0.5f; 

            cis[i * 3 + 0] = x;
            cis[i * 3 + 1] = y;
            cis[i * 3 + 2] = z;
        }

        return cis;
    }

    void move(const Vec3f &delta) {
        float scale = 1.0f;

        for (int i = 0; i < layers.size(); i++) {
            layers[i].pos += scale * delta;

            layers[i].pos.x = std::fmod(layers[i].pos.x, 1.0f);
            layers[i].pos.y = std::fmod(layers[i].pos.y, 1.0f);
            layers[i].pos.z = std::fmod(layers[i].pos.z, 1.0f);

            scale *= scaleFactor;
        }
    }
};

FloatBuffer getSubImage(const sf::Image &img, sf::Vector2i pos, sf::Vector2i size) {
    FloatBuffer imgf(size.x * size.y * 3);

    for (int dx = 0; dx < size.x; dx++)
        for (int dy = 0; dy < size.y; dy++) {
            int x = pos.x + dx;
            int y = pos.y + dy;

            if (x < 0 || y < 0 || x >= img.getSize().x || y >= img.getSize().y) {
                imgf[address3(Int3(dx, dy, 0), Int3(size.x, size.y, 3))] = 0.0f;
                imgf[address3(Int3(dx, dy, 1), Int3(size.x, size.y, 3))] = 0.0f;
                imgf[address3(Int3(dx, dy, 2), Int3(size.x, size.y, 3))] = 0.0f;
            }
            else {
                sf::Color pixel = img.getPixel(x, y);

                imgf[address3(Int3(dx, dy, 0), Int3(size.x, size.y, 3))] = pixel.r / 255.0f;
                imgf[address3(Int3(dx, dy, 1), Int3(size.x, size.y, 3))] = pixel.g / 255.0f;
                imgf[address3(Int3(dx, dy, 2), Int3(size.x, size.y, 3))] = pixel.b / 255.0f;
            }
        }

    return imgf;
}

int main() {
    std::mt19937 rng(time(nullptr));

    setNumThreads(8);

    sf::Image img;

    img.loadFromFile("../resources/navi.png");

    sf::RenderWindow window;

    window.create(sf::VideoMode(img.getSize().x * showScale, img.getSize().y * showScale), "Grid Cells", sf::Style::Default);

    window.setFramerateLimit(60);

    sf::Font font;
    font.loadFromFile("../resources/Hack-Regular.ttf");

    GridCells gcs;

    gcs.init(4);

    sf::Texture tex;
    tex.loadFromImage(img);

    Int3 hiddenSize(4, 4, 32);

    Array<ImageEncoder::VisibleLayerDesc> vlds(1);
    vlds[0].size = Int3(subImgWidth, subImgHeight, 3);
    vlds[0].radius = 3;

    ImageEncoder enc;
    enc.initRandom(hiddenSize, vlds);


    for (float f = -10.0f; f <= 10.0f; f += 0.1f) {
        std::cout << aon::tanh(f) << std::endl;
    }

    // Create hierarchy
    Array<Hierarchy::LayerDesc> lds(3);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(4, 4, 32);

        lds[i].ffRadius = lds[i].pRadius = 2;

        lds[i].ticksPerUpdate = 2;
        lds[i].temporalHorizon = 2;
    }

    Hierarchy h;

    Array<Hierarchy::IODesc> ioDescs(hasVision ? 3 : 2);
    ioDescs[0] = Hierarchy::IODesc(Int3(1, 3, moveRes), IOType::prediction, 2, 2, 2, 32);
    ioDescs[1] = Hierarchy::IODesc(Int3(3, gcs.layers.size(), gridRes), IOType::prediction, 2, 2, 2, 32);

    if (hasVision)
        ioDescs[2] = Hierarchy::IODesc(hiddenSize, IOType::prediction, 2, 2, 2, 32);

    IntBuffer gcsBuf = gcs.getCIs(gridRes);
    IntBuffer moveBuf(3);
    moveBuf.fill(moveRes / 2);

    Array<const IntBuffer*> inputs(hasVision ? 3 : 2);
    inputs[0] = &moveBuf;
    inputs[1] = &gcsBuf;

    if (hasVision)
        inputs[2] = &enc.getHiddenCIs();

    h.initRandom(ioDescs, lds);

    Vec3f actualPosOffset(img.getSize().x * unitsPerPixel * 0.5f, img.getSize().y * unitsPerPixel * 0.5f, 0.0f);
    Vec3f actualPos(0.0f, 0.0f, 0.0f);
    Vec3f actualDelta(0.0f, 0.0f, 0.0f);

    float avgError = 0.0f;
    float avgError2 = 0.0f;

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    bool trainMode = false;
    bool showMode = true;

    bool tPressedPrev = false;
    bool sPressedPrev = false;

    do {
        // ----------------------------- Input -----------------------------

        // Receive events
        sf::Event windowEvent;

        while (window.pollEvent(windowEvent)) {
            switch (windowEvent.type)
            {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            bool tPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::T);

            if (tPressed && !tPressedPrev)
                trainMode = !trainMode;
        
            tPressedPrev = tPressed; 

            bool sPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::S);

            if (sPressed && !sPressedPrev)
                showMode = !showMode;
        
            sPressedPrev = sPressed; 
        }

        int subIters = trainMode ? 100 : 1;

        std::normal_distribution<float> noiseDist(0.0f, 1.0f);

        for (int ss = 0; ss < subIters; ss++) {
            Vec3f prevFinalPos = gcs.getFinalPos();

            // Move randomly
            Vec3f targetDelta(noiseDist(rng), noiseDist(rng), 0.0f);

            actualDelta += 0.04f * (targetDelta * 3.0f - actualDelta);

            Vec3f oldPos = actualPos;

            actualPos += actualDelta;

            if (actualPos.x < -static_cast<float>(img.getSize().x) * unitsPerPixel * 0.5f)
                actualPos.x = -static_cast<float>(img.getSize().x) * unitsPerPixel * 0.5f;

            if (actualPos.y < -static_cast<float>(img.getSize().y) * unitsPerPixel * 0.5f)
                actualPos.y = -static_cast<float>(img.getSize().y) * unitsPerPixel * 0.5f;

            if (actualPos.x > static_cast<float>(img.getSize().x) * unitsPerPixel * 0.5f)
                actualPos.x = static_cast<float>(img.getSize().x) * unitsPerPixel * 0.5f;
            
            if (actualPos.y > static_cast<float>(img.getSize().y) * unitsPerPixel * 0.5f)
                actualPos.y = static_cast<float>(img.getSize().y) * unitsPerPixel * 0.5f;

            actualDelta = actualPos - oldPos;

            // Noisy IMU measurement
            Vec3f imuDelta = actualDelta + Vec3f(noiseDist(rng) * imuNoise, noiseDist(rng) * imuNoise, 0.0f);

            // Merge predictions of delta
            Vec3f predDelta(h.getPredictionCIs(0)[0] / static_cast<float>(moveRes - 1) * 2.0f - 1.0f, h.getPredictionCIs(0)[1] / static_cast<float>(moveRes - 1) * 2.0f - 1.0f, 0.0f);

            Vec3f useDelta = predMix * predDelta + (1.0f - predMix) * imuDelta;

            gcs.move(useDelta);

            // Get image
            FloatBuffer imgf = getSubImage(img, sf::Vector2i((actualPos.x + actualPosOffset.x) * pixelsPerUnit, (actualPos.y + actualPosOffset.y) * pixelsPerUnit), sf::Vector2i(subImgWidth, subImgHeight)); 

            Array<const FloatBuffer*> imgs(1);
            imgs[0] = &imgf;

            //enc.step(imgs, true);

            gcsBuf = gcs.getCIs(gridRes);

            moveBuf[0] = min(1.0f, max(0.0f, useDelta.x * 0.5f + 0.5f)) * (moveRes - 1) + 0.5f;
            moveBuf[1] = min(1.0f, max(0.0f, useDelta.y * 0.5f + 0.5f)) * (moveRes - 1) + 0.5f;
            moveBuf[2] = min(1.0f, max(0.0f, useDelta.z * 0.5f + 0.5f)) * (moveRes - 1) + 0.5f;

            h.step(inputs, true);

            float err = -(actualDelta - predDelta).magnitude();
            float err2 = -(actualDelta - imuDelta).magnitude();
            //float err = -(actualDelta - resDelta).magnitude();

            avgError = 0.999f * avgError + 0.001f * err;
            avgError2 = 0.999f * avgError2 + 0.001f * err2;

            std::cout << avgError << " " << avgError2 << std::endl;
        }

        // Show on main window
        window.clear();

        sf::Sprite s;
        s.setTexture(tex);
        s.setScale(showScale, showScale);
        window.draw(s);

        sf::RectangleShape rs;
        rs.setPosition(sf::Vector2f((actualPos.x + actualPosOffset.x) * pixelsPerUnit * showScale, (actualPos.y + actualPosOffset.y) * pixelsPerUnit * showScale));
        rs.setSize(sf::Vector2f(subImgWidth * showScale, subImgHeight * showScale));
        rs.setOutlineThickness(2.0f);
        rs.setOutlineColor(sf::Color::White);
        rs.setFillColor(sf::Color::Transparent);
        window.draw(rs);

        window.display();
    } while (!quit);

    return 0;
}
