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
#include <thread>
#include <mutex>

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/Helpers.h>
#include <aogmaneo/ImageEncoder.h>
#include <cmath>

#include <vis/Vis3D.h>

using namespace aon;
using namespace cv;

class CustomStreamReader : public aon::StreamReader {
public:
    std::ifstream ins;

    void read(
        void* data,
        int len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::StreamWriter {
public:
    std::ofstream outs;

    void write(
        const void* data,
        int len
    ) override {
        outs.write(static_cast<const char*>(data), len);
    }
};

int main() {
    // Initialize a random number generator
    std::mt19937 rng(time(nullptr));

    std::string encFileName = "resources/donkey.oenc";
    std::string hFileName = "resources/donkey.ohr";

    const unsigned int windowWidth = 1200;
    const unsigned int windowHeight = 800;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Video Test", sf::Style::Default);

    // Uncap framerate
    window.setFramerateLimit(0);

    sf::Font font;
    font.loadFromFile("resources/Hack-Regular.ttf");

    // --------------------------- Create the Hierarchy ---------------------------

    aon::setNumThreads(8);

    // Forward declare
    ImageEncoder imgEnc;
    Hierarchy h;

    bool quit = false;
    bool graph = true;
    bool loadHierarchy = false;
    bool saveHierarchy = true;

    std::cout << "Loading hierarchy from " << hFileName << " and " << encFileName << std::endl;

    {
        imgEnc = ImageEncoder();
        CustomStreamReader reader;
        reader.ins.open(encFileName.c_str(), std::ios::binary);
        int magic;
        reader.ins.read(reinterpret_cast<char*>(&magic), sizeof(int));
        imgEnc.read(reader);
    }

    {
        h = Hierarchy();
        CustomStreamReader reader;
        reader.ins.open(hFileName.c_str(), std::ios::binary);
        int magic;
        reader.ins.read(reinterpret_cast<char*>(&magic), sizeof(int));
        h.read(reader);
    }

    // ---------------------------- Presentation Simulation Loop -----------------------------
    
    Int3 imgSize = imgEnc.getVisibleLayerDesc(0).size;

    window.setVerticalSyncEnabled(true);
    quit = false;

    std::vector<Vis3D::ImgEncDesc> descs(1);

    std::mutex mut;

    Array<const IntBuffer*> inputCIs(2);

    inputCIs[0] = &h.getPredictionCIs(0);
    inputCIs[1] = &h.getPredictionCIs(1);

    std::thread th([&]{
        Vis3D v(900, 1200, "Test");

        while (!quit) {
            mut.lock();
            v.update(inputCIs, h, descs);
            mut.unlock();
            v.render();
        }
    });

    do {
        // ----------------------------- Input -----------------------------

        sf::Event windowEvent;

        while (window.pollEvent(windowEvent)) {
            switch (windowEvent.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
                quit = true;
        }

        window.clear();

        mut.lock();

        h.step(inputCIs, false);

        imgEnc.reconstruct(&h.getPredictionCIs(0));

        ByteBuffer pred = imgEnc.getReconstruction(0);

        descs[0].enc = &imgEnc;
        descs[0].imgs.resize(1);

        descs[0].imgs[0] = pred;

        mut.unlock();

        sf::Image img;

        img.create(imgSize.x, imgSize.y);

        for (int x = 0; x < imgSize.x; x++)
            for (int y = 0; y < imgSize.y; y++) {
                sf::Color c;

                c.r = static_cast<sf::Uint8>(pred[0 + y * 3 + x * 3 * imgSize.y]);
                c.g = static_cast<sf::Uint8>(pred[1 + y * 3 + x * 3 * imgSize.y]);
                c.b = static_cast<sf::Uint8>(pred[2 + y * 3 + x * 3 * imgSize.y]);

                img.setPixel(x, y, c);
            }

        sf::Texture tex;
        tex.loadFromImage(img);

        float scale = std::min(static_cast<float>(window.getSize().x) / img.getSize().x, static_cast<float>(window.getSize().y) / img.getSize().y);

        sf::Sprite s;
        s.setPosition(window.getSize().x * 0.5f, window.getSize().y * 0.5f);
        s.setTexture(tex);
        s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));
        s.setScale(sf::Vector2f(scale, scale));

        window.draw(s);

        window.display();
    } while (!quit);

    quit = true;
    th.join();

    return 0;
}
