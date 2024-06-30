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

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/helpers.h>
#include <aogmaneo/image_encoder.h>
#include <cmath>

using namespace aon;
using namespace cv;

class CustomStreamReader : public aon::Stream_Reader {
public:
    std::ifstream ins;

    void read(
        void* data,
        long len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        long len
    ) override {
        outs.write(static_cast<const char*>(data), len);
    }
};

class LoopSOM1D {
private:
    int num_inputs;
    int num_cells;

    std::vector<float> dists;
    std::vector<float> protos;

public:
    float lr;
    float falloff;
    float decay;
    float jump_max_ratio;
    int t;

    LoopSOM1D()
    :
    lr(1.0f),
    falloff(0.01f),
    decay(0.0001f),
    jump_max_ratio(0.1f),
    t(0)
    {}

    void init(int num_inputs, int num_cells, std::mt19937 &rng) {
        this->num_inputs = num_inputs;
        this->num_cells = num_cells;

        protos.resize(num_cells * num_inputs);

        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        for (int i = 0; i < protos.size(); i++)
            protos[i] = dist01(rng);

        dists.resize(num_cells);
    }

    int step(const std::vector<float> &inputs, bool learn_enabled = true) {
        int max_jump = jump_max_ratio * num_cells;

        // find bmu
        int bmu = 0;
        float min_dist = 999999.0f;

        for (int i = 0; i < num_cells; i++) {
            float dist = 0.0f;

            for (int j = 0; j < num_inputs; j++) {
                float diff = inputs[j] - protos[j + i * num_inputs];

                dist += diff * diff;
            }

            dists[i] = dist;

            if (dist < min_dist) {
                min_dist = dist;
                bmu = i;
            }
        }

        int delta_t = std::min(((t - bmu) + num_cells) % num_cells, ((bmu - t) + num_cells) % num_cells); // looping delta

        if (delta_t > max_jump)
            t = bmu;

        if (learn_enabled) {
            for (int i = 0; i < num_cells; i++) {
                int delta = std::min(((i - t) + num_cells) % num_cells, ((t - i) + num_cells) % num_cells); // looping delta

                float rate = lr * std::exp(-falloff * delta * delta / std::max(0.0001f, lr));

                for (int j = 0; j < num_inputs; j++) {
                    float diff = inputs[j] - protos[j + i * num_inputs];

                    protos[j + i * num_inputs] += rate * diff;
                }
            }
            
            lr *= 1.0f - decay;

            t = (t + 1) % num_cells;
        }

        return t;
    }
};

int main() {
    const std::string fileName = "resources/track.avi";

    // Initialize a random number generator
    std::mt19937 rng(time(nullptr));

    const unsigned int windowWidth = 1200;
    const unsigned int windowHeight = 800;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Video Test", sf::Style::Default);

    // Uncap framerate
    window.setFramerateLimit(60);

    sf::Font font;
    font.loadFromFile("resources/Hack-Regular.ttf");

    // Open the video file
    VideoCapture capture(fileName);
    Mat frame;

    if (!capture.isOpened()) {
        std::cerr << "Could not open capture: " << fileName << std::endl;
        return 1;
    }

    const int movieWidth = static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH));
    const int movieHeight = static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT));

    const float videoScale = 0.25f; // Rescale ratio
    const unsigned int rescaleWidth = videoScale * movieWidth;
    const unsigned int rescaleHeight = videoScale * movieHeight;
    int num_inputs = rescaleWidth * rescaleHeight * 3;

    LoopSOM1D som;
    som.init(num_inputs, 100, rng);

    // Video rescaling render target
    sf::RenderTexture rescaleRT;
    rescaleRT.create(rescaleWidth, rescaleHeight);

    int captureLength = static_cast<int>(capture.get(CAP_PROP_FRAME_COUNT));

    std::vector<std::vector<float>> images(captureLength);

    for (int i = 0; i < captureLength; i++) {
        capture >> frame;

        resize(frame, frame, Size(rescaleWidth, rescaleHeight));

        images[i].resize(num_inputs);

        for (int j = 0; j < num_inputs; j++)
            images[i][j] = frame.data[j] / 255.0f;
    }
        
    std::cout << "Capture has " << captureLength << " frames" << std::endl;

    std::uniform_int_distribution<int> image_dist(0, images.size() - 1);

    for (int it = 0; it < 100000; it++) {
        int rand_index = image_dist(rng);

        som.step(images[rand_index], true);

        if (it % 100 == 0)
            std::cout << it << std::endl;
    }

    std::vector<float> errors(captureLength, 0.0f);

    bool quit = false;

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    int fi = 0;
    sf::Texture tex;

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

        int bmu = som.step(images[fi], false);

        std::cout << bmu << std::endl;

        window.clear();

        sf::Image img;
        img.create(rescaleWidth, rescaleHeight);

        for (int x = 0; x < rescaleWidth; x++)
            for (int y = 0; y < rescaleHeight; y++) {
                sf::Uint8 r = images[fi][0 + 3 * (x + y * rescaleWidth)] * 255.0f;
                sf::Uint8 g = images[fi][1 + 3 * (x + y * rescaleWidth)] * 255.0f;
                sf::Uint8 b = images[fi][2 + 3 * (x + y * rescaleWidth)] * 255.0f;

                img.setPixel(x, y, sf::Color(r, g, b));
            }

        tex.loadFromImage(img);

        sf::Sprite s;

        s.setTexture(tex);
        s.setScale(4.0f, 4.0f);

        window.draw(s);

        fi++;

        if (fi >= images.size())
            fi = 0;

        window.display();
    } while (!quit);

    quit = true;

    return 0;
}

