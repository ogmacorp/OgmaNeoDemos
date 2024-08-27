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
#include <vector>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <cmath>

using namespace cv;

const float pi = 3.141592f;

float min_angle_delta(float delta) {
    return std::fmod(delta + pi, 2.0f * pi) - pi;
}

float length(const sf::Vector2f &v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

class LoopSOM1D {
public:
    struct Node {
        sf::Vector2f position;
        float angle;

        std::vector<float> proto;
    };

    int num_inputs;

    std::vector<Node> nodes;

    float lr;
    float mix;
    float drift;
    float falloff;
    int t;

    sf::Vector2f position;
    float angle;

    LoopSOM1D()
    :
    lr(0.001f),
    mix(3.0f),
    drift(0.01f),
    falloff(0.1f),
    t(0),
    position(0.0f, 0.0f),
    angle(0.0f)
    {}

    void init(int num_inputs, int num_nodes, std::mt19937 &rng) {
        this->num_inputs = num_inputs;

        nodes.resize(num_nodes);

        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
        std::normal_distribution<float> ndist(0.0f, 1.0f);

        for (int i = 0; i < nodes.size(); i++) {
            nodes[i].proto.resize(num_inputs);

            for (int j = 0; j < num_inputs; j++)
                nodes[i].proto[j] = dist01(rng);

            nodes[i].position.x = ndist(rng);
            nodes[i].position.y = ndist(rng);
            nodes[i].angle = dist01(rng) * 2.0f * pi;
        }
    }

    int step(float velocity, float turn, const std::vector<float> &inputs, bool learn_enabled = true) {
        angle = std::fmod(angle + turn + 2.0f * pi, 2.0f * pi);
        position += velocity * sf::Vector2f(std::cos(angle), std::sin(angle));

        // find bmu
        int bmu = 0;
        float min_dist = 999999.0f;

        for (int i = 0; i < nodes.size(); i++) {
            float dist = 0.0f;

            for (int j = 0; j < num_inputs; j++) {
                float diff = inputs[j] - nodes[i].proto[j];

                dist += diff * diff;
            }
            
            float mixed_dist = std::sqrt(dist) / num_inputs + mix * length(position - nodes[i].position);

            if (mixed_dist < min_dist) {
                min_dist = mixed_dist;
                bmu = i;
            }
        }

        t = bmu;

        // attract to current
        position += drift * (nodes[t].position - position);
        angle = std::fmod(angle + drift * min_angle_delta(nodes[t].angle - angle) + 2.0f * pi, 2.0f * pi);

        if (learn_enabled) {
            for (int i = 0; i < nodes.size(); i++) {
                int delta = std::min(((i - t) + nodes.size()) % nodes.size(), ((t - i) + nodes.size()) % nodes.size()); // looping delta

                float rate = lr * std::exp(-falloff * delta * delta);

                for (int j = 0; j < num_inputs; j++) {
                    float diff = inputs[j] - nodes[i].proto[j];

                    nodes[i].proto[j] += rate * diff;
                }

                nodes[i].position += rate * (position - nodes[i].position);
                nodes[i].angle = std::fmod(angle + rate * min_angle_delta(angle - nodes[i].angle) + 2.0f * pi, 2.0f * pi);
            }
        }

        return t;
    }
};

int main() {
    const std::string fileName = "resources/video0.avi";

    std::ifstream ff("resources/control0.txt");

    std::vector<std::tuple<float, float>> data;

    while (ff.good() && !ff.eof()) {
        float throttle, steer;

        ff >> throttle >> steer;

        data.push_back(std::make_tuple(throttle * 0.03f, steer * 0.003f));
    }

    ff.close();

    // Initialize a random number generator
    std::mt19937 rng(time(nullptr));

    const unsigned int windowWidth = 1200;
    const unsigned int windowHeight = 800;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Video Test", sf::Style::Default);

    // Uncap framerate
    window.setFramerateLimit(120);

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
        
    std::cout << "Capture has " << captureLength << " frames. Commands has " << data.size() << std::endl;

    std::uniform_int_distribution<int> image_dist(0, images.size() - 1);

    for (int it = 0; it < 100000; it++) {
        int t = it % images.size();

        som.step(std::get<0>(data[t]), std::get<1>(data[t]), images[t], true);

        if (it % 100 == 0)
            std::cout << it << std::endl;
    }

    std::vector<float> errors(captureLength, 0.0f);

    bool quit = false;

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    int fi = som.t;
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

        int bmu = som.step(std::get<0>(data[fi]), std::get<1>(data[fi]), images[fi], true);

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

        sf::CircleShape cs;

        cs.setRadius(2.0f);
        cs.setOrigin(1.0f, 1.0f);
        cs.setFillColor(sf::Color::Red);

        for (int i = 0; i < som.nodes.size(); i++) {
            cs.setPosition(som.nodes[i].position * 10.0f + sf::Vector2f(window.getSize().x * 0.5f, window.getSize().y * 0.5f));

            window.draw(cs);
        }

        cs.setFillColor(sf::Color::Green);
        cs.setPosition(som.position * 10.0f + sf::Vector2f(window.getSize().x * 0.5f, window.getSize().y * 0.5f));

        window.draw(cs);

        fi++;

        if (fi >= images.size())
            fi = 0;

        window.display();
    } while (!quit);

    quit = true;

    return 0;
}

