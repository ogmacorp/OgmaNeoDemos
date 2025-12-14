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

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/image_encoder.h>

#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <cmath>

using namespace cv;

const bool load_existing_loop_points = true;

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

const float pi = 3.141592f;

float min_angle_delta(float delta) {
    return std::fmod(delta + pi, 2.0f * pi) - pi;
}

float length(const sf::Vector2f &v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

bool limit_90 = false;
float limit_90_rate = 0.05f;

int main() {
    aon::set_num_threads(8);

    const std::string fileName = "resources/singlelap.avi";

    std::ifstream ff("resources/racevit_controls.txt");

    std::vector<std::tuple<float, float>> data;

    while (ff.good() && !ff.eof()) {
        float throttle, steer;

        ff >> throttle >> steer;

        data.push_back(std::make_tuple(throttle, (steer - 0.42f) * 4.0f));
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

    const float videoScale = 0.5f; // Rescale ratio
    const unsigned int rescaleWidth = videoScale * movieWidth;
    const unsigned int rescaleHeight = videoScale * movieHeight;
    int num_inputs = rescaleWidth * rescaleHeight * 3;

    // Video rescaling render target
    sf::RenderTexture rescaleRT;
    rescaleRT.create(rescaleWidth, rescaleHeight);

    int captureLength = static_cast<int>(capture.get(CAP_PROP_FRAME_COUNT));

    std::vector<std::vector<unsigned char>> images(captureLength);

    for (int i = 0; i < captureLength; i++) {
        capture >> frame;

        resize(frame, frame, Size(rescaleWidth, rescaleHeight));

        images[i].resize(num_inputs);

        for (int j = 0; j < num_inputs; j++)
            images[i][j] = frame.data[j];
    }
        
    std::cout << "Capture has " << captureLength << " frames. Commands has " << data.size() << std::endl;

    std::vector<sf::Vector3f> chain(1000);
    int radius = 5;

    const float turn_speed = 0.1f;

    sf::Vector3f transform(0.0f, 0.0f, 0.0f);

    for (int c = 0; c < chain.size(); c++) {
        float ratio = static_cast<float>(c) / chain.size();

        int t = static_cast<int>(ratio * images.size());

        // gather
        float average_velocity = 0.0f;
        float average_turn = 0.0f;
        
        for (int dt = -radius; dt <= radius; dt++) {
            average_velocity += std::get<0>(data[t]);
            average_turn += std::get<1>(data[t]);
        }

        average_velocity /= (radius * 2 + 1);
        average_turn /= (radius * 2 + 1);

        transform.z += average_turn * turn_speed;
        transform.z = min_angle_delta(transform.z);

        transform.x += std::cos(transform.z) * average_velocity;
        transform.y += std::sin(transform.z) * average_velocity;

        chain[c] = transform;
    }

    // mend prep - store distances for preservation
    std::vector<float> lengths(chain.size());
    std::vector<float> init_angles(chain.size());

    for (int c = 0; c < chain.size() - 1; c++) {
        lengths[c] = length(sf::Vector2f(chain[c].x - chain[c + 1].x, chain[c].y - chain[c + 1].y));

        init_angles[c] = chain[c].z;
    }

    lengths[lengths.size() - 1] = 0.0f;

    // mend loop
    float mr = 0.05f; // mend rate
    float sr = 0.05f; // static rate (keep the original angles alive somewhat)

    // mend
    for (int it = 0; it < 1000; it++) {
        for (int c = 0; c < chain.size(); c++) {
            int c_prev = (c - 1 + chain.size()) % chain.size();
            int c_next = (c + 1) % chain.size();

            chain[c].z = min_angle_delta(chain[c].z + mr * (min_angle_delta(chain[c_next].z - chain[c].z) + min_angle_delta(chain[c].z - chain[c_prev].z)) + sr * min_angle_delta(init_angles[c] - chain[c].z));

            if (limit_90) {
                int d_use = 0;
                float min_angle = 999999.0f;

                for (int d = 0; d < 4; d++) {
                    float a = d * pi * 0.5f;

                    float diff = abs(min_angle_delta(chain[c].z - a));

                    if (diff < min_angle) {
                        min_angle = diff;
                        d_use = d;
                    }
                }

                chain[c].z += limit_90_rate * min_angle_delta(d_use * pi * 0.5f - chain[c].z);
                chain[c].z = min_angle_delta(chain[c].z);
            }

            sf::Vector2f pred_from_prev = sf::Vector2f(chain[c_prev].x, chain[c_prev].y) + lengths[c_prev] * sf::Vector2f(std::cos(chain[c_prev].z), std::sin(chain[c_prev].z));
            sf::Vector2f pred_from_next = sf::Vector2f(chain[c_next].x, chain[c_next].y) - lengths[c] * sf::Vector2f(std::cos(chain[c_next].z), std::sin(chain[c_next].z));

            sf::Vector2f target = (pred_from_prev + pred_from_next) * 0.5f;

            chain[c].x = target.x;
            chain[c].y = target.y;
        }
    }

    std::cout << "Ready." << std::endl;

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

        window.clear();

        sf::Image img;
        img.create(rescaleWidth, rescaleHeight);

        for (int x = 0; x < rescaleWidth; x++)
            for (int y = 0; y < rescaleHeight; y++) {
                sf::Uint8 r = images[fi][0 + 3 * (x + y * rescaleWidth)];
                sf::Uint8 g = images[fi][1 + 3 * (x + y * rescaleWidth)];
                sf::Uint8 b = images[fi][2 + 3 * (x + y * rescaleWidth)];

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

        float show_scale = 1.0f;

        for (int c = 0; c < chain.size(); c++) {
            cs.setPosition(sf::Vector2f(chain[c].x, chain[c].y) * show_scale + sf::Vector2f(window.getSize().x * 0.5f, window.getSize().y * 0.5f));

            window.draw(cs);
        }

        fi++;

        if (fi >= images.size())
            fi = 0;

        window.display();
    } while (!quit);

    quit = true;

    return 0;
}


