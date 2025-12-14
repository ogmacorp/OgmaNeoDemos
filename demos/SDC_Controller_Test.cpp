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

#include "sdc/sdc_controller.h"

#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <cmath>

using namespace cv;

const bool load = false;
const bool train = true;
const bool show = true;

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

int main() {
    aon::set_num_threads(8);

    const int width = 32;
    const int height = 32;

    const int num_inputs = width * height * 1;

    const std::string fileNameBase = "resources/data/video";
    const std::string fileNameExt = ".avi";

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

    SDC_Controller controller;

    if (load)
        controller.init_load("test.aon");
    else
        controller.init_random();

    if (train) {
        for (int f = 0; f < 1; f++) {
            // Open the video file
            std::string fileName = fileNameBase + std::to_string(f) + fileNameExt;

            VideoCapture capture(fileName);
            Mat frame;

            if (!capture.isOpened()) {
                std::cerr << "Could not open capture: " << fileName << std::endl;
                return 1;
            }
            else
                std::cout << "Loaded " << fileName << std::endl;

            int captureLength = static_cast<int>(capture.get(CAP_PROP_FRAME_COUNT));

            std::vector<std::vector<unsigned char>> images(captureLength);

            for (int i = 0; i < captureLength; i++) {
                capture >> frame;

                cvtColor(frame, frame, COLOR_BGR2GRAY);

                images[i].resize(num_inputs);

                for (int j = 0; j < num_inputs; j++)
                    images[i][j] = frame.data[j];
            }
                
            std::cout << "Capture has " << captureLength << " frames." << std::endl;

            for (int t = 0; t < images.size(); t++) {
                float target_throttle = images[t][0] / 255.0f * 2.0f - 1.0f;
                float target_steer = images[t][1] / 255.0f * 2.0f - 1.0f;

                images[t][0] = 0;
                images[t][1] = 0;

                controller.step(images[t], SDC_Mode::bc, 0.0f, { target_throttle, target_steer });
                
                if (t % 1000 == 0) {
                    std::cout << "T: " << t << std::endl;
                }
            }
        }
    }

    if (show) {
        for (int f = 0; f < 13; f++) {
            // Open the video file
            std::string fileName = fileNameBase + std::to_string(f) + fileNameExt;

            VideoCapture capture(fileName);
            Mat frame;

            if (!capture.isOpened()) {
                std::cerr << "Could not open capture: " << fileName << std::endl;
                return 1;
            }
            else
                std::cout << "Loaded " << fileName << std::endl;

            int captureLength = static_cast<int>(capture.get(CAP_PROP_FRAME_COUNT));

            std::vector<std::vector<unsigned char>> images(captureLength);

            for (int i = 0; i < captureLength; i++) {
                capture >> frame;

                cvtColor(frame, frame, COLOR_BGR2GRAY);

                images[i].resize(num_inputs);

                for (int j = 0; j < num_inputs; j++)
                    images[i][j] = frame.data[j];
            }
                
            std::cout << "Capture has " << captureLength << " frames." << std::endl;

            for (int t = 0; t < images.size(); t++) {
                float target_throttle = images[t][0] / 255.0f * 2.0f - 1.0f;
                float target_steer = images[t][1] / 255.0f * 2.0f - 1.0f;

                images[t][0] = 0;
                images[t][1] = 0;

                std::pair<float, float> pred = controller.step(images[t], SDC_Mode::inf, 0.0f, { 0.0f, 0.0f });

                std::cout << pred.first << " " << pred.second << std::endl;

                if (t % 1000 == 0) {
                    std::cout << "T: " << t << std::endl;
                }
            }
        }

    }

    return 0;
}
