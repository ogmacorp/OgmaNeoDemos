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

const bool load_existing_loop_points = false;

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

class Loop_Detector {
private:
    aon::Hierarchy h;
    aon::Image_Encoder enc;

public:
    int num_hidden_cis;

    void init_random(int width, int height, int channels) {
        aon::Int3 image_size(height, width, channels);

        aon::Int3 hidden_size(10, 10, 16);

        aon::Array<aon::Image_Encoder::Visible_Layer_Desc> vlds(1);
        vlds[0].size = image_size;
        vlds[0].radius = 8;

        enc.init_random(hidden_size, vlds);

        aon::Array<aon::Hierarchy::Layer_Desc> lds(4);

        num_hidden_cis = 0;

        for (int i = 0; i < lds.size(); i++) {
            lds[i].hidden_size = aon::Int3(5, 5, 32);

            num_hidden_cis += lds[i].hidden_size.x * lds[i].hidden_size.y;
        }

        aon::Array<aon::Hierarchy::IO_Desc> iods(1);
        iods[0].size = hidden_size;
        iods[0].up_radius = 4;

        h.init_random(iods, lds);
    }

    void step(const aon::Byte_Buffer &image, bool learn_enabled) {
        aon::Array<aon::Byte_Buffer_View> imgs(1);
        imgs[0] = image;

        enc.step(imgs, learn_enabled);

        aon::Array<aon::Int_Buffer_View> encs(1);
        encs[0] = enc.get_hidden_cis();

        h.step(encs, learn_enabled);
    }

    aon::Array<aon::Int_Buffer> get_full_cis() const {
        aon::Array<aon::Int_Buffer>  cis(h.get_num_layers());

        for (int i = 0; i < h.get_num_layers(); i++)
            cis[i] = h.get_encoder(i).get_hidden_cis();

        return cis;
    }

    int similarity(aon::Array<aon::Int_Buffer> &compare_cis) {
        int sim = 0;

        for (int i = 0; i < h.get_num_layers(); i++) {
            for (int j = 0; j < compare_cis[i].size(); j++)
                sim += (compare_cis[i][j] == h.get_encoder(i).get_hidden_cis()[j]);
        }

        return sim;
    }
};

int main() {
    aon::set_num_threads(8);

    const std::string fileName = "resources/video0.avi";

    std::ifstream ff("resources/control0.txt");

    std::vector<std::tuple<float, float>> data;

    while (ff.good() && !ff.eof()) {
        float throttle, steer;

        ff >> throttle >> steer;

        data.push_back(std::make_tuple(throttle, steer));
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

    std::vector<int> loop_points;
    std::vector<std::vector<int>> csdrs;

    if (load_existing_loop_points) {
        {
            std::ifstream from_file("loop_points.txt");

            while (from_file.good() && !from_file.eof()) {
                int t;

                from_file >> t;

                loop_points.push_back(t);
            }

            std::cout << "Loaded " << loop_points.size() << " loop points." << std::endl;
        }

        {
            std::ifstream from_file("csdrs.txt");

            while (from_file.good() && !from_file.eof()) {
                std::string line;

                std::getline(from_file, line);

                std::sstream from_s(line);

                csdrs.push_back(std::vector<int>());

                while (from_s.good() && !from_s.eof()) {
                    int t;

                    from_s >> t;

                    csdrs.back().push_back(t);
                }
            }

            std::cout << "Loaded " << loop_points.size() << " loop points." << std::endl;
        }
    }
    else {
        Loop_Detector detector;
        detector.init_random(rescaleWidth, rescaleHeight, 3);

        // train detector
        aon::Byte_Buffer bimage(num_inputs);

        for (int it = 0; it < 5; it++) {
            for (int t = 0; t < images.size(); t++) {
                for (int i = 0; i < bimage.size(); i++)
                    bimage[i] = images[t][i];

                detector.step(bimage, true);
                
                if (t % 1000 == 0) {
                    std::cout << "T: " << t << std::endl;
                }
            }

            std::cout << "It " << it << " complete." << std::endl;
        }

        auto cis = detector.get_full_cis();

        int loop_thresh = static_cast<int>(0.5f * detector.num_hidden_cis);

        std::cout << "Thresh: " << loop_thresh << std::endl;

        int last_detect = 0;

        int min_detect_dt = 500;

        for (int t = 0; t < images.size(); t++) {
            for (int i = 0; i < bimage.size(); i++)
                bimage[i] = images[t][i];

            detector.step(bimage, false);
            
            int sim = detector.similarity(cis);

            bool detect = sim >= loop_thresh;

            if (detect) {
                if (t - last_detect > min_detect_dt) {
                    std::cout << "Loop at T " << t << " with sim " << sim << std::endl;

                    loop_points.push_back(t);

                    last_detect = t;
                }
            }
        }

        std::cout << "Loop count: " << loop_points.size() << std::endl;

        // save loop points
        {
            std::ofstream to_file("loop_points.txt");

            for (int p = 0; p < loop_points.size(); p++) 
                to_file << loop_points[p] << " ";
        }
    }

    std::vector<sf::Vector3f> chain(1000);
    int radius = 5;

    const float turn_speed = 0.045f;
    const float lr = 0.01f;

    sf::Vector3f transform(0.0f, 0.0f, 0.0f);

    for (int p = 0; p < loop_points.size() - 1; p++) {
        int t0 = loop_points[p];
        int t1 = loop_points[p + 1];

        // set
        if (p == 0) {
            for (int c = 0; c < chain.size(); c++) {
                float ratio = static_cast<float>(c) / chain.size();

                int t = static_cast<int>(ratio * (t1 - t0)) + t0;

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
        }
        else { // fit
            sf::Vector3f transform(0.0f, 0.0f, 0.0f);

            for (int c = 0; c < chain.size(); c++) {
                float ratio = static_cast<float>(c) / chain.size();

                int t = static_cast<int>(ratio * (t1 - t0)) + t0;

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

                chain[c] += lr * (transform - chain[c]);
            }
        }
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
    float mr = 0.1f; // mend rate
    float sr = 0.01f; // static rate (keep the original angles alive somewhat)

    // mend
    for (int it = 0; it < 1000; it++) {
        for (int c = 0; c < chain.size(); c++) {
            int c_prev = (c - 1 + chain.size()) % chain.size();
            int c_next = (c + 1) % chain.size();

            chain[c].z = min_angle_delta(chain[c].z + mr * (min_angle_delta(chain[c_next].z - chain[c].z) + min_angle_delta(chain[c].z - chain[c_prev].z)) + sr * min_angle_delta(init_angles[c] - chain[c].z));

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

        float show_scale = 2.0f;

        for (int c = 0; c < chain.size(); c++) {
            cs.setPosition(sf::Vector2f(chain[c].x, chain[c].y) * show_scale + sf::Vector2f(window.getSize().x * 0.5f, window.getSize().y * 0.5f));

            window.draw(cs);
        }

        // find closest image

        fi++;

        if (fi >= images.size())
            fi = 0;

        window.display();
    } while (!quit);

    quit = true;

    return 0;
}


