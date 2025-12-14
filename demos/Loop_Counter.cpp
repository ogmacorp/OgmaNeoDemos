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

class Loop_Tracker {
private:
    aon::Hierarchy h;
    aon::Image_Encoder enc;

    struct Node {
        float t;

        aon::Array<aon::Int_Buffer> full_cis;
    };

    std::vector<Node> nodes;

    int last_index = -1;

public:
    int num_hidden_cis;
    float threshold = 0.5f;
    float smoothing = 0.1f;

    void init_random(int width, int height, int channels) {
        aon::Int3 image_size(height, width, channels);

        aon::Int3 hidden_size(10, 10, 16);

        aon::Array<aon::Image_Encoder::Visible_Layer_Desc> vlds(1);
        vlds[0].size = image_size;
        vlds[0].radius = 5;

        enc.init_random(hidden_size, vlds);

        aon::Array<aon::Hierarchy::Layer_Desc> lds(2);

        num_hidden_cis = 0;

        for (int i = 0; i < lds.size(); i++) {
            lds[i].hidden_size = aon::Int3(5, 5, 32);
            //lds[i].temporal_size = 1;

            num_hidden_cis += lds[i].hidden_size.x * lds[i].hidden_size.y;
        }

        aon::Array<aon::Hierarchy::IO_Desc> iods(1);
        iods[0].size = hidden_size;
        iods[0].up_radius = 4;

        h.init_random(iods, lds);
    }

    float step(const aon::Byte_Buffer &image, bool learn_enabled) {
        aon::Array<aon::Byte_Buffer_View> imgs(1);
        imgs[0] = image;

        enc.step(imgs, learn_enabled);

        aon::Array<aon::Int_Buffer_View> encs(1);
        encs[0] = enc.get_hidden_cis();

        h.step(encs, learn_enabled);

        int detect_thresh = static_cast<int>(threshold * num_hidden_cis);

        // check nodes
        int max_index = -1;
        int max_sim = 0;

        for (int i = 0; i < nodes.size(); i++) {
            int sim = similarity(nodes[i].full_cis);

            if (sim > max_sim) {
                max_sim = sim;
                max_index = i;
            }
        }

        if (max_sim < detect_thresh) {
            // add new node
            nodes.push_back(Node());

            Node &node = nodes.back();

            node.full_cis = get_full_cis();

            if (max_index == -1) {
                max_index = 0;
                node.t = 0.0f;
            }
            else
                node.t = nodes[max_index].t + 1.0f;

            max_index = nodes.size() - 1;
        }
        else {
            nodes[max_index].full_cis = get_full_cis();
            nodes[max_index].t += smoothing * (nodes[last_index].t + 1.0f - nodes[max_index].t);
        }

        last_index = max_index;

        return nodes[max_index].t;
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

    const int width = 64;
    const int height = 64;

    const int num_inputs = width * height * 1;

    const std::string fileNameBase = "resources/data/video";
    const std::string fileNameExt = ".avi";

    const int num_videos = 1;
    const int iters_per_video = 1;

    // Initialize a random number generator
    std::mt19937 rng(time(nullptr));

    const unsigned int windowWidth = 1200;
    const unsigned int windowHeight = 800;

    Loop_Tracker tracker;
    tracker.init_random(width, height, 1);

    // pretrain
    for (int f = 0; f < num_videos; f++) {
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

        // train tracker
        aon::Byte_Buffer bimage(num_inputs);

        for (int it = 0; it < iters_per_video; it++) {
            for (int t = 0; t < images.size(); t++) {
                for (int i = 0; i < bimage.size(); i++)
                    bimage[i] = images[t][i];

                float rt = tracker.step(bimage, true);

                std::cout << "RT: " << rt << std::endl;
                
                if (t % 1000 == 0) {
                    std::cout << "T: " << t << std::endl;
                }
            }

            std::cout << "It " << it << " complete." << std::endl;
        }
    }

    return 0;
}


