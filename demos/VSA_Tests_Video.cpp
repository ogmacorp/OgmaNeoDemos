#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/image_encoder.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <valarray>
#include <random>
#include <omp.h>

#include "vec.h"

#include <assert.h>

const int S = 256;
const int L = 16;
const int N = S * L;

typedef v::Vec<S, L> Vec1;
typedef v::Bundle<S, L> Bundle1;

Vec1 embedding1d(float x, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 2] * x + loc[i * 2 + 1];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

float unembedding1d(const Vec1 &v, const std::valarray<float> &loc) {
    // mat mul
    float x = 0.0f;

    for (int i = 0; i < S; i++) {
        float val = v[i] / static_cast<float>(L);

        val = (val - loc[i * 2 + 1]) / v::max(v::limit_small, abs(loc[i * 2])) * (loc[i * 2] > 0.0f ? 1.0f : -1.0f);

        x += val;
    }

    return x / S;
}

Vec1 embedding2d(float x, float y, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 3] * x + loc[i * 3 + 1] * y + loc[i * 3 + 2];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

Vec1 embedding3d(float x, float y, float z, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 4] * x + loc[i * 4 + 1] * y + loc[i * 4 + 2] * z + loc[i * 4 + 3];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

void print(const Vec1 &v) {
    std::cout << "[ ";

    for (int i = 0; i < v.segments(); i++)
        std::cout << static_cast<int>(v[i]) << " ";

    std::cout << " ]" << std::endl;
}

void print(const Bundle1 &b) {
    std::cout << "[ ";

    for (int i = 1; i < b.size(); i++)
        std::cout << b[i] << " ";

    std::cout << " ]" << std::endl;
}

class VSA_Image_Encoder {
public:
    std::valarray<float> Z; // location
    aon::Image_Encoder enc;
    int stride = 12;

    void init(
        const aon::Int3 &hidden_size,
        const aon::Int3 &patch_size,
        std::mt19937 &rng
    ) {
        
        aon::Array<aon::Image_Encoder::Visible_Layer_Desc> vlds(1);

        vlds[0].size = patch_size;
        vlds[0].radius = 4;

        enc.init_random(hidden_size, vlds);

        std::normal_distribution<float> ndist(0.0f, 1.0f);

        Z.resize(S * 2);

        for (int i = 0; i < Z.size(); i++)
            Z[i] = ndist(rng) * 1.0f;
    }

    Vec1 step(
        aon::U8_Array &image,
        const aon::Int3 &image_size,
        bool learn_enabled
    ) {
        aon::Int3 patch_size = enc.get_visible_layer_desc(0).size;

        aon::U8_Array patch(patch_size.x * patch_size.y * patch_size.z);

        // sample some random patches
        aon::Array<aon::U8_Array_View> images(1);
        images[0] = patch;

        Bundle1 b = 0.0f;

        for (int x = 0; x < image_size.x - patch_size.x; x += stride)
            for (int y = 0; y < image_size.y - patch_size.y; y += stride) {
                for (int px = 0; px < patch_size.x; px++)
                    for (int py = 0; py < patch_size.y; py++) {
                        int fx = x + px;
                        int fy = y + py;

                        patch[0 + 3 * (py + px * patch_size.y)] = image[0 + 3 * (fy + fx * image_size.y)];
                        patch[1 + 3 * (py + px * patch_size.y)] = image[1 + 3 * (fy + fx * image_size.y)];
                        patch[2 + 3 * (py + px * patch_size.y)] = image[2 + 3 * (fy + fx * image_size.y)];
                    }

                enc.step(images, learn_enabled, learn_enabled);

                // get code as hypervector
                Vec1 v;

                for (int i = 0; i < S; i++)
                    v[i] = enc.get_hidden_cis()[i];

                float xf = x / static_cast<float>(image_size.x - 1);
                float yf = y / static_cast<float>(image_size.y - 1);

                Vec1 loc = embedding1d(xf, Z) * embedding1d(yf, Z).permute(1);

                b += v * loc;
            }

        return b.thin();
    }

    void reconstruct(
        const Vec1 &v,
        const aon::Int3 &image_size,
        aon::U8_Array &image
    ) {
        image.fill(0);

        aon::S32_Array imagei(image_size.x * image_size.y * image_size.z, 0);
        aon::S32_Array counts(image_size.x * image_size.y, 0);

        aon::Int3 patch_size = enc.get_visible_layer_desc(0).size;

        aon::S32_Array csdr(S);

        for (int x = 0; x < image_size.x - patch_size.x; x += stride)
            for (int y = 0; y < image_size.y - patch_size.y; y += stride) {
                float xf = x / static_cast<float>(image_size.x - 1);
                float yf = y / static_cast<float>(image_size.y - 1);

                Vec1 loc = embedding1d(xf, Z) * embedding1d(yf, Z).permute(1);

                Vec1 local = v / loc;

                for (int i = 0; i < S; i++)
                    csdr[i] = local[i];

                enc.reconstruct(csdr);

                const aon::U8_Array &patch = enc.get_reconstruction(0);

                for (int px = 0; px < patch_size.x; px++)
                    for (int py = 0; py < patch_size.y; py++) {
                        int fx = x + px;
                        int fy = y + py;

                        imagei[0 + 3 * (fy + fx * image_size.y)] += patch[0 + 3 * (py + px * patch_size.y)];
                        imagei[1 + 3 * (fy + fx * image_size.y)] += patch[1 + 3 * (py + px * patch_size.y)];
                        imagei[2 + 3 * (fy + fx * image_size.y)] += patch[2 + 3 * (py + px * patch_size.y)];
                        counts[fy + fx * image_size.y]++;
                    }
            }

        for (int i = 0; i < counts.size(); i++) {
            image[0 + 3 * i] = imagei[0 + 3 * i] / std::max(1, counts[i]);
            image[1 + 3 * i] = imagei[1 + 3 * i] / std::max(1, counts[i]);
            image[2 + 3 * i] = imagei[2 + 3 * i] / std::max(1, counts[i]);
        }
    }
};

int main() {
    unsigned int windowWidth = 1600;
    unsigned int windowHeight = 1200;

    sf::RenderWindow window;

    window.create(sf::VideoMode(sf::Vector2u(windowWidth, windowHeight)), "VSA Test", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    std::mt19937 rng(time(nullptr));

    std::string fileName = "resources/Bullfinch192_small.mp4";

    // Open the video file
    cv::VideoCapture capture(fileName);
    cv::Mat frame;

    if (!capture.isOpened()) {
        std::cerr << "Could not open capture: " << fileName << std::endl;
        return 1;
    }

    const int movieWidth = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_WIDTH));
    const int movieHeight = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_HEIGHT));

    // --------------------------- Create the Hierarchy ---------------------------

    aon::set_num_threads(8);

    // Create hierarchy
    aon::Array<aon::Hierarchy::Layer_Desc> lds(2);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = aon::Int3(8, 8, 32);
    }

    aon::Int3 hiddenSize(16, 16, 16);
    aon::Int3 patchSize(16, 16, 3);

    aon::Array<aon::Hierarchy::IO_Desc> ioDescs(1);
    ioDescs[0] = aon::Hierarchy::IO_Desc(hiddenSize, aon::IO_Type::prediction, 4, 2);

    // Forward declare
    aon::Hierarchy h;
    h.init_random(ioDescs, lds);

    std::cout << "Running through capture: " << fileName << std::endl;

    int captureLength = static_cast<int>(capture.get(cv::CAP_PROP_FRAME_COUNT));

    // Calculate actual number of frames
    //int i = 1;
    //for (; i <= captureLength; i++) {
    //    capture >> frame;

    //    if (frame.empty())
    //        break;
    //}
    //    
    //captureLength = i;

    std::cout << "Capture has " << captureLength << " frames" << std::endl;

    VSA_Image_Encoder enc;
    enc.init(hiddenSize, patchSize, rng);

    int numIter = 40;

    for (int iter = 0; iter < numIter; iter++) {
        std::cout << "Iteration " << (iter + 1) << " of " << numIter << ":" << std::endl;

        int currentFrame = 0;

        capture.set(cv::CAP_PROP_POS_FRAMES, 0.0f);

        // Run through video
        do {
            // Read several discarded frames if frame skip is > 0
            for (int i = 0; i < 2; i++) {
                capture >> frame;

                currentFrame++;

                if (frame.empty())
                    break;
            }

            if (frame.empty())
                break;

            if (currentFrame > captureLength)
                break;

            aon::U8_Array image(frame.rows * frame.cols * 3);

            for (int x = 0; x < frame.cols; x++)
                for (int y = 0; y < frame.rows; y++) {
                    image[0 + 3 * (y + x * frame.rows)] = frame.data[2 + x * 3 + y * 3 * frame.cols]; // Reverse order so it's BGR -> RGB
                    image[1 + 3 * (y + x * frame.rows)] = frame.data[1 + x * 3 + y * 3 * frame.cols]; // OpencV is a different matrix order than OgmaNeo
                    image[2 + 3 * (y + x * frame.rows)] = frame.data[0 + x * 3 + y * 3 * frame.cols];
                }

            Vec1 v = enc.step(image, aon::Int3(frame.rows, frame.cols, 3), true);

            aon::S32_Array csdr(S);

            for (int i = 0; i < S; i++)
                csdr[i] = v[i];

            aon::Array<aon::S32_Array_View> inputs(1);
            inputs[0] = csdr;

            h.step(inputs, true);
        } while (!frame.empty());
    }

    bool quit = false;

    aon::U8_Array recon_image(64 * 64 * 3);

    do {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                quit = true;
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape))
                quit = true;
        }

        aon::Array<aon::S32_Array_View> inputs(1);
        inputs[0] = h.get_prediction_cis(0);

        h.step(inputs, false);

        Vec1 v;

        for (int i = 0; i < S; i++)
            v[i] = h.get_prediction_cis(0)[i];

        enc.reconstruct(v, aon::Int3(64, 64, 3), recon_image);

        sf::Image img(sf::Vector2u(64, 64));

        for (int x = 0; x < img.getSize().x; x++)
            for (int y = 0; y < img.getSize().y; y++) {
                std::uint8_t r = recon_image[0 + 3 * (y + x * img.getSize().y)];
                std::uint8_t g = recon_image[1 + 3 * (y + x * img.getSize().y)];
                std::uint8_t b = recon_image[2 + 3 * (y + x * img.getSize().y)];

                sf::Color c(r, g, b);

                img.setPixel(sf::Vector2u(x, y), c);
            }

        sf::Texture tex(img);

        sf::Sprite s(tex);
        s.setScale(sf::Vector2f(4.0f, 4.0f));

        window.draw(s);
        
        window.display();
    } while (!quit);

    return 0;
}

