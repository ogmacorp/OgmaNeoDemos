// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <opencv2/core/core.hpp>
#include <opencv2/highgui.hpp>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <random>

#include <neo/Predictor.h>

using namespace ogmaneo;
using namespace cv;

int main() {
    // Initialize a random number generator
    std::mt19937 generator(time(nullptr));

    // Uniform distribution in [0, 1]
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(800, 800), "Video Test", sf::Style::Default);

    // Uncap framerate
    window.setFramerateLimit(0);

    // Target file name
    std::string fileName = "resources/Tesseract.wmv";

    sf::Font font;

#ifdef _WINDOWS
    font.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#else
    font.loadFromFile("/usr/share/fonts/truetype/freefont/FreeMono.ttf");
#endif

    // Parameters
    const int frameSkip = 2; // Frames to skip
    const float videoScale = 1.0f; // Rescale ratio
    const float blendPred = 0.0f; // Ratio of how much prediction to blend in to input (part of input corruption)
    const float noise = 0.01f; // Noise (part of input corruption)

    // Video rescaling render target
    sf::RenderTexture rescaleRT;
    rescaleRT.create(128, 128);

    ComputeSystem cs;

    // Use GPU
    cs.create(ComputeSystem::_gpu);

    ComputeProgram prog;

    // Load the main portion of the kernels
    prog.loadMainKernel(cs);

    // --------------------------- Create the Sparse Coder ---------------------------

    // Input images
    cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), static_cast<cl_int>(rescaleRT.getSize().x), static_cast<cl_int>(rescaleRT.getSize().y));
    cl::Image2D inputImageCorrupted = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), static_cast<cl_int>(rescaleRT.getSize().x), static_cast<cl_int>(rescaleRT.getSize().y));

    // Hierarchy structure
    std::vector<FeatureHierarchy::LayerDesc> layerDescs(3);
    std::vector<Predictor::PredLayerDesc> pLayerDescs(3);

    layerDescs[0]._size = { 64, 64 };
    layerDescs[1]._size = { 64, 64 };
    layerDescs[2]._size = { 64, 64 };

    // Predictive hierarchy
    Predictor ph;

    ph.createRandom(cs, prog, { static_cast<cl_int>(rescaleRT.getSize().x), static_cast<cl_int>(rescaleRT.getSize().y) }, pLayerDescs, layerDescs, { -0.01f, 0.01f }, generator);

    // Host image buffer
    std::vector<float> pred(rescaleRT.getSize().x * rescaleRT.getSize().y, 0.0f);

    // Unit Gaussian noise for input corruption
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);

    // Training time
    const int numIter = 8;

    // UI update resolution
    const int progressBarLength = 40;

    const int progressUpdateTicks = 4;

    bool quit = false;

    // Train for a bit
    for (int iter = 0; iter < numIter && !quit; iter++) {
        std::cout << "Iteration " << (iter + 1) << " of " << numIter << ":" << std::endl;

        // Open the video file
        VideoCapture capture(fileName);
        Mat frame;

        if (!capture.isOpened())
            std::cerr << "Could not open capture: " << fileName << std::endl;

        std::cout << "Running through capture: " << fileName << std::endl;

        const int captureLength = capture.get(CAP_PROP_FRAME_COUNT);

        int currentFrame = 0;

        // Run through video
        do {
            // Read several discarded frames if frame skip is > 0
            for (int i = 0; i < frameSkip; i++) {
                capture >> frame;

                currentFrame++;

                if (frame.empty())
                    break;
            }

            if (frame.empty())
                break;

            // Convert to SFML image
            sf::Image img;

            img.create(frame.cols, frame.rows);

            for (int x = 0; x < img.getSize().x; x++)
                for (int y = 0; y < img.getSize().y; y++) {
                    sf::Uint8 r = frame.data[(x + y * img.getSize().x) * 3 + 0];
                    sf::Uint8 g = frame.data[(x + y * img.getSize().x) * 3 + 1];
                    sf::Uint8 b = frame.data[(x + y * img.getSize().x) * 3 + 2];

                    img.setPixel(x, y, sf::Color(r, g, b));
                }

            // To SFML texture
            sf::Texture tex;
            tex.loadFromImage(img);

            tex.setSmooth(true);

            // Rescale using render target
            sf::Sprite s;

            s.setPosition(rescaleRT.getSize().x * 0.5f, rescaleRT.getSize().y * 0.5f);

            s.setTexture(tex);

            s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));

            float scale = videoScale * std::min(static_cast<float>(rescaleRT.getSize().x) / img.getSize().x, static_cast<float>(rescaleRT.getSize().y) / img.getSize().y);

            s.setScale(scale, scale);

            rescaleRT.clear();

            rescaleRT.draw(s);

            rescaleRT.display();

            // SFML image from rescaled frame
            sf::Image reImg = rescaleRT.getTexture().copyToImage();

            // Get input buffers
            std::vector<float> monochrome(reImg.getSize().x * reImg.getSize().y);
            std::vector<float> monochromeCorrupted(reImg.getSize().x * reImg.getSize().y);

            for (int x = 0; x < reImg.getSize().x; x++)
                for (int y = 0; y < reImg.getSize().y; y++) {
                    sf::Color c = reImg.getPixel(x, y);

                    float v = (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f) * 0.3333f;

                    float mono = invert ? 1.0f - v : v;

                    monochrome[x + y * reImg.getSize().x] = mono;// > thresh ? 1.0f : 0.0f;

                    monochromeCorrupted[x + y * reImg.getSize().x] = blendPred * pred[x + y * reImg.getSize().x] + (1.0f - blendPred) * monochrome[x + y * reImg.getSize().x] + noise * noiseDist(generator);
                }

            cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { reImg.getSize().x, reImg.getSize().y, 1 }, 0, 0, monochrome.data());
            cs.getQueue().enqueueWriteImage(inputImageCorrupted, CL_TRUE, { 0, 0, 0 }, { reImg.getSize().x, reImg.getSize().y, 1 }, 0, 0, monochromeCorrupted.data());

            // Run a simulation step of the hierarchy (learning enabled)
            ph.simStep(cs, inputImage, inputImageCorrupted, generator, true);

            // Get the resulting prediction (for prediction blending)
            cs.getQueue().enqueueReadImage(ph.getPrediction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(rescaleRT.getSize().x), static_cast<cl::size_type>(rescaleRT.getSize().y), 1 }, 0, 0, pred.data());

            // Show progress bar
            float ratio = static_cast<float>(currentFrame + 1) / captureLength;

            // Console
            if (currentFrame % progressUpdateTicks == 0) {
                std::cout << "\r";
                std::cout << "[";

                int bars = static_cast<int>(std::round(ratio * progressBarLength));

                int spaces = progressBarLength - bars;

                for (int i = 0; i < bars; i++)
                    std::cout << "=";

                for (int i = 0; i < spaces; i++)
                    std::cout << " ";

                std::cout << "] " << static_cast<int>(ratio * 100.0f) << "%";
            }

            // UI
            if (currentFrame % progressUpdateTicks == 0) {
                sf::Event windowEvent;

                while (window.pollEvent(windowEvent))
                {
                    switch (windowEvent.type)
                    {
                    case sf::Event::Closed:
                        quit = true;
                        break;
                    }
                }

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                    quit = true;

                // Show progress bar
                window.clear();

                sf::RectangleShape rs;

                rs.setPosition(8.0f, 8.0f);

                rs.setSize(sf::Vector2f(128.0f * ratio, 32.0f));

                rs.setFillColor(sf::Color::Red);

                window.draw(rs);

                rs.setFillColor(sf::Color::Transparent);

                rs.setOutlineColor(sf::Color::White);
                rs.setOutlineThickness(2.0f);

                rs.setSize(sf::Vector2f(128.0f, 32.0f));

                window.draw(rs);

                sf::Text t;

                t.setFont(font);

                t.setCharacterSize(20);

                std::string st;

                st += std::to_string(static_cast<int>(ratio * 100.0f)) + "% (pass " + std::to_string(iter + 1) + " of " + std::to_string(numIter) + ")";

                t.setString(st);

                t.setPosition(144.0f, 8.0f);

                t.setColor(sf::Color::White);

                window.draw(t);

                window.display();
            }

        } while (!frame.empty() && !quit);

        // Make sure bar is at 100%
        std::cout << "\r";
        std::cout << "[";

        for (int i = 0; i < progressBarLength; i++)
            std::cout << "=";

        std::cout << "] 100%";

        std::cout << std::endl;
    }

    // ---------------------------- Presentation Simulation Loop -----------------------------

    window.setVerticalSyncEnabled(true);

    do {
        // ----------------------------- Input -----------------------------

        sf::Event windowEvent;

        while (window.pollEvent(windowEvent))
        {
            switch (windowEvent.type)
            {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            quit = true;

        window.clear();

        // Write prediction as input
        cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { rescaleRT.getSize().x, rescaleRT.getSize().y, 1 }, 0, 0, pred.data());

        // Run a simulation step with learning disabled
        ph.simStep(cs, inputImage, inputImage, generator, false);

        // Display prediction
        cs.getQueue().enqueueReadImage(ph.getPrediction(), CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(rescaleRT.getSize().x), static_cast<cl::size_type>(rescaleRT.getSize().y), 1 }, 0, 0, pred.data());

        sf::Image img;

        img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);

        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c;

                c.r = c.g = c.b = 255.0f * std::min(1.0f, std::max(0.0f, pred[x + y * img.getSize().x]));

                img.setPixel(x, y, c);
            }

        sf::Texture tex;

        tex.loadFromImage(img);

        sf::Sprite s;

        s.setPosition(window.getSize().x * 0.5f, window.getSize().y * 0.5f);

        s.setTexture(tex);

        s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));

        float scale = videoScale * std::min(static_cast<float>(window.getSize().x) / img.getSize().x, static_cast<float>(window.getSize().y) / img.getSize().y);

        s.setScale(sf::Vector2f(scale, scale));

        window.draw(s);

        window.display();
    } while (!quit);

    return 0;
}
