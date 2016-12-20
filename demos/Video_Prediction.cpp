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

#include <neo/Architect.h>
#include <neo/Hierarchy.h>

#include <vis/DebugWindow.h>

using namespace ogmaneo;
using namespace cv;

int main() {
    // Initialize a random number generator
    std::mt19937 generator(time(nullptr));

    // Uniform distribution in [0, 1]
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    unsigned int windowWidth = 512;
    unsigned int windowHeight = 512;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Video Test", sf::Style::Default);

    // Uncap framerate
    window.setFramerateLimit(0);

    bool enableDebugWindow = false;

    vis::DebugWindow debugWindow;

    if (enableDebugWindow) {
        window.setPosition(sf::Vector2i(64, 64));

        debugWindow.create(
            sf::String("Debug"),
            sf::Vector2i(window.getPosition().x + windowWidth + 8, window.getPosition().y),
            sf::Vector2u(512, 512));
    }

    sf::Font font;

#if defined(_WINDOWS)
    font.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#elif defined(__APPLE__)
    font.loadFromFile("/Library/Fonts/Courier New.ttf");
#else
    font.loadFromFile("/usr/share/fonts/truetype/freefont/FreeMono.ttf");
#endif

    // Parameters
    int netScale;
    int chunkLayers;
    std::vector<int> layerSizes;
    std::string fileName;

    bool useBullFinchMovie = false;

    if (useBullFinchMovie) {
        fileName = "resources/Bullfinch192.mp4";
        netScale = 192;
        chunkLayers = 6;
        layerSizes.push_back(96);
        layerSizes.push_back(96);
        layerSizes.push_back(96);
        layerSizes.push_back(60);
        layerSizes.push_back(60);
        layerSizes.push_back(60);
    }
    else {
        fileName = "resources/Tesseract.wmv";
        netScale = 128;
        chunkLayers = 4;
        layerSizes.push_back(96);
        layerSizes.push_back(96);
        layerSizes.push_back(60);
        layerSizes.push_back(60);
    }

    // Open the video file
    VideoCapture capture(fileName);
    Mat frame;

    if (!capture.isOpened()) {
        std::cerr << "Could not open capture: " << fileName << std::endl;
        return 1;
    }

    const int movieWidth = static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH));
    const int movieHeight = static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT));

    if (movieWidth != movieHeight) {
        std::cerr << "Movie file " << fileName << " has non-square frame" << std::endl;
        return 1;
    }

    const int frameSkip = 1;        // Frames to skip
    const float videoScale = 1.0f;  // Rescale ratio
    const float blendPred = 0.0f;   // Ratio of how much prediction to blend in to input (part of input corruption)

    // Video rescaling render target
    sf::RenderTexture rescaleRT;
    rescaleRT.create(netScale, netScale);

    // --------------------------- Create the Hierarchy ---------------------------

    std::shared_ptr<ogmaneo::Resources> res = std::make_shared<ogmaneo::Resources>();

    res->create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::Architect arch;
    arch.initialize(1234, res);

    // 3 input layers for RGB
    arch.addInputLayer(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y));
    arch.addInputLayer(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y));
    arch.addInputLayer(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y));

    for (int l = 0; l < chunkLayers; l++)
        arch.addHigherLayer(ogmaneo::Vec2i(layerSizes[l], layerSizes[l]), ogmaneo::_chunk);

    // Generate the hierarchy
    std::shared_ptr<ogmaneo::Hierarchy> h = arch.generateHierarchy();

    if (enableDebugWindow)
        debugWindow.registerHierarchy(res, h);

    bool saveArchitectAndHierarchy = false;
	
    if (saveArchitectAndHierarchy)
        arch.save("Video_Prediction.oar");

    // Input and prediction fields for color components
    ValueField2D inputFieldR(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y), 0.0f);
    ValueField2D inputFieldG(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y), 0.0f);
    ValueField2D inputFieldB(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y), 0.0f);
    ValueField2D predFieldR(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y), 0.0f);
    ValueField2D predFieldG(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y), 0.0f);
    ValueField2D predFieldB(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y), 0.0f);

    std::cout << "Running through capture: " << fileName << std::endl;

    int captureLength = static_cast<int>(capture.get(CAP_PROP_FRAME_COUNT));

    // Calculate actual number of frames
    int i = 1;
    for (; i <= captureLength; i++) {
        capture >> frame;

        if (frame.empty())
            break;
    }
	
    captureLength = i;

    std::cout << "Capture has " << captureLength << " frames" << std::endl;

    std::vector<float> errors(captureLength, 0.0f);

    // Unit Gaussian noise for input corruption
    std::normal_distribution<float> noiseDist(0.0f, 1.0f);

    // Training time
    const int numIter = 16;

    // UI update resolution
    const int progressBarLength = 40;
    const int progressUpdateTicks = 1;

    bool quit = false;
    bool graph = true;

    float graphScaleX;
    float graphScaleY;

    if (useBullFinchMovie) {
        graphScaleX = 1.0f;
        graphScaleY = 25.0f;
    }
    else {
        graphScaleX = 3.0f;
        graphScaleY = 25.0f;
    }

    // Train for a bit
    for (int iter = 0; iter < numIter && !quit; iter++) {
        std::cout << "Iteration " << (iter + 1) << " of " << numIter << ":" << std::endl;

        int currentFrame = 0;
        float movieError = 0.0f;

        capture.set(CAP_PROP_POS_FRAMES,0.0f);

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

            if (currentFrame > captureLength)
                break;

            // Convert to SFML image
            sf::Image img;

            img.create(frame.cols, frame.rows);

            for (unsigned int x = 0; x < img.getSize().x; x++)
                for (unsigned int y = 0; y < img.getSize().y; y++) {
                    sf::Uint8 r = frame.data[(x + y * img.getSize().x) * 3 + 2];
                    sf::Uint8 g = frame.data[(x + y * img.getSize().x) * 3 + 1];
                    sf::Uint8 b = frame.data[(x + y * img.getSize().x) * 3 + 0];

                    img.setPixel(x, y, sf::Color(r, g, b));
                }

            // To SFML texture
            sf::Texture tex;
            tex.loadFromImage(img);
            tex.setSmooth(true);

            // Rescale using render target
            float scale = videoScale * std::min(static_cast<float>(rescaleRT.getSize().x) / img.getSize().x, static_cast<float>(rescaleRT.getSize().y) / img.getSize().y);

            sf::Sprite s;
            s.setPosition(rescaleRT.getSize().x * 0.5f, rescaleRT.getSize().y * 0.5f);
            s.setTexture(tex);
            s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));
            s.setScale(scale, scale);

            rescaleRT.clear();
            rescaleRT.draw(s);
            rescaleRT.display();

            // SFML image from rescaled frame
            sf::Image reImg = rescaleRT.getTexture().copyToImage();

            float predError = 0.0;
            // Get input buffers
            for (unsigned int x = 0; x < reImg.getSize().x; x++)
                for (unsigned int y = 0; y < reImg.getSize().y; y++) {
                    sf::Color c = reImg.getPixel(x, y);

                    float errr = c.r / 255.0f - predFieldR.getValue(ogmaneo::Vec2i(x, y));
                    float errg = c.g / 255.0f - predFieldG.getValue(ogmaneo::Vec2i(x, y));
                    float errb = c.b / 255.0f - predFieldB.getValue(ogmaneo::Vec2i(x, y));
                    predError += ((errr * errr) + (errg * errg) + (errb * errb)) / 3.0f;

                    inputFieldR.setValue(ogmaneo::Vec2i(x, y), c.r / 255.0f * (1.0f - blendPred) + predFieldR.getValue(ogmaneo::Vec2i(x, y)) * blendPred);
                    inputFieldG.setValue(ogmaneo::Vec2i(x, y), c.g / 255.0f * (1.0f - blendPred) + predFieldG.getValue(ogmaneo::Vec2i(x, y)) * blendPred);
                    inputFieldB.setValue(ogmaneo::Vec2i(x, y), c.b / 255.0f * (1.0f - blendPred) + predFieldB.getValue(ogmaneo::Vec2i(x, y)) * blendPred);
                }

            errors[currentFrame] = predError / (reImg.getSize().x * reImg.getSize().y);

            std::vector<ogmaneo::ValueField2D> inputVector = { inputFieldR, inputFieldG, inputFieldB };

            h->simStep(inputVector, true);

            predFieldR = h->getPredictions()[0];
            predFieldG = h->getPredictions()[1];
            predFieldB = h->getPredictions()[2];

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

                while (window.pollEvent(windowEvent)) {
                    switch (windowEvent.type) {
                    case sf::Event::Closed:
                        quit = true;
                        break;
                    default:
                        break;
                    }
                }

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                    quit = true;
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::G))
                    graph = !graph;

                window.clear();

                // Show progress bar
                sf::RectangleShape rs;
                rs.setPosition(8.0f, 8.0f);
                rs.setSize(sf::Vector2f(128.0f * ratio, 32.0f));
                rs.setFillColor(sf::Color::Red);

                window.draw(rs);

                float errorTotal = 0.0f;

                if (graph) {
                    // Horizontal grid lines
                    for (int i = 0; i <= 10; i++) {
                        rs.setPosition(8.0f, windowHeight - 16.0f - graphScaleY * i);
                        rs.setSize(sf::Vector2f(graphScaleX * captureLength, 1.0f));
                        rs.setFillColor(sf::Color::White);

                        window.draw(rs);
                    }

                    // Vertical error bars
                    for (int i = 0; i < captureLength; i++) {
                        rs.setPosition(8.0f + graphScaleX * i, windowHeight - 16.0f - graphScaleY * 10.0f * errors[i]);
                        rs.setSize(sf::Vector2f(graphScaleX, graphScaleY * 10.0f * errors[i]));

                        if (i <= currentFrame)
                            rs.setFillColor(sf::Color::Red);
                        else
                            rs.setFillColor(sf::Color::Green);

                        window.draw(rs);

                        errorTotal += errors[i];
                    }

                    // Horizontal average error line
                    rs.setPosition(8.0f, windowHeight - 16.0f - graphScaleY * 10.0f * frameSkip * (errorTotal / captureLength));
                    rs.setFillColor(sf::Color::Yellow);
                    rs.setSize(sf::Vector2f(graphScaleX * captureLength, 2.0f));

                    window.draw(rs);
                }

                // Progress bar outline
                rs.setPosition(8.0f, 8.0f);
                rs.setFillColor(sf::Color::Transparent);
                rs.setOutlineColor(sf::Color::White);
                rs.setOutlineThickness(2.0f);
                rs.setSize(sf::Vector2f(128.0f, 32.0f));

                window.draw(rs);

                std::string st;
                st += std::to_string(static_cast<int>(ratio * 100.0f)) +
                    "% (pass " + std::to_string(iter + 1) + " of " + std::to_string(numIter) + ") " +
                    std::to_string(currentFrame) + "/" + std::to_string(captureLength) + " MSE: " +
                    std::to_string(predError / (reImg.getSize().x * reImg.getSize().y));

                sf::Text t;
                t.setFont(font);
                t.setCharacterSize(16);
                t.setString(st);
                t.setPosition(144.0f, 12.0f);
                t.setColor(sf::Color::White);

                window.draw(t);

                if (graph) {
                    std::string st2;
                    st2 += std::to_string(frameSkip * errorTotal / captureLength);
                    t.setString(st2);
                    t.setPosition(16.0f + graphScaleX * captureLength, windowHeight - 16.0f - graphScaleY * 10.0f * frameSkip * (errorTotal / captureLength));

                    window.draw(t);
                }

                window.display();

                if (enableDebugWindow) {
                    debugWindow.update();
                    debugWindow.display();
                }
            }

        } while (!frame.empty() && !quit);

        // Make sure bar is at 100%
        std::cout << "\r";
        std::cout << "[";

        for (int i = 0; i < progressBarLength; i++)
            std::cout << "=";

        std::cout << "] 100%";

        std::cout << std::endl;

        if (saveArchitectAndHierarchy) {
            std::string fileName = "VideoPrediction" + std::to_string(iter) + ".ohr";
            h->save(*res->getComputeSystem(), fileName);
        }
    }

    // ---------------------------- Presentation Simulation Loop -----------------------------

    window.setVerticalSyncEnabled(true);
    quit = false;

    do {
        // ----------------------------- Input -----------------------------

        sf::Event windowEvent;

        while (window.pollEvent(windowEvent)) {
            switch (windowEvent.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            default:
                break;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
            quit = true;

        window.clear();

        std::vector<ogmaneo::ValueField2D> inputVector = { predFieldR, predFieldG, predFieldB };
        h->simStep(inputVector, false);

        predFieldR = h->getPredictions()[0];
        predFieldG = h->getPredictions()[1];
        predFieldB = h->getPredictions()[2];

        sf::Image img;

        img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);

        for (unsigned int x = 0; x < rescaleRT.getSize().x; x++)
            for (unsigned int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c;

                c.r = static_cast<sf::Uint8>(255.0f * std::min(1.0f, std::max(0.0f, predFieldR.getValue(ogmaneo::Vec2i(x, y)))));
                c.g = static_cast<sf::Uint8>(255.0f * std::min(1.0f, std::max(0.0f, predFieldG.getValue(ogmaneo::Vec2i(x, y)))));
                c.b = static_cast<sf::Uint8>(255.0f * std::min(1.0f, std::max(0.0f, predFieldB.getValue(ogmaneo::Vec2i(x, y)))));

                img.setPixel(x, y, c);
            }

        sf::Texture tex;
        tex.loadFromImage(img);

        float scale = videoScale * std::min(static_cast<float>(window.getSize().x) / img.getSize().x, static_cast<float>(window.getSize().y) / img.getSize().y);

        sf::Sprite s;
        s.setPosition(window.getSize().x * 0.5f, window.getSize().y * 0.5f);
        s.setTexture(tex);
        s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));
        s.setScale(sf::Vector2f(scale, scale));

        window.draw(s);

        window.display();

        if (enableDebugWindow) {
            debugWindow.update();
            debugWindow.display();
        }

    } while (!quit);

    return 0;
}