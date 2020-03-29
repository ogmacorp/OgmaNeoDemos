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

#include <ogmaneo/Hierarchy.h>
#include <ogmaneo/ImageEncoder.h>

using namespace ogmaneo;
using namespace cv;

int main() {
    // Capture file name
    std::string fileName = "resources/Bullfinch192.mp4";

    std::string encFileName = "videoPrediction.oenc";
    std::string hFileName = "videoPrediction.ohr";

    // Initialize a random number generator
    std::mt19937 rng(time(nullptr));

    // Uniform distribution in [0, 1]
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    const unsigned int windowWidth = 800;
    const unsigned int windowHeight = 600;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Video Test", sf::Style::Default);

    // Uncap framerate
    window.setFramerateLimit(0);

    sf::Font font;

#if defined(_WINDOWS)
    font.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#elif defined(__APPLE__)
    font.loadFromFile("/Library/Fonts/Courier New.ttf");
#else
    font.loadFromFile("/usr/share/fonts/TTF/VeraMono.ttf");
#endif

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

    const float videoScale = 1.0f; // Rescale ratio
    const unsigned int rescaleWidth = videoScale * movieWidth;
    const unsigned int rescaleHeight = videoScale * movieHeight;

    // Video rescaling render target
    sf::RenderTexture rescaleRT;
    rescaleRT.create(rescaleWidth, rescaleHeight);

    // --------------------------- Create the Hierarchy ---------------------------

    // Create hierarchy
    ComputeSystem::setNumThreads(8);
    ComputeSystem cs;

    Int3 hiddenSize(12, 12, 16);

    ImageEncoder::VisibleLayerDesc vld;
    vld.size = Int3(rescaleRT.getSize().x, rescaleRT.getSize().y, 3);
    vld.radius = 12;

    ImageEncoder enc;

    std::vector<Hierarchy::LayerDesc> lds(4);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(6, 6, 16);

        lds[i].ffRadius = lds[i].pRadius = 2;

        lds[i].ticksPerUpdate = 2;
        lds[i].temporalHorizon = 4;
    }

    Hierarchy h;
    
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

    // Frame skip
    int frameSkip = 1; // 1 means no frame skip (stride of 1)

    // UI update resolution
    const int progressBarLength = 40;
    const int progressUpdateTicks = 1;

    bool quit = false;
    bool graph = true;
    bool loadHierarchy = false;
    bool saveHierarchy = true;

    const float graphScaleX = 1.0f;
    const float graphScaleY = 25.0f;

    if (!loadHierarchy) {
        // Initialize hierarchy randomly
        enc.initRandom(cs, hiddenSize, { vld });
        h.initRandom(cs, { hiddenSize }, { prediction }, lds);

        // Train for a bit
        for (int iter = 0; iter < numIter && !quit; iter++) {
            std::cout << "Iteration " << (iter + 1) << " of " << numIter << ":" << std::endl;

            int currentFrame = 0;
            float movieError = 0.0f;

            capture.set(CAP_PROP_POS_FRAMES, 0.0f);

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

                for (int x = 0; x < img.getSize().x; x++)
                    for (int y = 0; y < img.getSize().y; y++) {
                        sf::Uint8 r = frame.data[2 + x * 3 + y * 3 * img.getSize().x]; // Reverse order so it's BGR -> RGB
                        sf::Uint8 g = frame.data[1 + x * 3 + y * 3 * img.getSize().x]; // OpencV is a different matrix order than OgmaNeo
                        sf::Uint8 b = frame.data[0 + x * 3 + y * 3 * img.getSize().x];

                        img.setPixel(x, y, sf::Color(r, g, b));
                    }

                // To SFML texture
                sf::Texture tex;
                tex.loadFromImage(img);
                tex.setSmooth(true);

                // Rescale using render target
                float scale = std::min(static_cast<float>(rescaleRT.getSize().x) / img.getSize().x, static_cast<float>(rescaleRT.getSize().y) / img.getSize().y);

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

                // Reconstruct last prediction
                enc.reconstruct(cs, &h.getPredictionCs(0));

                std::vector<float> pred = enc.getVisibleLayer(0).reconActs;
                std::vector<float> input(pred.size());

                float predError = 0.0f;

                // Get input buffers
                for (int x = 0; x < reImg.getSize().x; x++)
                    for (int y = 0; y < reImg.getSize().y; y++) {
                        sf::Color c = reImg.getPixel(x, y);

                        float errR = c.r / 255.0f - pred[0 + y * 3 + x * 3 * rescaleRT.getSize().y];
                        float errG = c.g / 255.0f - pred[1 + y * 3 + x * 3 * rescaleRT.getSize().y];
                        float errB = c.b / 255.0f - pred[2 + y * 3 + x * 3 * rescaleRT.getSize().y];

                        predError += ((errR * errR) + (errG * errG) + (errB * errB)) / 3.0f;

                        input[0 + y * 3 + x * 3 * rescaleRT.getSize().y] = c.r / 255.0f;
                        input[1 + y * 3 + x * 3 * rescaleRT.getSize().y] = c.g / 255.0f;
                        input[2 + y * 3 + x * 3 * rescaleRT.getSize().y] = c.b / 255.0f;
                    }

                errors[currentFrame] = predError / (reImg.getSize().x * reImg.getSize().y);

                // Step pre-encoder and hierarchy
                enc.step(cs, { &input }, true);

                h.step(cs, { &enc.getHiddenCs() }, true);

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

        if (saveHierarchy) {
            std::cout << "Saving hierarchy as " << hFileName << " and " << encFileName << std::endl;

            {
                std::ofstream toFile(encFileName);

                enc.writeToStream(toFile);
            }

            {
                std::ofstream toFile(hFileName);

                h.writeToStream(toFile);
            }
        }
    }
    else {
        std::cout << "Loading hierarchy from " << hFileName << " and " << encFileName << std::endl;

        {
            std::ifstream fromFile(encFileName);

            enc.readFromStream(fromFile);
        }

        {
            std::ifstream fromFile(hFileName);

            h.readFromStream(fromFile);
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

        h.step(cs, { &h.getPredictionCs(0) }, false);

        enc.reconstruct(cs, &h.getPredictionCs(0));

        std::vector<float> pred = enc.getVisibleLayer(0).reconActs;

        sf::Image img;

        img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);

        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c;

                c.r = static_cast<sf::Uint8>(255.0f * std::min(1.0f, std::max(0.0f, pred[0 + y * 3 + x * 3 * rescaleRT.getSize().y])));
                c.g = static_cast<sf::Uint8>(255.0f * std::min(1.0f, std::max(0.0f, pred[1 + y * 3 + x * 3 * rescaleRT.getSize().y])));
                c.b = static_cast<sf::Uint8>(255.0f * std::min(1.0f, std::max(0.0f, pred[2 + y * 3 + x * 3 * rescaleRT.getSize().y])));

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

    return 0;
}