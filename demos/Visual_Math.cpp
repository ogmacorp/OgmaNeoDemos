// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <Box2D/Box2D.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <random>

#include <aogmaneo/Sheet.h>
#include <aogmaneo/ImageEncoder.h>

using namespace aon;

int main() {
    std::mt19937 rng(time(nullptr));

    setNumThreads(8);

    sf::RenderWindow window;
    window.setFramerateLimit(10);

    window.create(sf::VideoMode(128 * 3, 128), "Visual_Math", sf::Style::Default);

    sf::RenderTexture numberRT;
    numberRT.create(32, 32);
   
    Int3 hiddenSize(8, 8, 16);

    Array<ImageEncoder::VisibleLayerDesc> vlds(1);
    vlds[0].size = Int3(numberRT.getSize().x, numberRT.getSize().y, 1);
    vlds[0].radius = 8;

    ImageEncoder enc;
    enc.initRandom(hiddenSize, vlds);
    enc.alpha = 0.01f;
    enc.gamma = 1.0f;

    Array<Sheet::InputDesc> inputDescs(2);
    Array<Sheet::OutputDesc> outputDescs(1);

    inputDescs[0].size = hiddenSize;
    inputDescs[0].radius = 4;

    inputDescs[1].size = hiddenSize;
    inputDescs[1].radius = 4;

    outputDescs[0].size = hiddenSize;
    outputDescs[0].radius = 4;

    Sheet sheet;
    sheet.initRandom(inputDescs, 1, outputDescs, Int3(8, 8, 16));

    sf::Font font;
    font.loadFromFile("resources/Hack-Regular.ttf");

    int base = 10;
    int numDigits = 2;

    int maxAdd = (std::pow(base, numDigits) - 1) / 2;

    std::uniform_int_distribution<int> addDist(0, maxAdd);

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    bool trainMode = true;
    bool showMode = false;

    bool tPressedPrev = false;
    bool sPressedPrev = false;

    Array<int> numbers(3);
    Array<FloatBuffer> images(numbers.size());
    Array<sf::Texture> textures(numbers.size());
    Array<IntBuffer> imageCSDRs(numbers.size());
    int prevSum = 0;

    do {
        // ----------------------------- Input -----------------------------

        // Receive events
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

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            bool tPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::T);

            if (tPressed && !tPressedPrev)
                trainMode = !trainMode;
        
            tPressedPrev = tPressed; 

            bool sPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::S);

            if (sPressed && !sPressedPrev)
                showMode = !showMode;
        
            sPressedPrev = sPressed; 
        }

        if (!showMode) {
            int subIters = trainMode ? 100 : 1;

            for (int it = 0; it < subIters; it++) {
                numbers[0] = addDist(rng);
                numbers[1] = addDist(rng);
                numbers[2] = prevSum; //numbers[0] + numbers[1];

                for (int n = 0; n < numbers.size(); n++) {
                    numberRT.clear(sf::Color::Black);

                    sf::Text text;
                    text.setFont(font);
                    text.setCharacterSize(24);
                    text.setString(std::to_string(numbers[n]));

                    numberRT.draw(text);

                    // Finish rendering
                    numberRT.display();

                    sf::Image rescaleImg = numberRT.getTexture().copyToImage();
                    textures[n] = numberRT.getTexture();

                    images[n].resize(numberRT.getSize().x * numberRT.getSize().y);

                    // Load into input field
                    for (int x = 0; x < numberRT.getSize().x; x++)
                        for (int y = 0; y < numberRT.getSize().y; y++) {
                            sf::Color c = rescaleImg.getPixel(x, y);

                            float mono = 0.333f * (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f);

                            images[n][y + x * numberRT.getSize().y] = mono;
                        }

                    // Encode
                    Array<const FloatBuffer*> encInputs(1);
                    encInputs[0] = &images[n];
                    enc.step(encInputs, true);

                    imageCSDRs[n] = enc.getHiddenCs();
                }

                Array<const IntBuffer*> inputCSDRs(2);
                Array<const IntBuffer*> targetCSDRs(1);

                inputCSDRs[0] = &imageCSDRs[0];
                inputCSDRs[1] = &imageCSDRs[1];
                targetCSDRs[0] = &imageCSDRs[2];

                sheet.step(inputCSDRs, targetCSDRs, 8, true, true);

                prevSum = numbers[0] + numbers[1];
            }
        }

        // Show on main window
        window.clear();

        // Reconstruct
        enc.reconstruct(&sheet.getPredictionCs(0));

        // Retrieve reconstructed prediction
        FloatBuffer pred = enc.getVisibleLayer(0).reconstruction;

        sf::Texture resultTexture;

        {
            // Display prediction
            sf::Image img;

            img.create(numberRT.getSize().x, numberRT.getSize().y);

            // Load back into image
            for (int x = 0; x < numberRT.getSize().x; x++)
                for (int y = 0; y < numberRT.getSize().y; y++) {
                    sf::Color c;

                    c.r = c.g = c.b = 255.0f * pred[y + x * numberRT.getSize().y];

                    img.setPixel(x, y, c);
                }

            // Load image into texture

            resultTexture.loadFromImage(img);
        }

        // Display
        sf::Sprite s;
        s.setScale(4, 4);

        s.setPosition(0, 0);

        // If in generation mode, show prediction, otherwise show the training data
        s.setTexture(textures[0]);
        
        window.draw(s);

        s.setPosition(128, 0);

        s.setTexture(textures[1]);

        window.draw(s);

        s.setPosition(256, 0);

        s.setTexture(resultTexture);

        window.draw(s);

        window.display();
    } while (!quit);

    return 0;
}
