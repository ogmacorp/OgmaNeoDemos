// --------------------------------------------------------------------------
//	Ogma Toolkit(OTK)
//	Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
// --------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <neo/Architect.h>
#include <neo/Hierarchy.h>

#include <time.h>
#include <iostream>
#include <random>

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int main() {
    std::mt19937 generator(time(nullptr));

    // Load the dataset (level image)
    sf::Texture exampleLevel;

    exampleLevel.loadFromFile("resources/exampleLevel.png");

    // Temporary render target (RT)
    sf::RenderTexture rt;

    // Size is square of level height
    rt.create(exampleLevel.getSize().y, exampleLevel.getSize().y);

    // ------------------------------------- Create Predictor -------------------------------------

    std::shared_ptr<ogmaneo::Resources> res = std::make_shared<ogmaneo::Resources>();

    res->create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::Architect arch;
    arch.initialize(1234, res); // Seed and provide resources

    // 3 input layers - RGB
    arch.addInputLayer(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y))
        .setValue("in_p_alpha", 0.05f)
        .setValue("in_p_radius", 12);

    arch.addInputLayer(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y))
        .setValue("in_p_alpha", 0.03f)
        .setValue("in_p_radius", 10);

    arch.addInputLayer(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y))
        .setValue("in_p_alpha", 0.03f)
        .setValue("in_p_radius", 10);

    // 5 chunk encoder layers with some parameter settings
    for (int l = 0; l < 6; l++)
        arch.addHigherLayer(ogmaneo::Vec2i(128, 128), ogmaneo::_chunk)
        .setValue("sfc_chunkSize", ogmaneo::Vec2i(8, 8))
        .setValue("sfc_ff_radius", 10)
        .setValue("hl_poolSteps", 2)
        .setValue("sfc_gamma", 2.0f)
        .setValue("p_alpha", 0.04f)
        .setValue("p_beta", 0.1f)
        .setValue("p_radius", 10);

    // Generate the hierarchy
    std::shared_ptr<ogmaneo::Hierarchy> h = arch.generateHierarchy();

    // Layers for IO (input and prediction)
    ogmaneo::ValueField2D inputFieldR(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y), 0.0f);
    ogmaneo::ValueField2D inputFieldG(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y), 0.0f);
    ogmaneo::ValueField2D inputFieldB(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y), 0.0f);

    ogmaneo::ValueField2D predFieldR(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y), 0.0f);
    ogmaneo::ValueField2D predFieldG(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y), 0.0f);
    ogmaneo::ValueField2D predFieldB(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y), 0.0f);
  
    // Image shown containing previously generated pixels
    sf::Image showImg;
    showImg.create(rt.getSize().x, rt.getSize().y);

    // Temporary sample accumulation buffers
    ogmaneo::ValueField2D resultBufferR(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y), 0.0f);
    ogmaneo::ValueField2D resultBufferG(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y), 0.0f);
    ogmaneo::ValueField2D resultBufferB(ogmaneo::Vec2i(rt.getSize().x, rt.getSize().y), 0.0f);

    // ---------------------------- Training -----------------------------

    const int step = 1; // Step 1 pixel at a time

    // 40 iterations of training on the level
    for (int iter = 0; iter < 30; iter++) {
        std::cout << "Training iter " << (iter + 1) << std::endl;

        // Go through level horizontally one pixel at a time
        for (int x = 0; x < exampleLevel.getSize().x - exampleLevel.getSize().y; x += step) {
            rt.clear();

            // Render the level image portion currently visible
            sf::RectangleShape rs; // Rectangle shape

            rs.setSize(sf::Vector2f(rt.getSize().x, rt.getSize().y));
            rs.setTexture(&exampleLevel);
            rs.setTextureRect(sf::IntRect(x, 0, rt.getSize().x, rt.getSize().y));

            rt.draw(rs);

            // Finish rendering
            rt.display();

            // Get SFML image
            sf::Image img = rt.getTexture().copyToImage();

            // Copy image values to input fields
            for (int x = 0; x < img.getSize().x; x++)
                for (int y = 0; y < img.getSize().y; y++) {
                    sf::Color c = img.getPixel(x, y);

                    inputFieldR.setValue(ogmaneo::Vec2i(x, y), c.r / 255.0f);
                    inputFieldG.setValue(ogmaneo::Vec2i(x, y), c.g / 255.0f);
                    inputFieldB.setValue(ogmaneo::Vec2i(x, y), c.b / 255.0f);
                }

            // Step the hierarchy
            std::vector<ogmaneo::ValueField2D> inputVector = { inputFieldR, inputFieldG, inputFieldB };

            h->simStep(inputVector, true); // Learning enabled
        }
    }

    // ---------------------------- Visualization Loop -----------------------------

    sf::RenderWindow window;

    window.create(sf::VideoMode(800, 800), "Level Generation Demo", sf::Style::Default);

    // 60 fps
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    bool quit = false;

    do {
        // ----------------------------- Input -----------------------------

        // Events
        sf::Event windowEvent;

        while (window.pollEvent(windowEvent)) {
            switch (windowEvent.type)
            {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            quit = true;

        // Get predictions from last frame
        predFieldR = h->getPredictions()[0];
        predFieldG = h->getPredictions()[1];
        predFieldB = h->getPredictions()[2];

        // Shift results
        for (int x = 0; x < rt.getSize().x - 1; x++) {
            for (int y = 0; y < rt.getSize().y; y++) {
                int index1 = x + y * rt.getSize().x;
                int index2 = (x + 1) + y * rt.getSize().x;

                resultBufferR.setValue(ogmaneo::Vec2i(x, y), resultBufferR.getValue(ogmaneo::Vec2i(x + 1, y)));
                resultBufferG.setValue(ogmaneo::Vec2i(x, y), resultBufferG.getValue(ogmaneo::Vec2i(x + 1, y)));
                resultBufferB.setValue(ogmaneo::Vec2i(x, y), resultBufferB.getValue(ogmaneo::Vec2i(x + 1, y)));
            }
        }

        // Add new data (blend into result buffer - average samples)
        for (int x = 0; x < rt.getSize().x; x++) {
            float blend = (x + 1) / static_cast<float>(rt.getSize().x); // How much of the prediction enters the result buffer

            for (int y = 0; y < rt.getSize().y; y++) {
                sf::Color c;

                // Clamp colors
                float rf = std::min(1.0f, std::max(0.0f, predFieldR.getValue(ogmaneo::Vec2i(x, y))));
                float gf = std::min(1.0f, std::max(0.0f, predFieldG.getValue(ogmaneo::Vec2i(x, y))));
                float bf = std::min(1.0f, std::max(0.0f, predFieldB.getValue(ogmaneo::Vec2i(x, y))));

                // Add (running average)
                resultBufferR.setValue(ogmaneo::Vec2i(x, y), resultBufferR.getValue(ogmaneo::Vec2i(x, y)) + blend * (rf - resultBufferR.getValue(ogmaneo::Vec2i(x, y))));
                resultBufferG.setValue(ogmaneo::Vec2i(x, y), resultBufferG.getValue(ogmaneo::Vec2i(x, y)) + blend * (gf - resultBufferG.getValue(ogmaneo::Vec2i(x, y))));
                resultBufferB.setValue(ogmaneo::Vec2i(x, y), resultBufferB.getValue(ogmaneo::Vec2i(x, y)) + blend * (bf - resultBufferB.getValue(ogmaneo::Vec2i(x, y))));
            }
        }

        // Append averaged results from result buffer to the show image

        // Shift
        for (int x = 0; x < showImg.getSize().x - 1; x++) {
            for (int y = 0; y < showImg.getSize().y; y++) {
                showImg.setPixel(x, y, showImg.getPixel(x + 1, y));
            }
        }

        // Take end of result buffer and place into show image
        for (int y = 0; y < showImg.getSize().y; y++) {
            float rf = resultBufferR.getValue(ogmaneo::Vec2i(0, y));
            float gf = resultBufferG.getValue(ogmaneo::Vec2i(0, y));
            float bf = resultBufferB.getValue(ogmaneo::Vec2i(0, y));

            sf::Color c;

            c.r = rf * 255;
            c.g = gf * 255;
            c.b = bf * 255;

            // Set the pixel at the rightmost portion of the image
            showImg.setPixel(showImg.getSize().x - 1, y, c);
        }

        // Noise distribution
        std::normal_distribution<float> noiseDist(0.0f, 0.22f);

        // Set input fields
        for (int x = 0; x < rt.getSize().x; x++) {
            for (int y = 0; y < rt.getSize().y; y++) {
                inputFieldR.setValue(ogmaneo::Vec2i(x, y), resultBufferR.getValue(ogmaneo::Vec2i(x, y)) + noiseDist(generator));
                inputFieldG.setValue(ogmaneo::Vec2i(x, y), resultBufferG.getValue(ogmaneo::Vec2i(x, y)) + noiseDist(generator));
                inputFieldB.setValue(ogmaneo::Vec2i(x, y), resultBufferB.getValue(ogmaneo::Vec2i(x, y)) + noiseDist(generator));
            }
        }

        // Generate level
        std::vector<ogmaneo::ValueField2D> inputVector = { predFieldR, predFieldG, predFieldB };

        h->simStep(inputVector, false); // Learning disabled

        // Create texture from show image
        sf::Texture tex;
        tex.loadFromImage(showImg);

        // Creat sprite
        sf::Sprite s;

        s.setTexture(tex);
        s.setScale(window.getSize().x / static_cast<float>(rt.getSize().x), window.getSize().y / static_cast<float>(rt.getSize().y));

        // Show the result
        window.draw(s);

        window.display();
    } while (!quit);

    return 0;
}