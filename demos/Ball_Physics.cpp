// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
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

#include <neo/Architect.h>
#include <neo/Hierarchy.h>

int main() {
    std::mt19937 generator(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(800, 800), "Physics Test", sf::Style::Default);

    // Render target for scene
    sf::RenderTexture rescaleRT;
    rescaleRT.create(48, 48);

    // Create the hierarchy
    std::shared_ptr<ogmaneo::Resources> res = std::make_shared<ogmaneo::Resources>();

    res->create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::Architect arch;
    arch.initialize(1234, res);

    // 1 input layer, greyscale image of scene
    arch.addInputLayer(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y))
        .setValue("in_p_alpha", 0.05f)
        .setValue("in_p_radius", 10);

    // 8 chunk layers
    for (int l = 0; l < 7; l++)
        arch.addHigherLayer(ogmaneo::Vec2i(128, 128), ogmaneo::_chunk)
        .setValue("sfc_chunkSize", ogmaneo::Vec2i(8, 8))
        .setValue("sfc_ff_radius", 10)
        .setValue("sfc_r_radius", 10)
        .setValue("hl_poolSteps", 2)
        .setValue("sfc_numSamples", 1)
        .setValue("sfc_gamma", 2.0f)
        .setValue("p_alpha", 0.05f)
        .setValue("p_beta", 0.1f)
        .setValue("p_radius", 10);

    std::shared_ptr<ogmaneo::Hierarchy> h = arch.generateHierarchy();

    // Fields for inputs and predictions
    ogmaneo::ValueField2D inputField(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y));
    ogmaneo::ValueField2D predField(ogmaneo::Vec2i(rescaleRT.getSize().x, rescaleRT.getSize().y));

    // ----------------------------- Physics ------------------------------

    // Box2D world
    std::shared_ptr<b2World> world = std::make_shared<b2World>(b2Vec2(0.0f, -9.81f));

    // Number of pixels in a physical unit
    const float pixelsPerMeter = 64.0f;

    // Ground box
    const float groundWidth = 5000.0f;
    const float groundHeight = 5.0f;

    // Create ground
    {
        b2BodyDef groundBodyDef;
        groundBodyDef.position.Set(0.0f, -groundHeight * 0.5f);

        b2Body* groundBody = world->CreateBody(&groundBodyDef);

        b2PolygonShape groundBox;
        groundBox.SetAsBox(groundWidth * 0.5f, groundHeight * 0.5f);

        groundBody->CreateFixture(&groundBox, 0.0f);
    }

    // Wall boxes
    const float wallWidth = 5.0f;
    const float wallHeight = 5000.0f;

    // Create wall
    {
        b2BodyDef leftWallBodyDef;
        leftWallBodyDef.position.Set(-10.0f, 0.0f);

        b2Body* leftWallBody = world->CreateBody(&leftWallBodyDef);

        b2PolygonShape leftWallBox;
        leftWallBox.SetAsBox(wallWidth * 0.5f, wallHeight * 0.5f);

        leftWallBody->CreateFixture(&leftWallBox, 0.0f);
    }

    // Create wall
    {
        b2BodyDef rightWallBodyDef;
        rightWallBodyDef.position.Set(10.0f, 0.0f);

        b2Body* rightWallBody = world->CreateBody(&rightWallBodyDef);

        b2PolygonShape rightWallBox;
        rightWallBox.SetAsBox(wallWidth * 0.5f, wallHeight * 0.5f);

        rightWallBody->CreateFixture(&rightWallBox, 0.0f);
    }

    // Create ball
    b2Vec2 ballStart(0.0f, 8.2f);

    b2BodyDef ballBodyDef;
    ballBodyDef.position = ballStart;
    ballBodyDef.type = b2BodyType::b2_dynamicBody;

    b2Body* ballBody = world->CreateBody(&ballBodyDef);

    b2CircleShape ballShape;
    ballShape.m_radius = 0.6f;

    b2Fixture* ballFixture = ballBody->CreateFixture(&ballShape, 5.0f);

    ballFixture->SetFriction(0.01f);
    ballFixture->SetRestitution(0.82f);

    // Frames per episode
    const int simFrames = 60;

    // Generation mode flag
    bool genMode = false;

    // Textures for visualizing states
    std::vector<sf::Texture> layerTextures(h->getPredictor().getHierarchy().getNumLayers());

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    int simFrame = simFrames;

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

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            quit = true;

        // If time for a new episode
        if (simFrame >= simFrames) {
            simFrame = 0;

            std::uniform_real_distribution<float> velDistX(-8.0f, 8.0f);
            std::uniform_real_distribution<float> velDistY(-8.0f, 8.0f);

            // Set up ball position and velocity
            ballBody->SetLinearVelocity(b2Vec2(velDistX(generator), velDistY(generator)));
            ballBody->SetAngularVelocity(0.0f);
            ballBody->SetTransform(ballStart, 0.0f);

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::G))
                genMode = !genMode;

            window.setFramerateLimit(genMode ? 60 : 0);
        }

        simFrame++;

        // Step physics simulation
        int subSteps = 3;

        for (int ss = 0; ss < subSteps; ss++) {
            world->ClearForces();

            world->Step(1.0f / 10.0f / subSteps, 8, 8);
        }

        // Render the scene to the rescaleRT
        rescaleRT.clear();

        sf::View v;

        v.setCenter(pixelsPerMeter * sf::Vector2f(0.0f, -7.5f));
        rescaleRT.setView(v);

        // Ground
        {
            sf::RectangleShape rs;
            rs.setSize(pixelsPerMeter * sf::Vector2f(groundWidth, groundHeight));
            rs.setOrigin(pixelsPerMeter * sf::Vector2f(groundWidth * 0.5f, groundHeight * 0.5f));
            rs.setPosition(pixelsPerMeter * sf::Vector2f(0.0f, groundHeight * 0.5f));

            rescaleRT.draw(rs);
        }

        // Wall
        {
            sf::RectangleShape rs;
            rs.setSize(pixelsPerMeter * sf::Vector2f(wallWidth, wallHeight));
            rs.setOrigin(pixelsPerMeter * sf::Vector2f(wallWidth * 0.5f, wallHeight * 0.5f));
            rs.setPosition(pixelsPerMeter * sf::Vector2f(-10.0f, 0.0f));

            rescaleRT.draw(rs);
        }

        // Wall
        {
            sf::RectangleShape rs;
            rs.setSize(pixelsPerMeter * sf::Vector2f(wallWidth, wallHeight));
            rs.setOrigin(pixelsPerMeter * sf::Vector2f(wallWidth * 0.5f, wallHeight * 0.5f));
            rs.setPosition(pixelsPerMeter * sf::Vector2f(10.0f, 0.0f));

            rescaleRT.draw(rs);
        }

        // Ball
        {
            sf::CircleShape circS;
            circS.setRadius(pixelsPerMeter * ballShape.m_radius);
            circS.setOrigin(pixelsPerMeter * sf::Vector2f(ballShape.m_radius, ballShape.m_radius));
            circS.setPosition(pixelsPerMeter * sf::Vector2f(ballBody->GetPosition().x, -ballBody->GetPosition().y));

            rescaleRT.draw(circS);
        }

        // Finish rendering
        rescaleRT.display();

        // Show on main window
        window.clear();

        sf::Image rescaleImg = rescaleRT.getTexture().copyToImage();

        // Load into input field
        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c = rescaleImg.getPixel(x, y);

                float mono = 0.333f * (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f);

                inputField.setValue(ogmaneo::Vec2i(x, y), mono);
            }

        // Feed first 5 frames from image, even when generating ("seed" sequence)
        if (simFrame > 5 && genMode) {
            inputField = predField;

            std::vector<ogmaneo::ValueField2D> inputVector = { inputField };

            h->simStep(inputVector, false);
        }
        else {
            std::vector<ogmaneo::ValueField2D> inputVector = { inputField };

            h->simStep(inputVector, true);
        }

        // Retrieve prediction
        predField = h->getPredictions().front();

        // Display prediction
        sf::Image img;

        img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);

        // Load back into image
        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c;

                c.r = c.g = c.b = 255.0f * std::min(1.0f, std::max(0.0f, predField.getValue(ogmaneo::Vec2i(x, y))));

                img.setPixel(x, y, c);
            }

        // Load image into texture
        sf::Texture tex;

        tex.loadFromImage(img);

        // Display
        sf::Sprite s;

        s.setPosition(window.getSize().x * 0.5f, window.getSize().y * 0.5f);

        // If in generation mode, show prediction, otherwise show the training data
        s.setTexture(genMode ? tex : rescaleRT.getTexture());

        s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));

        // Scale up to size of main window
        float scale = std::min(static_cast<float>(window.getSize().x) / img.getSize().x, static_cast<float>(window.getSize().y) / img.getSize().y);

        s.setScale(sf::Vector2f(scale, scale));

        window.draw(s);

        // If pressing K, show SDRs
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::K)) {
            float xOffset = 0.0f;

            float SDRScale = 1.0f;

            // Show all layer SDRs
            for (int l = 0; l < layerTextures.size(); l++) {
                // Temporary data buffer (host side)
                std::vector<float> data(h->getPredictor().getHierarchy().getLayer(l)._sf->getHiddenSize().x * h->getPredictor().getHierarchy().getLayer(l)._sf->getHiddenSize().y * 2);

                // Retrieve SDR. Need low level access for this
                res->getComputeSystem()->getQueue().enqueueReadImage(h->getPredictor().getHierarchy().getLayer(l)._sf->getHiddenStates()[ogmaneo::_back], CL_TRUE, { 0, 0, 0 }, { static_cast<cl::size_type>(h->getPredictor().getHierarchy().getLayer(l)._sf->getHiddenSize().x), static_cast<cl::size_type>(h->getPredictor().getHierarchy().getLayer(l)._sf->getHiddenSize().y), 1 }, 0, 0, data.data());

                // Convert into image
                sf::Image img;

                img.create(h->getPredictor().getHierarchy().getLayer(l)._sf->getHiddenSize().x, h->getPredictor().getHierarchy().getLayer(l)._sf->getHiddenSize().y);

                for (int x = 0; x < img.getSize().x; x++)
                    for (int y = 0; y < img.getSize().y; y++) {
                        sf::Color c = sf::Color::White;

                        c.r = c.b = c.g = 255.0f * data[(x + y * img.getSize().x) * 2 + 0];

                        img.setPixel(x, y, c);
                    }

                // Display
                layerTextures[l].loadFromImage(img);

                sf::Sprite s;

                s.setTexture(layerTextures[l]);

                s.setPosition(xOffset, window.getSize().y - img.getSize().y * SDRScale);

                s.setScale(SDRScale, SDRScale);

                window.draw(s);

                xOffset += img.getSize().x * SDRScale;
            }
        }

        window.display();
    } while (!quit);

    return 0;
}