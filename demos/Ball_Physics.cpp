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

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/ImageEncoder.h>

using namespace aon;

int main() {
    std::mt19937 generator(time(nullptr));

    setNumThreads(8);

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;
    window.setFramerateLimit(0); // No limit

    window.create(sf::VideoMode(800, 800), "Physics Test", sf::Style::Default);

    // Render target for scene
    sf::RenderTexture rescaleRT;
    rescaleRT.create(32, 32);
   
    // --------------------------- Create the Hierarchy ---------------------------

    // Create hierarchy
    Int3 hiddenSize(8, 8, 32);

    Array<ImageEncoder::VisibleLayerDesc> vlds(1);
    vlds[0].size = Int3(rescaleRT.getSize().x, rescaleRT.getSize().y, 1);
    vlds[0].radius = 8;

    ImageEncoder enc;
    enc.initRandom(hiddenSize, vlds);

    // Create hierarchy
    Array<Hierarchy::LayerDesc> lds(3);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(4, 4, 32);

        lds[i].ffRadius = 2;
        lds[i].pRadius = 2;

        lds[i].ticksPerUpdate = 2;
        lds[i].temporalHorizon = 2;
    }

    Hierarchy h;

    Array<Hierarchy::IODesc> ioDescs(1);
    ioDescs[0] = Hierarchy::IODesc(hiddenSize, IOType::prediction, 2, 2, 2);

    Array<const IntBuffer*> inputCIs(1);
    inputCIs[0] = &enc.getHiddenCIs();

    h.initRandom(ioDescs, lds);

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
    ballShape.m_radius = 1.4f;

    b2Fixture* ballFixture = ballBody->CreateFixture(&ballShape, 5.0f);

    ballFixture->SetFriction(0.01f);
    ballFixture->SetRestitution(0.82f);

    // Frames per episode
    const int simFrames = 60;

    // Generation mode flag
    bool genMode = false;

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    int simFrame = simFrames;

    bool gPressedPrev = false;

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

            bool gPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::G);

            if (gPressed && !gPressedPrev)
                genMode = !genMode;

            gPressedPrev = gPressed;
        }

        // If time for a new episode
        if (simFrame >= simFrames) {
            simFrame = 0;

            std::uniform_real_distribution<float> velDistX(-8.0f, 8.0f);
            std::uniform_real_distribution<float> velDistY(-8.0f, 8.0f);

            // Set up ball position and velocity
            ballBody->SetLinearVelocity(b2Vec2(velDistX(generator), velDistY(generator)));
            ballBody->SetAngularVelocity(0.0f);
            ballBody->SetTransform(ballStart, 0.0f);

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

        ByteBuffer image(rescaleRT.getSize().x * rescaleRT.getSize().y);

        // Load into input field
        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c = rescaleImg.getPixel(x, y);

                float mono = 0.333f * (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f);

                image[y + x * rescaleRT.getSize().y] = mono * 255.0f;
            }

        // Feed first 5 frames from image, even when generating ("seed" sequence)
        if (simFrame > 5 && genMode) {
            inputCIs[0] = &h.getPredictionCIs(0);
            h.step(inputCIs, false);
            inputCIs[0] = &enc.getHiddenCIs();
        }
        else {
            Array<const ByteBuffer*> images(1);
            images[0] = &image;

            enc.step(images, true);

            h.step(inputCIs, true);
        }

        // Reconstruct
        enc.reconstruct(&h.getPredictionCIs(0));

        // Retrieve reconstructed prediction
        ByteBuffer pred = enc.getVisibleLayer(0).reconstruction;

        // Display prediction
        sf::Image img;

        img.create(rescaleRT.getSize().x, rescaleRT.getSize().y);

        // Load back into image
        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c;

                c.r = c.g = c.b = pred[y + x * rescaleRT.getSize().y];

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

        window.display();
    } while (!quit);

    return 0;
}
