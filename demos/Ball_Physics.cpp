// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2022 Ogma Intelligent Systems Corp. All rights reserved.
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

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/image_encoder.h>

//#include "vis/visadapter.h"

using namespace aon;

int main() {
    std::mt19937 generator(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(800, 800)), "Physics Test", sf::Style::Default);

    window.setFramerateLimit(0); // No limit

    // Render target for scene
    sf::RenderTexture rescaleRT(sf::Vector2u(64, 64));

    // --------------------------- Create the Hierarchy ---------------------------

    set_num_threads(8);

    // Create hierarchy
    Int3 hiddenSize(20, 20, 16);

    Array<Image_Encoder::Visible_Layer_Desc> imgVlds(1);
    imgVlds[0].size = Int3(rescaleRT.getSize().x, rescaleRT.getSize().y, 1);
    imgVlds[0].radius = 6;

    Image_Encoder enc;
    enc.init_random(hiddenSize, imgVlds);

    Array<Hierarchy::Layer_Desc> lds(2);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(10, 10, 32);
        //lds[i].temporal_size = 8;
        //lds[i].spatial_activity = 8;
    }

    Array<Hierarchy::IO_Desc> ioDescs(1);
    ioDescs[0].size = hiddenSize;
    ioDescs[0].type = prediction;
    ioDescs[0].up_radius = 4;

    Hierarchy h;
    h.init_random(ioDescs, lds);

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
    const int simFrames = 90;

    // Generation mode flag
    bool genMode = false;

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    int simFrame = simFrames;

    bool gPressedPrev = false;

    U8_Array imgb(rescaleRT.getSize().x * rescaleRT.getSize().y, 0.0f);
    Array<U8_Array_View> imgs(1);
    imgs[0] = imgb;

    do {
        // ----------------------------- Input -----------------------------

        // Receive events
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                quit = true;
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape))
                quit = true;

            bool gPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::G);

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

        // Load into input field
        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c = rescaleImg.getPixel(sf::Vector2u(x, y));

                float mono = 0.333f * (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f);

                imgb[y + x * rescaleRT.getSize().y] = mono * 255.0f;
            }

        // Feed first 5 frames from image, even when generating ("seed" sequence)
        if (simFrame > 5 && genMode) {
            Array<S32_Array_View> inputCIs(1);

            inputCIs[0] = h.get_prediction_cis(0);

            h.step(inputCIs, false);
        }
        else {
            enc.step(imgs, true, true);

            Array<S32_Array_View> inputCIs(1);

            inputCIs[0] = enc.get_hidden_cis();

            h.step(inputCIs, true);
        }

        // Reconstruct
        enc.reconstruct(h.get_prediction_cis(0));

        // Retrieve reconstructed prediction
        U8_Array pred = enc.get_reconstruction(0);

        // Display prediction
        sf::Image img(sf::Vector2u(rescaleRT.getSize().x, rescaleRT.getSize().y));

        // Load back into image
        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c;

                c.r = c.g = c.b = pred[y + x * rescaleRT.getSize().y];

                img.setPixel(sf::Vector2u(x, y), c);
            }

        // Load image into texture
        sf::Texture tex;

        tex.loadFromImage(img);

        // Display
        sf::Sprite s(genMode ? tex : rescaleRT.getTexture());

        s.setPosition(sf::Vector2f(window.getSize().x * 0.5f, window.getSize().y * 0.5f));

        s.setOrigin(sf::Vector2f(tex.getSize().x * 0.5f, tex.getSize().y * 0.5f));

        // Scale up to size of main window
        float scale = std::min(static_cast<float>(window.getSize().x) / img.getSize().x, static_cast<float>(window.getSize().y) / img.getSize().y);

        s.setScale(sf::Vector2f(scale, scale));

        window.draw(s);

        //Float3 pos = h.get_integrator(0).get_integrals()[0];

        //float pos_scale = 100.0f;

        //sf::CircleShape cs;
        //cs.setRadius(2.0f);

        //cs.setFillColor(sf::Color::Red);
        //cs.setPosition(sf::Vector2f(window.getSize().x * 0.5f + pos.x * pos_scale, window.getSize().y * 0.5f + pos.y * pos_scale));
        //window.draw(cs);

        window.display();
    } while (!quit);

    return 0;
}
