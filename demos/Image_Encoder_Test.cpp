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

using namespace aon;

int main() {
    std::mt19937 generator(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;
    window.setFramerateLimit(0); // No limit

    window.create(sf::VideoMode(800, 800), "Physics Test", sf::Style::Default);

    // Render target for scene
    sf::RenderTexture rescaleRT;
    rescaleRT.create(32, 32);

    // --------------------------- Create the Hierarchy ---------------------------

    set_num_threads(8);

    Image_Encoder enc;
    enc.init(Int2(rescaleRT.getSize().x, rescaleRT.getSize().y));

    sf::Image bases_img;
    bases_img.create(enc.get_dct_size() * enc.get_dct_size(), enc.get_dct_size() * enc.get_dct_size());
    for (int bx = 0; bx < enc.get_dct_size(); bx++)
        for (int by = 0; by < enc.get_dct_size(); by++) {
            for (int px = 0; px < enc.get_dct_size(); px++)
                for (int py = 0; py < enc.get_dct_size(); py++) {
                    int index = py + enc.get_dct_size() * (px + enc.get_dct_size() * (by + enc.get_dct_size() * bx));

                    Byte v = enc.get_dct_bases()[index];

                    int x = bx * enc.get_dct_size() + px;
                    int y = by * enc.get_dct_size() + py;

                    bases_img.setPixel(x, y, sf::Color(v, v, v));
                }
        }

    sf::Texture bases_tex;
    bases_tex.loadFromImage(bases_img);
    bases_tex.setSmooth(true);

    // Create hierarchy
    Int3 hiddenSize = enc.get_encoding_size();

    Array<Hierarchy::Layer_Desc> lds(3);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(8, 8, 32);
        //lds[i].spatial_activity = 8;
    }

    Array<Hierarchy::IO_Desc> ioDescs(1);
    ioDescs[0].size = hiddenSize;
    ioDescs[0].type = prediction;

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

    Byte_Buffer imgb(rescaleRT.getSize().x * rescaleRT.getSize().y, 0.0f);

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

        // Load into input field
        for (int x = 0; x < rescaleRT.getSize().x; x++)
            for (int y = 0; y < rescaleRT.getSize().y; y++) {
                sf::Color c = rescaleImg.getPixel(x, y);

                float mono = 0.333f * (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f);

                imgb[y + x * rescaleRT.getSize().y] = mono * 255.0f;
            }

        // Feed first 5 frames from image, even when generating ("seed" sequence)
        if (simFrame > 5 && genMode) {
            Array<Int_Buffer_View> inputCIs(1);

            inputCIs[0] = h.get_prediction_cis(0);

            h.step(inputCIs, false);
        }
        else {
            enc.encode(imgb);

            Array<Int_Buffer_View> inputCIs(1);

            inputCIs[0] = enc.get_encoded_cis();

            h.step(inputCIs, true);
        }

        for (int i = 0; i < enc.get_encoded_cis().size(); i++)
            std::cout << enc.get_encoded_cis()[i] << " ";
        std::cout << std::endl;

        // Display
        sf::Sprite s;

        s.setPosition(window.getSize().x * 0.5f, window.getSize().y * 0.5f);

        // If in generation mode, show prediction, otherwise show the training data
        s.setTexture(rescaleRT.getTexture());

        s.setOrigin(sf::Vector2f(rescaleRT.getSize().x * 0.5f, rescaleRT.getSize().y * 0.5f));

        // Scale up to size of main window
        float scale = std::min(static_cast<float>(window.getSize().x) / rescaleRT.getSize().x, static_cast<float>(window.getSize().y) / rescaleRT.getSize().y);

        s.setScale(sf::Vector2f(scale, scale));

        window.draw(s);

        sf::Sprite bases_sprite;
        bases_sprite.setTexture(bases_tex);
        bases_sprite.setScale(8.0f, 8.0f);
        window.draw(bases_sprite);

        window.display();
    } while (!quit);

    return 0;
}
