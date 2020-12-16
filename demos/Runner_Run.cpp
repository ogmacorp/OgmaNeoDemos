// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <runner/Runner.h>

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/Helpers.h>

#include <time.h>
#include <iostream>
#include <random>

using namespace aon;

int main() {
    // RNG
    std::mt19937 rng(time(nullptr));

    // Create window
    sf::RenderWindow window;

    sf::ContextSettings glContextSettings;
    glContextSettings.antialiasingLevel = 4;

    window.create(sf::VideoMode(800, 600), "Runner Demo", sf::Style::Default, glContextSettings);

    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    // Physics
    std::shared_ptr<b2World> world = std::make_shared<b2World>(b2Vec2(0.0f, -9.81f));

    const float pixelsPerMeter = 256.0f;

    const float groundWidth = 5000.0f;
    const float groundHeight = 5.0f;

    // Create ground body
    b2BodyDef groundBodyDef;
    groundBodyDef.position.Set(0.0f, 0.0f);

    b2Body* groundBody = world->CreateBody(&groundBodyDef);

    b2PolygonShape groundBox;
    groundBox.SetAsBox(groundWidth * 0.5f, groundHeight * 0.5f);

    groundBody->CreateFixture(&groundBox, 0.0f); // 0 density (static)

    // Background image
    sf::Texture skyTexture;

    skyTexture.loadFromFile("resources/background1.png");

    skyTexture.setSmooth(true);

    // Floor image
    sf::Texture floorTexture;

    floorTexture.loadFromFile("resources/floor1.png");

    floorTexture.setRepeated(true);
    floorTexture.setSmooth(true);

    // Create a runner
    Runner runner0;

    runner0.createDefault(world, b2Vec2(0.0f, 2.762f), 0.0f, 1);

    const int inputCount = 3 + 3 + 2 + 2 + 1 + 4; // 3 inputs for hind legs, 2 for front, body angle, contacts for each leg
    const int outputCount = 3 + 3 + 2 + 2; // Motor output for each joint

    // Create the agent
    setNumThreads(8);

    Array<Hierarchy::LayerDesc> lds(4);

    for (int i = 0; i < lds.size(); i++)
        lds[i].hiddenSize = Int3(4, 4, 16);

    const int sensorResolution = 17;
    const int actionResolution = 9;

    Array<Hierarchy::IODesc> ioDescs(2);
    ioDescs[0] = Hierarchy::IODesc(Int3(3, 5, sensorResolution), IOType::none, 3, 2, 2, 64);
    ioDescs[1] = Hierarchy::IODesc(Int3(2, 5, actionResolution), IOType::action, 3, 2, 2, 64);

    Hierarchy h;
    h.initRandom(ioDescs, lds);

    // ---------------------------- Game Loop -----------------------------

    sf::View view = window.getDefaultView();

    bool quit = false;

    sf::Clock clock;

    float dt = 0.017f;

    int steps = 0;

    // Run past real-time
    bool speedMode = false;

    // Reverse rewarding direction
    bool runBackwards = false;

    // key buffers
    bool kDownPrev = false;
    bool tDownPrev = false;

    IntBuffer actionCIs(outputCount, 0);

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_int_distribution<int> actionDist(0, actionResolution - 1);

    do {
        clock.restart();

        // ----------------------------- Input -----------------------------

        sf::Event windowEvent;

        while (window.pollEvent(windowEvent)) {
            switch (windowEvent.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            quit = true;

        // Stabilization (keep runner from tipping over) parameters
        const float maxRunnerBodyAngle = 0.3f;
        const float runnerBodyAngleStab = 10.0f;

        {
            if (window.hasFocus()) {
                // Reward is velocity (flipped direction if K is pressed)
                if (!kDownPrev && sf::Keyboard::isKeyPressed(sf::Keyboard::K))
                    runBackwards = !runBackwards;

                kDownPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::K);

                if (!tDownPrev && sf::Keyboard::isKeyPressed(sf::Keyboard::T))
                    speedMode = !speedMode;

                tDownPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::T);
            }

            // Retrieve the sensor states
            std::vector<float> state;

            runner0.getStateVector(state);

            IntBuffer sensorCIs(inputCount);

            for (int i = 0; i < state.size(); i++)
                sensorCIs[i] = sigmoid(state[i] * 2.0f) * (sensorResolution - 1) + 0.5f;

            Array<const IntBuffer*> inputCIs(2);
            inputCIs[0] = &sensorCIs;
            inputCIs[1] = &actionCIs;

            float reward;

            if (runBackwards)
                reward = -runner0.body->GetLinearVelocity().x;
            else
                reward = runner0.body->GetLinearVelocity().x;

            reward *= 10.0f;

            h.step(inputCIs, true, reward);

            actionCIs = h.getPredictionCIs(1);

            for (int i = 0; i < actionCIs.size(); i++) {
                if (dist01(rng) < 0.3f)
                    actionCIs[i] = actionDist(rng);
            }

            // Go through tiles
            std::vector<float> rescaledActions(outputCount);

            for (int i = 0; i < rescaledActions.size(); i++)
                rescaledActions[i] = actionCIs[i] / static_cast<float>(actionResolution - 1);

            // Update motors with actions
            runner0.motorUpdate(rescaledActions);

            // Keep upright (prevent from tipping over)
            if (std::abs(runner0.body->GetAngle()) > maxRunnerBodyAngle)
                runner0.body->SetAngularVelocity(-runnerBodyAngleStab * runner0.body->GetAngle());
        }

        // Step the physics simulation
        int subSteps = 1;

        for (int ss = 0; ss < subSteps; ss++) {
            world->ClearForces();

            world->Step(1.0f / 60.0f / subSteps, 24, 24);
        }

        // Display every 200 timesteps, or if not in speed mode
        if (!speedMode || steps % 100 == 1) {
            // -------------------------------------------------------------------

            // Center view on the runner
            view.setCenter(runner0.body->GetPosition().x * pixelsPerMeter, -runner0.body->GetPosition().y * pixelsPerMeter);

            // Draw sky
            sf::Sprite skySprite;
            skySprite.setTexture(skyTexture);

            // Sky doesn't move
            window.setView(window.getDefaultView());

            window.draw(skySprite);

            window.setView(view);

            // Draw floor
            sf::RectangleShape floorShape;
            floorShape.setSize(sf::Vector2f(groundWidth * pixelsPerMeter, groundHeight * pixelsPerMeter));
            floorShape.setTexture(&floorTexture);
            floorShape.setTextureRect(sf::IntRect(0, 0, groundWidth * pixelsPerMeter, groundHeight * pixelsPerMeter));

            floorShape.setOrigin(sf::Vector2f(groundWidth * pixelsPerMeter * 0.5f, groundHeight * pixelsPerMeter * 0.5f));

            window.draw(floorShape);

            // Draw the runner
            runner0.renderDefault(window, sf::Color::Red, pixelsPerMeter);

            window.setView(window.getDefaultView());

            window.setView(view);

            window.display();
        }

        // Show distance traveled
        if (steps % 100 == 0)
            std::cout << "Steps: " << steps << " Distance: " << runner0.body->GetPosition().x << std::endl;

        steps++;

    } while (!quit);

    world->DestroyBody(groundBody);

    return 0;
}
