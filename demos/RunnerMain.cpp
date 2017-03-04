// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <runner/Runner.h>

#include <neo/Architect.h>
#include <neo/Hierarchy.h>

#include <time.h>
#include <iostream>
#include <random>

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

int main() {
    // RNG
    std::mt19937 generator(time(nullptr));

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
    const int actionTileWidth = 3;

    // Create the agent
    std::shared_ptr<ogmaneo::Resources> res = std::make_shared<ogmaneo::Resources>();

    // Use GPU
    res->create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::Architect arch;
    arch.initialize(1234, res);

    // State
    arch.addInputLayer(ogmaneo::Vec2i(4, 4));

    // Q
    arch.addInputLayer(ogmaneo::Vec2i(4 * actionTileWidth, 3 * actionTileWidth), true);

    const float decay = 0.1f;

    for (int l = 0; l < 6; l++) {
        if (l == 0)
            arch.addHigherLayer(ogmaneo::Vec2i(36, 36), ogmaneo::_distance);
        else
            arch.addHigherLayer(ogmaneo::Vec2i(36, 36), ogmaneo::_chunk);
    }

    // Generate
    std::shared_ptr<ogmaneo::Hierarchy> h = arch.generateHierarchy();

    // Create necessary fields
    ogmaneo::ValueField2D stateField(ogmaneo::Vec2i(4, 4), 0.0f);
    ogmaneo::ValueField2D qField(ogmaneo::Vec2i(4 * actionTileWidth, 3 * actionTileWidth), 0.0f);

    std::vector<float> valuePrevs(4 * 3, 0.0f);
    std::vector<float> maxPrevs(4 * 3, 0.0f);

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

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

    // Layer textures for debugging
    std::vector<sf::Texture> layerTextures(h->getPredictor().getHierarchy().getNumLayers());

    do {
        clock.restart();

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

        // Stabilization (keep runner from tipping over) parameters
        const float maxRunnerBodyAngle = 0.3f;
        const float runnerBodyAngleStab = 10.0f;

        {
            // Reward is velocity (flipped direction if K is pressed)
            float reward;

            if (!kDownPrev && sf::Keyboard::isKeyPressed(sf::Keyboard::K))
                runBackwards = !runBackwards;

            kDownPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::K);

            if (!tDownPrev && sf::Keyboard::isKeyPressed(sf::Keyboard::T))
                speedMode = !speedMode;

            tDownPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::T);

            // Retrieve the sensor states
            std::vector<float> state;

            runner0.getStateVector(state);

            for (int i = 0; i < state.size(); i++)
                stateField.getData()[i] = state[i];

            // Activate
            std::vector<ogmaneo::ValueField2D> inputVector{ stateField, qField };

            h->activate(inputVector);

            if (runBackwards)
                reward = -runner0._pBody->GetLinearVelocity().x;
            else
                reward = runner0._pBody->GetLinearVelocity().x;

            // Go through tiles
            std::vector<int> actionIndices(4 * 3);

            float averageTDError = 0.0f;

            for (int tx = 0; tx < 4; tx++)
                for (int ty = 0; ty < 3; ty++) {
                    float maxQ = -99999.0f;

                    int maxIndex = 0;
                    
                    for (int dx = 0; dx < actionTileWidth; dx++)
                        for (int dy = 0; dy < actionTileWidth; dy++) {
                            int x = tx * actionTileWidth + dx;
                            int y = ty * actionTileWidth + dy;

                            float q = h->getPredictions()[1].getValue(ogmaneo::Vec2i(x, y));

                            if (q > maxQ) {
                                maxQ = q;
                                maxIndex = dx + dy * actionTileWidth;
                            }

                            qField.setValue(ogmaneo::Vec2i(x, y), 0.0f);
                        }

                    float valuePrev = valuePrevs[tx + ty * 4];
                    float maxPrev = maxPrevs[tx + ty * 4];

                    int actionExplore = maxIndex;

                    if (dist01(generator) < 0.02f) {
                        std::uniform_int_distribution<int> actionDist(0, actionTileWidth * actionTileWidth - 1);

                        actionExplore = actionDist(generator);
                    }

                    actionIndices[tx + ty * 4] = actionExplore;

                    float nextQ;

                    {
                        int dx = actionExplore % 4;
                        int dy = actionExplore / 4;

                        int x = tx * actionTileWidth + dx;
                        int y = ty * actionTileWidth + dy;

                        nextQ = valuePrevs[tx + ty * 4] = h->getPredictions()[1].getValue(ogmaneo::Vec2i(x, y));

                        qField.setValue(ogmaneo::Vec2i(x, y), 1.0f);
                    }
                    
                    maxPrevs[tx + ty * 4] = maxQ;

                    float tdError = maxPrev + (reward + 0.98f * nextQ - maxPrev) * 1.0f - valuePrev;

                    averageTDError += tdError;
                }

            averageTDError /= 4 * 3;

            std::cout << averageTDError << std::endl;

            inputVector[1] = qField;

            h->learn(inputVector, averageTDError * 0.5f);

            std::vector<float> rescaledActions(4 * 3);

            for (int i = 0; i < rescaledActions.size(); i++) {
                rescaledActions[i] = actionIndices[i] / static_cast<float>(4 * 3 - 1);
            }

            // Update motors with actions
            runner0.motorUpdate(rescaledActions);

            // Keep upright (prevent from tipping over)
            if (std::abs(runner0._pBody->GetAngle()) > maxRunnerBodyAngle)
                runner0._pBody->SetAngularVelocity(-runnerBodyAngleStab * runner0._pBody->GetAngle());
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
            view.setCenter(runner0._pBody->GetPosition().x * pixelsPerMeter, -runner0._pBody->GetPosition().y * pixelsPerMeter);

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
            std::cout << "Steps: " << steps << " Distance: " << runner0._pBody->GetPosition().x << std::endl;

        steps++;

    } while (!quit);

    world->DestroyBody(groundBody);

    return 0;
}