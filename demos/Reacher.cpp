// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/Helpers.h>

#include <box2d/box2d.h>

#include <time.h>
#include <iostream>
#include <random>

using namespace aon;

struct Limb {
    b2PolygonShape bodyShape;
    b2Body* body;
    b2RevoluteJoint* joint;
};

struct Reacher {
    // Base
    b2PolygonShape bodyShape;
    b2Body* body;

    std::vector<Limb> limbs;
};

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
    b2World world(b2Vec2(0.0f, 0.0f));

    const float pixelsPerMeter = 256.0f;

    Reacher r;

    r.limbs.resize(3);

    {
        b2BodyDef bodyDef;
        bodyDef.position.Set(0.0f, 0.0f);

        r.body = world.CreateBody(&bodyDef);

        r.bodyShape.SetAsBox(0.5f, 0.5f);

        r.body->CreateFixture(&r.bodyShape, 0.0f); // 0 density flag (static)
    }

    float length = 1.0f;
    float thickness = 0.1f;

    b2Body* prevBody = r.body;
    b2Vec2 prevAttachPoint = b2Vec2(0.0f, 0.0f);

    for (int i = 0; i < r.limbs.size(); i++) {
        b2BodyDef bodyDef;

        bodyDef.type = b2_dynamicBody;

        float offset = length * 0.5f - thickness * 0.5f;

        float angle = prevBody->GetAngle() + 0.0f;

        bodyDef.position = prevBody->GetWorldPoint(prevAttachPoint) + b2Vec2(std::cos(angle) * offset, std::sin(angle) * offset);
        bodyDef.angle = angle;
        bodyDef.allowSleep = false;

        r.limbs[i].body = world.CreateBody(&bodyDef);

        r.limbs[i].bodyShape = b2PolygonShape();
        r.limbs[i].bodyShape.SetAsBox(length * 0.5f, thickness * 0.5f);

        b2FixtureDef fixtureDef;

        fixtureDef.shape = &r.limbs[i].bodyShape;

        fixtureDef.density = 1.0f;
        fixtureDef.friction = 1.0f;
        fixtureDef.restitution = 0.001f;

        r.limbs[i].body->CreateFixture(&fixtureDef);

        b2RevoluteJointDef jointDef;

        jointDef.bodyA = prevBody;

        jointDef.bodyB = r.limbs[i].body;

        jointDef.referenceAngle = 0.0f;
        jointDef.localAnchorA = prevAttachPoint;
        jointDef.localAnchorB = b2Vec2(-offset, 0.0f);
        jointDef.collideConnected = false;
        jointDef.maxMotorTorque = 10.0f;

        jointDef.enableMotor = true;

        r.limbs[i].joint = static_cast<b2RevoluteJoint*>(world.CreateJoint(&jointDef));

        prevBody = r.limbs[i].body;
        prevAttachPoint = b2Vec2(offset, 0.0f);
    }

    // Create the agent
    setNumThreads(8);

    Array<Hierarchy::LayerDesc> lds(5);

    for (int i = 0; i < lds.size(); i++)
        lds[i].hiddenSize = Int3(4, 4, 16);

    const int motionRes = 13;

    Array<Hierarchy::IODesc> ioDescs(1);
    ioDescs[0] = Hierarchy::IODesc(Int3(1, r.limbs.size(), motionRes), IOType::prediction, 2, 2, 64);

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

    bool reset = false;
    float stuckTimer = 0.0f;
    const float stuckTime = 5.0f;
    float averageVel = 0.0f;
    float velPrev = 0.0f;

    IntBuffer actionCIs(r.limbs.size(), 0);

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

        const float maxRunnerBodyAngle = 2.0f;

        world.ClearForces();

        std::vector<float> rescaledActions(outputCount, 0.5f);

        {
            if (window.hasFocus()) {
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                    quit = true;

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

            runner.getStateVector(state);

            IntBuffer sensorCIs(inputCount, 0);

            for (int i = 0; i < state.size(); i++)
                sensorCIs[i] = sigmoid(state[i] * 2.0f) * (sensorResolution - 1) + 0.5f;

            int nextHurdleIndex = 0;

            for (; nextHurdleIndex < hurdles.size(); nextHurdleIndex++) {
                if (hurdles[nextHurdleIndex]->GetPosition().x > runner.body->GetPosition().x) {
                    break;
                }
            }

            if (nextHurdleIndex == hurdles.size())
                sensorCIs[state.size()] = sensorResolution - 1;
            else {
                float dist = hurdles[nextHurdleIndex]->GetPosition().x - runner.body->GetPosition().x;

                sensorCIs[state.size()] = min(1.0f, 0.5f * dist / hurdleOffset) * (sensorResolution - 1) + 0.5f;
            }

            Array<const IntBuffer*> inputCIs(2);
            inputCIs[0] = &sensorCIs;
            inputCIs[1] = &actionCIs;

            float vel = runner.body->GetLinearVelocity().x;

            float accel = (vel - velPrev) / max(0.0001f, dt);

            velPrev = vel;

            std::normal_distribution<float> noiseDist(0.0f, 0.1f);

            float reward = vel;// * (1.0f + noiseDist(rng));

            reward *= 1.0f;

            if (reset)
                reward -= 100.0f;

            h.step(inputCIs, true, reward);

            actionCIs = h.getPredictionCIs(1);

            //for (int i = 0; i < actionCIs.size(); i++) {
            //    if (dist01(rng) < 0.02f)
            //        actionCIs[i] = actionDist(rng);
            //}

            // Go through tiles
            for (int i = 0; i < rescaledActions.size(); i++) {
                rescaledActions[i] = actionCIs[i] / static_cast<float>(actionResolution - 1);
            }
        }

        // Step the physics simulation
        int subSteps = 3;

        for (int ss = 0; ss < subSteps; ss++) {
            world.ClearForces();
            runner.motorUpdate(rescaledActions);
            world.Step(1.0f / 60.0f / subSteps, 8, 8);
        }

        averageVel = 0.99f * averageVel + 0.01f * runner.body->GetLinearVelocity().x;

        if (std::abs(averageVel) < 0.1f)
            stuckTimer += dt;
        else
            stuckTimer = std::max(0.0f, stuckTimer - dt * 2.0f);

        reset = false;

        if (std::abs(runner.body->GetAngle()) > maxRunnerBodyAngle) {
            std::cout << "Reset due to flip." << std::endl;
            reset = true;
        }

        if (runner.infrontOfWall()) {
            std::cout << "Reset due to wall collision." << std::endl;
            reset = true;
        }

        if (stuckTimer >= stuckTime) {
            std::cout << "Reset due to stuck timer." << std::endl;
            reset = true;
        }

        // Keep upright (prevent from tipping over)
        if (reset) {
            stuckTimer = 0.0f;
            runner.createDefault(&world, b2Vec2(0.0f, runnerSpawnHeight), 0.0f, 1);
            velPrev = 0.0f;
        }

        // Display every 200 timesteps, or if not in speed mode
        if (!speedMode || steps % 100 == 1) {
            // -------------------------------------------------------------------

            // Center view on the runner
            view.setCenter(runner.body->GetPosition().x * pixelsPerMeter, -runner.body->GetPosition().y * pixelsPerMeter);

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

            // Draw hurdles
            for (int i = 0; i < hurdles.size(); i++) {
                sf::RectangleShape rs;
                rs.setSize(sf::Vector2f(hurdleWidth * pixelsPerMeter, (hurdleHeight + hurdleHeightInc * i) * pixelsPerMeter));
                rs.setTexture(&floorTexture);
                rs.setPosition(sf::Vector2f((i * hurdleOffset + hurdleStart) * pixelsPerMeter, -(groundHeight * 0.5f + (hurdleHeight + hurdleHeightInc * i) * 0.5f) * pixelsPerMeter));
                rs.setTextureRect(sf::IntRect((i * hurdleOffset + hurdleStart) * pixelsPerMeter, (groundHeight * 0.5f + (hurdleHeight + hurdleHeightInc * i) * 0.5f) * pixelsPerMeter, hurdleWidth * pixelsPerMeter, (hurdleHeight + hurdleHeightInc * i) * pixelsPerMeter));

                rs.setOrigin(sf::Vector2f(hurdleWidth * pixelsPerMeter * 0.5f, (hurdleHeight + hurdleHeightInc * i) * pixelsPerMeter * 0.5f));

                window.draw(rs);
            }

            // Draw the runner
            runner.renderDefault(window, sf::Color::Red, pixelsPerMeter);

            window.setView(window.getDefaultView());

            window.setView(view);

            window.display();
        }

        // Show distance traveled
        if (steps % 100 == 0)
            std::cout << "Steps: " << steps << " Distance: " << runner.body->GetPosition().x << std::endl;

        steps++;

    } while (!quit);

    return 0;
}
