#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "runner/Runner.h"

#include <aogmaneo/hierarchy.h>

#include <time.h>
#include <iostream>
#include <random>

using namespace aon;

int main() {
    // RNG
    std::mt19937 rng(time(nullptr));

    // Create window
    sf::ContextSettings glContextSettings;

    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(800, 600)), "Runner Demo", sf::Style::Default, sf::State::Windowed, glContextSettings);

    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    // Physics
    b2WorldDef worldDef = b2DefaultWorldDef();

    worldDef.gravity = (b2Vec2){0.0f, -9.81f};

    b2WorldId world = b2CreateWorld(&worldDef);

    const float pixelsPerMeter = 256.0f;

    const float groundWidth = 20000.0f;
    const float groundHeight = 5.0f;

    const float hurdleWidth = 0.15f;
    const float hurdleHeight = 0.1f;
    const float hurdleHeightInc = 0.02f;
    const float hurdleOffset = 5.0f;
    const float hurdleStart = 10.0f;

    // Create ground body
    b2BodyDef groundBodyDef = b2DefaultBodyDef();
    groundBodyDef.type = b2_staticBody;
    groundBodyDef.position = (b2Vec2){0.0f, 0.0f};

    b2BodyId groundBody = b2CreateBody(world, &groundBodyDef);

    b2Polygon groundBox = b2MakeBox(groundWidth * 0.5f, groundHeight * 0.5f);

    b2ShapeDef groundShapeDef = b2DefaultShapeDef();
    //groundShape.material.friction = 1.0f;
    //groundShape.material.restitution = 0.0f;

    b2ShapeId groundShape = b2CreatePolygonShape(groundBody, &groundShapeDef, &groundBox);

    // Spawn some hurdles
    std::vector<b2BodyId> hurdles(100);

    for (int i = 0; i < hurdles.size(); i++) {
        b2BodyDef hurdleBodyDef = b2DefaultBodyDef();
        hurdleBodyDef.type = b2_staticBody;
        hurdleBodyDef.position = (b2Vec2){i * hurdleOffset + hurdleStart, groundHeight * 0.5f + (hurdleHeight + hurdleHeightInc * i) * 0.5f};

        b2BodyId hurdleBody = b2CreateBody(world, &hurdleBodyDef);

        b2Polygon hurdleBox = b2MakeBox(hurdleWidth * 0.5f, (hurdleHeight + hurdleHeightInc * i) * 0.5f);

        b2ShapeDef hurdleShapeDef = b2DefaultShapeDef();
        //groundShape.material.friction = 1.0f;
        //groundShape.material.restitution = 0.0f;

        b2ShapeId hurdleShape = b2CreatePolygonShape(hurdleBody, &hurdleShapeDef, &hurdleBox);

        hurdles[i] = hurdleBody;
    }

    // Background image
    sf::Texture skyTexture("resources/background1.png");

    skyTexture.setSmooth(true);

    // Floor image
    sf::Texture floorTexture("resources/floor1.png");

    floorTexture.setRepeated(true);
    floorTexture.setSmooth(true);

    // Create a runner
    const float runnerSpawnHeight = 2.762f;

    Runner runner;
    runner.createDefault(world, (b2Vec2){0.0f, runnerSpawnHeight}, 0.0f, 1);

    const int inputCount = 2 + 2 + 2 + 2 + 1 + 4 + 6 + 3 + 1; // 3 inputs for hind legs, 2 for front, body angle, contacts for each leg, 6 whiskers, IMU (lAccel, rAccel), distance to next hurdle
    const int outputCount = 2 + 2 + 2 + 2; // motor output for each joint

    // Create the agent
    set_num_threads(8);

    Array<Hierarchy::Layer_Desc> lds(2);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(5, 5, 32);
    }

    const int sensorResolution = 31;
    const int actionResolution = 11;

    Array<Hierarchy::IO_Desc> ioDescs(2);
    ioDescs[0] = Hierarchy::IO_Desc(Int3(4, 6, sensorResolution), IO_Type::none, 16, 6, 5);
    ioDescs[1] = Hierarchy::IO_Desc(Int3(2, 4, actionResolution), IO_Type::action, 16, 4, 5);

    Hierarchy h;
    h.init_random(ioDescs, lds);

    h.params.ios[1].importance = 0.0f;

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

    S32_Array actionCIs(outputCount, 0);

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_int_distribution<int> actionDist(0, actionResolution - 1);

    do {
        clock.restart();

        // ----------------------------- Input -----------------------------

        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                quit = true;
        }

        const float maxRunnerBodyAngle = 2.2f;

        std::vector<float> rescaledActions(outputCount, 0.5f);

        {
            if (window.hasFocus()) {
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape))
                    quit = true;

                kDownPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::K);

                if (!tDownPrev && sf::Keyboard::isKeyPressed(sf::Keyboard::Key::T))
                    speedMode = !speedMode;

                tDownPrev = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::T);
            }

            // Retrieve the sensor states
            std::vector<float> state;

            runner.getStateVector(state);

            S32_Array sensorCIs(h.get_io_size(0).x * h.get_io_size(0).y, 0);

            for (int i = 0; i < state.size(); i++)
                sensorCIs[i] = sigmoidf(state[i] * 1.0f) * (sensorResolution - 1) + 0.5f;

            b2Vec2 runnerPos = b2Body_GetPosition(runner.body);

            int nextHurdleIndex = 0;

            for (; nextHurdleIndex < hurdles.size(); nextHurdleIndex++) {
                b2Vec2 p = b2Body_GetPosition(hurdles[nextHurdleIndex]);

                if (p.x > runnerPos.x) {
                    break;
                }
            }

            if (nextHurdleIndex == hurdles.size())
                sensorCIs[state.size()] = sensorResolution - 1;
            else {
                b2Vec2 p = b2Body_GetPosition(hurdles[nextHurdleIndex]);
                float dist = p.x - runnerPos.x;

                sensorCIs[state.size()] = min(1.0f, 0.5f * dist / hurdleOffset) * (sensorResolution - 1) + 0.5f;
            }

            Array<S32_Array_View> inputCIs(2);
            inputCIs[0] = sensorCIs;
            inputCIs[1] = actionCIs;

            float vel = b2Body_GetLinearVelocity(runner.body).x;

            float accel = (vel - velPrev) / max(0.0001f, dt);

            velPrev = vel;

            std::normal_distribution<float> noiseDist(0.0f, 0.1f);

            float reward = vel * 1.0f;

            if (reset)
                reward -= 100.0f;

            h.step(inputCIs, true, reward * 1.0f);

            actionCIs = h.get_prediction_cis(1);

            for (int i = 0; i < actionCIs.size(); i++) {
                if (dist01(rng) < 0.0f)
                    actionCIs[i] = actionDist(rng);
            }

            // Go through tiles
            for (int i = 0; i < rescaledActions.size(); i++)
                rescaledActions[i] = actionCIs[i] / static_cast<float>(actionResolution - 1);
        }

        // Step the physics simulation
        int subSteps = 1;

        for (int ss = 0; ss < subSteps; ss++) {
            runner.motorUpdate(rescaledActions);

            b2World_Step(world, 1.0f / 60.0f / subSteps, 16);
        }

        averageVel = 0.99f * averageVel + 0.01f * b2Body_GetLinearVelocity(runner.body).x;

        if (std::abs(averageVel) < 0.1f)
            stuckTimer += dt;
        else
            stuckTimer = std::max(0.0f, stuckTimer - dt * 2.0f);

        reset = false;

        if (std::abs(b2Rot_GetAngle(b2Body_GetRotation(runner.body))) > maxRunnerBodyAngle) {
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
            runner.createDefault(world, (b2Vec2){0.0f, runnerSpawnHeight}, 0.0f, 1);
            velPrev = 0.0f;
        }

        // Display every 200 timesteps, or if not in speed mode
        if (!speedMode || steps % 100 == 1) {
            // -------------------------------------------------------------------

            // Center view on the runner
            b2Vec2 runnerPos = b2Body_GetPosition(runner.body);

            view.setCenter(sf::Vector2f(runnerPos.x * pixelsPerMeter, -runnerPos.y * pixelsPerMeter));

            // Draw sky
            sf::Sprite skySprite(skyTexture);

            // Sky doesn't move
            window.setView(window.getDefaultView());

            window.draw(skySprite);

            window.setView(view);

            // Draw floor
            sf::RectangleShape floorShape;
            floorShape.setSize(sf::Vector2f(groundWidth * pixelsPerMeter, groundHeight * pixelsPerMeter));
            floorShape.setTexture(&floorTexture);
            floorShape.setTextureRect(sf::IntRect(sf::Vector2i(0, 0), sf::Vector2i(groundWidth * pixelsPerMeter, groundHeight * pixelsPerMeter)));

            floorShape.setOrigin(sf::Vector2f(groundWidth * pixelsPerMeter * 0.5f, groundHeight * pixelsPerMeter * 0.5f));

            window.draw(floorShape);

            // Draw hurdles
            for (int i = 0; i < hurdles.size(); i++) {
                sf::RectangleShape rs;
                rs.setSize(sf::Vector2f(hurdleWidth * pixelsPerMeter, (hurdleHeight + hurdleHeightInc * i) * pixelsPerMeter));
                rs.setTexture(&floorTexture);
                rs.setPosition(sf::Vector2f((i * hurdleOffset + hurdleStart) * pixelsPerMeter, -(groundHeight * 0.5f + (hurdleHeight + hurdleHeightInc * i) * 0.5f) * pixelsPerMeter));
                rs.setTextureRect(sf::IntRect(sf::Vector2i((i * hurdleOffset + hurdleStart) * pixelsPerMeter, (groundHeight * 0.5f + (hurdleHeight + hurdleHeightInc * i) * 0.5f) * pixelsPerMeter), sf::Vector2i(hurdleWidth * pixelsPerMeter, (hurdleHeight + hurdleHeightInc * i) * pixelsPerMeter)));

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
            std::cout << "Steps: " << steps << " Distance: " << b2Body_GetPosition(runner.body).x << std::endl;

        steps++;

    } while (!quit);

    runner.destroy();
    b2DestroyWorld(world);

    return 0;
}
