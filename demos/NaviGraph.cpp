#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "constructs/Quaternion.h"
#include "navigraph/Graph.h"

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

CSDR Unorm8ToCSDR(float x) {
    x = std::min(1.0f, std::max(0.0f, x));

    int i = static_cast<int>(x * 255.0f + 0.5) & 0xff;

    CSDR res = { i & 0x0f, (i & 0xf0) >> 4 };

    return res;
};

float CSDRToUnorm8(const std::vector<int> &csdr) {
    return (csdr[0] | (csdr[1] << 4)) / 255.0f;
};

CSDR transToCSDR(const Matrix4x4f &trans, float range) {
    CSDR res(32);

    for (int i = 0; i < 16; i++) {
        float v = trans.elements[i] / range * 0.5f + 0.5f;

        CSDR sub = Unorm8ToCSDR(v);

        res[i * 2 + 0] = sub[0];
        res[i * 2 + 1] = sub[1];
    }

    return res;
}

int main() {
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::normal_distribution<float> nDist(0.0f, 1.0f);

    const float dt = 0.017f;
    const float zoomRate = 0.4f;
    const float viewInterpolateRate = 20.0f;

    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Gen Demo", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    sf::View view = window.getDefaultView();
    view.setCenter(0.0f, 0.0f);
    sf::View newView = view;

    Matrix4x4f trans;
    trans.setIdentity();

    Vec3f pos(0.0f, 0.0f, 0.0f);
    Vec3f rot(0.0f, 0.0f, 0.0f);
    Vec3f lvel(0.0f, 0.0f, 0.0f);
    Vec3f rvel(0.0f, 0.0f, 0.0f);

    Graph g;
    g.init(16);

    bool quit = false;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            if (window.hasFocus()) {
                switch (event.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                case sf::Event::MouseWheelMoved:
                    int dWheel = event.mouseWheel.delta;

                    newView = view;

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta0 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    newView.setSize(view.getSize() + newView.getSize() * (-zoomRate * dWheel));

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta1 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    window.setView(view);

                    newView.setCenter(view.getCenter() + mouseZoomDelta0 - mouseZoomDelta1);

                    break;
                }
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            float moveX = 0.0f;
            float moveY = 0.0f;
            float speed = 1.0f;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
                moveX = -speed;
            else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
                moveX = speed;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
                moveY = speed;
            else if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
                moveY = -speed;

            if (true) {
                // Move
                lvel += -10.0f * dt * lvel + Vec3f(moveX, moveY, 0.0f) * dt - 0.1f * dt * pos;
                rvel += -10.0f * dt * rvel + Vec3f(0.0f, 0.0f, nDist(rng) * 0.3f) * dt;

                pos += lvel * dt;
                //rot += rvel * dt;

                if (pos.x > 1.0f)
                    pos.x = 1.0f;
                else if (pos.x < -1.0f)
                    pos.x = -1.0f;

                if (pos.y > 1.0f)
                    pos.y = 1.0f;
                else if (pos.y < -1.0f)
                    pos.y = -1.0f;

                Matrix4x4f inv;

                trans.inverse(inv);

                trans = Matrix4x4f::translateMatrix(pos) * Matrix4x4f::rotateMatrix(rot);

                float driftNoise = 0.0f;
                Matrix4x4f delta = inv * trans * Matrix4x4f::translateMatrix(Vec3f(nDist(rng) * driftNoise, nDist(rng) * driftNoise, 0.0f));

                // Add node
                g.step(delta, transToCSDR(trans, 1.0f));

                std::cout << pos << std::endl;
            }
        }

        view.setCenter(view.getCenter() + (newView.getCenter() - view.getCenter()) * viewInterpolateRate * dt);
        view.setSize(view.getSize() + (newView.getSize() - view.getSize()) * viewInterpolateRate * dt);

        window.setView(view);

        window.clear(sf::Color::Black);

        const float renderScale = 400.0f;

        g.renderXY(window, renderScale);

        {
            sf::CircleShape cs;
            Vec3f transPos = trans * Vec3f(0.0f, 0.0f, 0.0f);
            cs.setPosition(transPos.x * renderScale, transPos.y * renderScale);
            cs.setRadius(4.0f);
            cs.setOrigin(4.0f, 4.0f);
            cs.setFillColor(sf::Color::Yellow);
            window.draw(cs);
        }

        {
            sf::CircleShape cs;
            Vec3f estPos = g.estimated * Vec3f(0.0f, 0.0f, 0.0f);
            cs.setPosition(estPos.x * renderScale, estPos.y * renderScale);
            cs.setRadius(4.0f);
            cs.setOrigin(4.0f, 4.0f);
            cs.setFillColor(sf::Color::Cyan);
            window.draw(cs);
        }

        window.display();
    } while (!quit);

    return 0;
}

