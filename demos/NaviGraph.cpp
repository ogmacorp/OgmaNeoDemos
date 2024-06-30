#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "constructs/Vec2f.h"
#include "navigraph/GridCells.h"

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

int main() {
    std::mt19937 rng(time(nullptr));

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

    GridCells gc;
    gc.init_random(4, 64, 64, 8, rng);

    Vec2f pos(0.0f, 0.0f);
    std::vector<float> features(4, 0.0f);

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
            float moveAngle = 0.0f;
            float speed = 1.0f;
            float angleSpeed = 0.5f;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
                moveX = -speed;
            else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
                moveX = speed;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
                moveY = speed;
            else if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
                moveY = -speed;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
                moveAngle = angleSpeed;
            else if (sf::Keyboard::isKeyPressed(sf::Keyboard::E))
                moveAngle = -angleSpeed;

            pos += Vec2f(moveX, moveY);

            Vec2f fpos = pos * 0.1f;

            // function of pos
            features[0] = std::sin(fpos.x * 0.5f - fpos.y * 0.3f + 0.5f);
            features[1] = std::sin(-fpos.x * 0.1f - fpos.y * 2.3f - 0.5f);
            features[2] = std::sin(fpos.x * 0.1f - 2.5f);
            features[3] = std::sin(-fpos.y * 2.1f + 0.9f);

            gc.step(features, moveX, moveY);
        }

        view.setCenter(view.getCenter() + (newView.getCenter() - view.getCenter()) * viewInterpolateRate * dt);
        view.setSize(view.getSize() + (newView.getSize() - view.getSize()) * viewInterpolateRate * dt);

        window.setView(view);

        window.clear(sf::Color::Black);

        const float render_scale = 4.0f;

        sf::Texture tex;
        tex.loadFromImage(gc.get_states_image());

        sf::Sprite s;
        s.setTexture(tex);
        s.setScale(sf::Vector2f(render_scale, render_scale));

        window.draw(s);

        window.display();
    } while (!quit);

    return 0;
}

