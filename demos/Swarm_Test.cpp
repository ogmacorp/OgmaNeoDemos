#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "swarm/Swarm.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

struct CartPoleEnv {
    float x;
    float dx;
    float theta;
    float dtheta;

    CartPoleEnv()
    :
    x(0.0f),
    dx(0.0f),
    theta(0.0f),
    dtheta(0.0f)
    {}

    void step(float force) {
        const float g = 9.81f;
        const float mp = 1.0f;
        const float mc = 2.0f;
        const float l = 1.0f;
        const float dt = 1.0f / 60.0f;

        float ddtheta = (g * std::sin(theta) + std::cos(theta) * ((-force - mp * l * dtheta * dtheta * std::sin(theta)) / (mp + mc))) / (l * (4.0f / 3.0f - (mp * std::pow(std::cos(theta), 2)) / (mp + mc)));
        float ddx = (force + mp * l * (dtheta * dtheta * std::sin(theta) - ddtheta * std::cos(theta))) / (mp + mc);

        dx += ddx * dt;
        dtheta += ddtheta * dt;

        x += dx * dt;
        theta += dtheta * dt;
    }
};

int main() {
    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "STDP Demo", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    int inValues = 5;
    int inRes = 5;
    int maxBound = std::pow(2, inRes);

    float reward = 0.0f;
    std::vector<bool> recurrents(inRes, false);

    std::uniform_real_distribution<float> thetaDist(-0.03f, 0.03f);
    float fallAngle = 0.4f;
    float fallDist = 2.0f;

    CartPoleEnv env;
    env.theta = thetaDist(rng);

    Swarm swarm;
    swarm.init(inValues * inRes, 12, rng);

    sf::Image img;
    img.create(swarm.getWidth(), swarm.getHeight(), sf::Color::Black);

    sf::Texture tex;

    bool quit = false;

    int t = 0;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;
        }

        int iters = 1;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
            iters = 10000;

        for (int it = 0; it < iters; it++) {
            std::vector<bool> inputs(inValues * inRes, false);

            std::vector<float> inputValuesf(4, 0.0f);

            inputValuesf[0] = env.x;
            inputValuesf[1] = env.theta;
            inputValuesf[2] = env.dx;
            inputValuesf[3] = env.dtheta;

            for (int i = 0; i < 4; i++) {
                int v = sigmoid(inputValuesf[i] * 2.0f) * (maxBound - 1) + 0.5f;

                for (int j = 0; j < inRes; j++)
                    inputs[i * inRes + j] = (v >> j) & 0x1;
            }

            // Recurrent
            for (int i = 0; i < inRes; i++) {
                inputs[4 * inRes + i] = recurrents[i];
            }

            swarm.step(inputs, reward, rng, iters != 1);

            int action = 0;

            for (int i = 0; i < 2; i++) {
                if (swarm.getLastOn(inRes + i))
                    action = action | (1 << i);
            }

            for (int i = 0; i < recurrents.size(); i++) {
                recurrents[i] = swarm.getLastOn(inRes * 2 + i);
            }

            if (action == 3)
                action = 1;

            float force = (action - 1.0f) * 3.0f;

            env.step(force);

            // Reset
            reward = 0.0f;

            if (std::abs(env.theta) > fallAngle || std::abs(env.x) > fallDist) {
                std::cout << "Survived " << t << " steps." << std::endl;
                t = 0;
                reward = -1.0f;

                env.x = 0.0f;
                env.theta = thetaDist(rng);
                env.dx = 0.0f;
                env.dtheta = 0.0f;
            }

            t++;
        }

        window.clear();

        for (int x = 0; x < swarm.getWidth(); x++)
            for (int y = 0; y < swarm.getHeight(); y++)
                img.setPixel(x, y, swarm.getOn(x, y) ? sf::Color::White : sf::Color::Black);

        tex.loadFromImage(img);

        sf::Sprite s;

        s.setTexture(tex);
        s.setScale(16.0f, 16.0f);

        window.draw(s);

        const float metersToPixel = 128.0f;

        {
            sf::RectangleShape rs;
            rs.setSize(sf::Vector2f(0.05f, 1.0f) * metersToPixel);
            rs.setOrigin(sf::Vector2f(0.025f, 1.0f) * metersToPixel);

            rs.setRotation(180.0f / M_PI * env.theta);
            rs.setPosition(sf::Vector2f(env.x + 3.0f, 3.0f) * metersToPixel);

            window.draw(rs);
        }

        {
            sf::RectangleShape rs;
            rs.setSize(sf::Vector2f(0.4f, 0.2f) * metersToPixel);
            rs.setOrigin(sf::Vector2f(0.2f, 0.1f) * metersToPixel);

            rs.setPosition(sf::Vector2f(env.x + 3.0f, 3.0f) * metersToPixel);

            window.draw(rs);
        }
        
        window.display();
    } while (!quit);

    return 0;
}
