#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/Hierarchy.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

using namespace aon;

const float fourThirds = 4.0f / 3.0f;

inline float square(float x) {
    return x * x;
}

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
        static const float g = 9.81f;
        static const float mp = 1.0f;
        static const float mc = 2.0f;
        static const float l = 1.0f;
        static const float dt = 1.0f / 60.0f;
        static const float mult = 1.0f / (mp + mc);

        float cosTheta = std::cos(theta);
        float sinTheta = std::sin(theta);
        float dtheta2 = square(dtheta);
        float ddtheta = (g * sinTheta + cosTheta * ((-force - mp * l * dtheta2 * sinTheta) * mult)) / (l * (fourThirds - (mp * square(cosTheta)) * mult));
        float ddx = (force + mp * l * (dtheta2 * sinTheta - ddtheta * cosTheta)) * mult;

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

    int inRes = 6;
    int maxBound = std::pow(2, inRes);

    float reward = 0.0f;
    std::vector<bool> recurrents(inRes, false);

    std::uniform_real_distribution<float> thetaDist(-0.03f, 0.03f);
    float fallAngle = 0.4f;
    float fallDist = 2.0f;

    CartPoleEnv env;
    env.theta = thetaDist(rng);

    // Create the agent
    setNumThreads(4);

    Array<Hierarchy::LayerDesc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(4, 4, 32);
    }

    const int sensorResolution = maxBound;
    const int actionResolution = 3;

    Array<Hierarchy::IODesc> ioDescs(2);
    ioDescs[0] = Hierarchy::IODesc(Int3(2, 2, sensorResolution), IOType::none, 2, 2, 64);
    ioDescs[1] = Hierarchy::IODesc(Int3(1, 1, actionResolution), IOType::action, 2, 2, 64);

    Hierarchy h;
    h.initRandom(ioDescs, lds);

    IntBuffer sensorCIs(4, 0);
    IntBuffer actionCIs(1, 0);

    Array<const IntBuffer*> inputCIs(2);
    inputCIs[0] = &sensorCIs;
    inputCIs[1] = &actionCIs;

    sf::Texture tex;

    bool quit = false;

    int t = 0;
    int attempt = 0;

    std::cout << "Ready." << std::endl;

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
            std::vector<float> inputValuesf(4, 0.0f);

            inputValuesf[0] = env.x;
            inputValuesf[1] = env.theta;
            inputValuesf[2] = env.dx;
            inputValuesf[3] = env.dtheta;

            for (int i = 0; i < 4; i++) {
                int v = sigmoidf(inputValuesf[i] * 4.0f) * (maxBound - 1) + 0.5f;

                sensorCIs[i] = v;
            }

            h.step(inputCIs, true, reward);

            actionCIs[0] = h.getPredictionCIs(1)[0];

            if (dist01(rng) < 0.0f) {
                std::uniform_int_distribution<int> actionDist(0, 2);
                actionCIs[0] = actionDist(rng);
            }

            float force = (actionCIs[0] - 1.0f) * 3.0f;

            env.step(force);

            // Reset
            reward = 0.0f;

            if (std::abs(env.theta) > fallAngle || std::abs(env.x) > fallDist) {
                std::cout << attempt << ": Survived " << t << " steps." << std::endl;
                t = 0;
                reward = -10.0f;
                attempt++;

                env.x = 0.0f;
                env.theta = thetaDist(rng);
                env.dx = 0.0f;
                env.dtheta = 0.0f;
            }

            t++;
        }

        window.clear();

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
