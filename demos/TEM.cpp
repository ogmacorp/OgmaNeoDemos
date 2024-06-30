#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "constructs/Vec2f.h"
#include "constructs/Vec3f.h"
#include "constructs/Matrix3x3f.h"
#include "constructs/Quaternion.h"

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

typedef std::vector<int> CSDR;

CSDR Unorm8ToCSDR(float x) {
    x = std::min(1.0f, std::max(0.0f, x));

    int i = static_cast<int>(x * 255.0f + 0.5) & 0xff;

    CSDR res = { i & 0x0f, (i & 0xf0) >> 4 };

    return res;
};

float CSDRToUnorm8(const CSDR &csdr) {
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

const float rootThreeOverTwo = std::sqrt(3) * 0.5f;

float triNorm(const Vec2f &diff) {
    const std::array<Vec2f, 7> sTable = {
        Vec2f(0.0f, 0.0f),
        Vec2f(-0.5f, rootThreeOverTwo),
        Vec2f(-0.5f, -rootThreeOverTwo),
        Vec2f(0.5f, rootThreeOverTwo),
        Vec2f(0.5f, -rootThreeOverTwo),
        Vec2f(-1.0f, 0.0f),
        Vec2f(1.0f, 0.0f)
    };

    float norm = 999999.0f;

    for (int i = 0; i < 7; i++)
        norm = std::min(norm, (diff + sTable[i]).magnitude());

    return norm;
}

class GridCellModule {
public:
    int sensorSize;
    int width, height;

    std::vector<float> weights;

    std::vector<float> activityA;
    std::vector<float> activityB;
    std::vector<float> targets;

    float I;
    float T;
    float sigma;
    float tau;

    float lr;
    float drift;

    GridCellModule()
    :
    I(0.3f),
    T(0.05f),
    sigma(0.24f),
    tau(0.8f),
    lr(0.5f),
    drift(0.01f)
    {}

    void init(int sensorSize, int width, int height, std::mt19937 &rng) {
        this->sensorSize = sensorSize;
        this->width = width;
        this->height = height;

        activityA.resize(width * height);

        std::uniform_real_distribution<float> actDist(0.0f, 1.0f / std::sqrt(static_cast<float>(activityA.size())));

        for (int i = 0; i < activityA.size(); i++)
            activityA[i] = actDist(rng);

        activityB = activityA;

        weights.resize(activityA.size() * sensorSize);

        std::fill(weights.begin(), weights.end(), 0.0f);

        targets = activityA;
    }

    void step(const std::vector<float> &sensors, const Vec2f &translation) {
        float avgActivityB = 0.0f;

        for (int i = 0; i < activityB.size(); i++)
            avgActivityB += activityB[i];

        avgActivityB /= std::max(0.0001f, static_cast<float>(activityB.size()));

        std::vector<float> activityAPrev = activityA;

        for (int i = 0; i < activityA.size(); i++) {
            Vec2f posi(((i % width) - 0.5f) / width, rootThreeOverTwo * ((i / width) - 0.5f) / height);

            float activityBPrev = activityB[i];

            float target = 0.0f;

            // Stimulate
            for (int j = 0; j < sensorSize; j++) {
                int wi = j + i * sensorSize;

                target += weights[wi] * sensors[j];
            }

            target /= sensorSize;

            targets[i] = target;

            float sum = 0.0f;

            // Propagate
            for (int j = 0; j < activityA.size(); j++) {
                Vec2f posj(((j % width) - 0.5f) / width, rootThreeOverTwo * ((j / width) - 0.5f) / height);

                float norm = triNorm(posi - posj + translation);

                float weight = I * std::exp(-norm * norm / (sigma * sigma)) - T;

                sum += weight * activityAPrev[j];
            }

            activityB[i] = activityA[i] + sum;

            activityA[i] = std::max(0.0f, activityB[i] + tau * (activityB[i] / std::max(0.0001f, avgActivityB) - activityB[i]) + drift * target);

            // Update weights
            float delta = lr * (activityA[i] - target);

            for (int j = 0; j < sensorSize; j++) {
                int wi = j + i * sensorSize;

                weights[wi] += delta * sensors[j];
            }
        }
    }

    std::vector<float> predict() {
        std::vector<float> predictions(sensorSize, 0.0f);

        for (int j = 0; j < sensorSize; j++) {
            for (int i = 0; i < activityA.size(); i++)
                predictions[j] += activityA[i] * weights[j + i * sensorSize];
        }

        return predictions;
    }

    Vec2f getPosition() const {
        Vec2f pos(0.0f, 0.0f);
        float total = 0.0f;

        for (int i = 0; i < activityA.size(); i++) {
            Vec2f posi(((i % width) - 0.5f) / width, rootThreeOverTwo * ((i / width) - 0.5f) / height);

            pos += activityA[i] * posi;
            total += activityA[i];
        }

        return pos / std::max(0.0001f, total);
    }
};

class GridCellAssembly {
public:
    std::vector<GridCellModule> mods;
    std::vector<float> angles;

    float scaling;

    GridCellAssembly()
    :
    scaling(2.0f)
    {}

    void init(int sensorSize, int numMods, int width, int height, std::mt19937 &rng) {
        mods.resize(numMods);
        angles.resize(numMods);

        std::uniform_real_distribution<float> angleDist(-3.1415f, 3.1415f);

        for (int i = 0; i < numMods; i++) {
            mods[i].init(sensorSize, width, height, rng);
            
            angles[i] = angleDist(rng);
        }
    }

    void step(const std::vector<float> &sensors, const Vec2f &translation) {
        float scale = 1.0f;

        for (int i = 0; i < mods.size(); i++) {
            mods[i].step(sensors, (Matrix3x3f::rotateMatrix(angles[i]) * translation) * scale);

            scale /= scaling;
        }
    }

    std::vector<float> predict() {
        std::vector<float> predictions(mods[0].sensorSize, 0.0f);

        for (int i = 0; i < mods.size(); i++) {
            std::vector<float> preds = mods[i].predict();

            for (int j = 0; j < predictions.size(); j++)
                predictions[j] += preds[j];
        }

        return predictions;
    }

    Vec2f getPosition() const {
        Vec2f pos(0.0f, 0.0f);

        float scale = 1.0f;

        for (int i = 0; i < mods.size(); i++) {
            Vec2f p = mods[i].getPosition();

            pos += Matrix3x3f::rotateMatrix(-angles[i]) * p / scale;

            scale /= scaling;
        }

        return pos;
    }
};


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

    sf::Font font;
    font.loadFromFile("resources/Hack-Regular.ttf");

    sf::View view = window.getDefaultView();
    view.setCenter(0.0f, 0.0f);
    sf::View newView = view;

    GridCellAssembly m;
    m.init(5, 5, 10, 9, rng);

    Vec2f pos(0.0f, 0.0f);
    Vec2f lvel(0.0f, 0.0f);

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
            float speed = 4.0f;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
                moveX = -speed;
            else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
                moveX = speed;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
                moveY = speed;
            else if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
                moveY = -speed;

            // Move
            lvel += -10.0f * dt * lvel + Vec2f(moveX, moveY) * dt - 0.1f * dt * pos;

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

            std::vector<float> sensors(5, 0.0f);

            const float minDist = 0.1f;

            if ((Vec2f(-0.8f, -0.8f) - pos).magnitude() < minDist)
                sensors[1] = 1.0f;
            else if ((Vec2f(0.8f, -0.8f) - pos).magnitude() < minDist)
                sensors[2] = 1.0f;
            else if ((Vec2f(0.8f, 0.8f) - pos).magnitude() < minDist)
                sensors[3] = 1.0f;
            else if ((Vec2f(-0.8f, 0.8f) - pos).magnitude() < minDist)
                sensors[4] = 1.0f;
            else
                sensors[0] = 1.0f;

            std::normal_distribution<float> noiseDist(0.0f, 0.2f);

            Vec2f noisyLVel = lvel + Vec2f(noiseDist(rng), noiseDist(rng)) * 0.0f;

            m.step(sensors, noisyLVel * 8.0f * dt);

            std::vector<float> predictions = m.predict();

            int mi = 0;

            for (int i = 0; i < predictions.size(); i++)
                if (predictions[i] > predictions[mi])
                    mi = i;

            std::cout << "Predicting " << mi << " and position " << m.getPosition() << " Actual position: " << pos << std::endl;
        }

        view.setCenter(view.getCenter() + (newView.getCenter() - view.getCenter()) * viewInterpolateRate * dt);
        view.setSize(view.getSize() + (newView.getSize() - view.getSize()) * viewInterpolateRate * dt);

        window.setView(view);

        window.clear(sf::Color::Black);

        const float renderScale = 300.0f;

        sf::Text t;

        t.setFont(font);
        t.setCharacterSize(30);

        t.setPosition(sf::Vector2f(-0.8f, -0.8f) * renderScale);
        t.setString("A");
        window.draw(t);

        t.setPosition(sf::Vector2f(0.8f, -0.8f) * renderScale);
        t.setString("B");
        window.draw(t);

        t.setPosition(sf::Vector2f(0.8f, 0.8f) * renderScale);
        t.setString("C");
        window.draw(t);

        t.setPosition(sf::Vector2f(-0.8f, 0.8f) * renderScale);
        t.setString("D");
        window.draw(t);

        sf::CircleShape cs;
        cs.setRadius(0.05f * renderScale);
        cs.setFillColor(sf::Color::Red);
        cs.setOrigin(cs.getRadius() * 0.5f, cs.getRadius() * 0.5f);
        cs.setPosition(sf::Vector2f(pos.x * renderScale, pos.y * renderScale));
        
        window.draw(cs);
        
        for (int i = 0; i < m.mods.size(); i++) {
            sf::Image img;
            img.create(m.mods[i].width, m.mods[i].height);

            for (int x = 0; x < img.getSize().x; x++) {
                for (int y = 0; y < img.getSize().y; y++) {
                    sf::Color c = sf::Color::White;
                    c.r = 255.0f * (1.0f - std::min(1.0f, std::max(0.0f, 0.1f * m.mods[i].activityA[x + y * m.mods[i].width])));
                    c.g = 255.0f * (1.0f - std::min(1.0f, std::max(0.0f, 0.1f * m.mods[i].targets[x + y * m.mods[i].width])));

                    img.setPixel(x, y, c);
                }
            }

            sf::Texture tex;
            tex.loadFromImage(img);

            sf::Sprite s;
            s.setTexture(tex);
            s.setPosition(-1.6f * renderScale, -1.4f * renderScale + 0.51f * i * renderScale);
            s.setScale(renderScale * 0.5f / img.getSize().x, renderScale * 0.5f / img.getSize().y);

            window.draw(s);
        }

        window.display();
    } while (!quit);

    return 0;
}

