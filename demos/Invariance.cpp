#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

struct SOM1D {
    int numInputs;
    int numColumns;
    int numUnits;
    std::vector<float> protos;
    std::vector<float> recon;
    std::vector<int> indices;
    std::vector<float> rates;

    float lr = 0.02f;
    float falloff = 0.01f;

    void init(
        int numInputs,
        int numColumns,
        int numUnits,
        float initRate,
        std::mt19937 &rng
    ) {
        this->numInputs = numInputs;
        this->numColumns = numColumns;
        this->numUnits = numUnits;

        int numProtos = numColumns * numUnits * numInputs;

        std::uniform_real_distribution<float> protoDist(0.0f, 1.0f);

        protos.resize(numProtos);

        for (int i = 0; i < protos.size(); i++)
            protos[i] = protoDist(rng);

        recon = std::vector<float>(numInputs, 0.0f);

        indices = std::vector<int>(numColumns, 0);

        rates = std::vector<float>(numColumns * numUnits, 0.5f);
    }

    void step(
        const std::vector<float> &inputs,
        bool learnEnabled = true
    ) {
        for (int c = 0; c < numColumns; c++) {
            int maxIndex = 0;
            float maxAct = -999999.0f;

            for (int i = 0; i < numUnits; i++) {
                float act = 0.0f;

                for (int j = 0; j < numInputs; j++)
                    act += inputs[j] * protos[j + numInputs * (i + numUnits * c)];

                if (act > maxAct) {
                    maxAct = act;
                    maxIndex = i;
                }
            }

            indices[c] = maxIndex;
        }

        if (learnEnabled) {
            std::fill(recon.begin(), recon.end(), 0.0f);

            for (int c = 0; c < numColumns; c++) {
                for (int j = 0; j < numInputs; j++)
                    recon[j] += protos[j + numInputs * (indices[c] + numUnits * c)];
            }

            for (int j = 0; j < numInputs; j++)
                recon[j] /= numColumns;

            for (int c = 0; c < numColumns; c++) {
                for (int i = 0; i < numUnits; i++) {
                    int delta = indices[c] - i;

                    //int minDelta = (delta + numUnits / 2) % numUnits - numUnits / 2;

                    float strength = rates[i + numUnits * c] * std::exp(-std::abs(delta) * falloff);

                    for (int j = 0; j < numInputs; j++) {
                        float err = inputs[j] - recon[j];

                        if (indices[c] == i)
                            protos[j + numInputs * (i + numUnits * c)] += strength * err;
                        else
                            protos[j + numInputs * (i + numUnits * c)] += std::max(0.0f, strength * err);
                    }

                    rates[i + numUnits * c] -= lr * strength;
                }
            }
        }
    }
};

int main() {
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::normal_distribution<float> nDist(0.0f, 1.0f);

    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;
    unsigned int frameWidth = 32;
    unsigned int frameHeight = 32;

    SOM1D som;
    som.init(frameWidth * frameHeight, 3, 256, 0.5f, rng);

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "STDP Demo", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    sf::RenderTexture rt;
    rt.create(frameWidth, frameHeight);

    sf::Vector2f ballPos(frameWidth * 0.5f, frameHeight * 0.5f);
    sf::Vector2f ballVel(-0.1235f, 0.4321f);

    sf::Image indexImg;
    indexImg.create(frameWidth, frameHeight);

    bool quit = false;
    int bmuX = 0;
    int bmuY = 0;

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
            iters = 100;

        for (int it = 0; it < iters; it++) {
            ballVel = 0.95f * ballVel + sf::Vector2f(nDist(rng), nDist(rng)) * 0.2f;
            ballPos += ballVel * 1.0f;

            if (ballPos.x < 0.0f) {
                ballPos.x = 0.0f;
                ballVel.x *= -1.0f;
            }
            else if (ballPos.x > frameWidth - 1) {
                ballPos.x = frameWidth - 1;
                ballVel.x *= -1.0f;
            }

            if (ballPos.y < 0.0f) {
                ballPos.y = 0.0f;
                ballVel.y *= -1.0f;
            }
            else if (ballPos.y > frameHeight - 1) {
                ballPos.y = frameHeight - 1;
                ballVel.y *= -1.0f;
            }

            sf::CircleShape cs;
            cs.setRadius(4.0f);
            cs.setOrigin(4.0f, 4.0f);
            cs.setPosition(ballPos);
            cs.setFillColor(sf::Color::White);

            rt.clear();
            rt.draw(cs);
            rt.display();

            sf::Image img = rt.getTexture().copyToImage();

            std::vector<float> inputs(frameWidth * frameHeight, 0.0f);

            for (int x = 0; x < frameWidth; x++)
                for (int y = 0; y < frameHeight; y++) {
                    sf::Color color = img.getPixel(x, y);

                    inputs[y + x * frameHeight] = color.r / 255.0f;
                }

            som.step(inputs, true);

            int px = ballPos.x;
            int py = ballPos.y;

            indexImg.setPixel(px, py, sf::Color(
                static_cast<unsigned char>(som.indices[0] / static_cast<float>(som.numUnits - 1) * 255.0f),
                static_cast<unsigned char>(som.indices[1] / static_cast<float>(som.numUnits - 1) * 255.0f),
                static_cast<unsigned char>(som.indices[2] / static_cast<float>(som.numUnits - 1) * 255.0f)));
        }

        window.clear();

        sf::Sprite s;

        s.setTexture(rt.getTexture());
        s.setScale(8.0f, 8.0f);

        window.draw(s);

        sf::Texture indexTex;
        indexTex.loadFromImage(indexImg);

        s.setTexture(indexTex);
        s.setPosition(frameWidth * 8.0f, 0.0f);

        window.draw(s);

        sf::CircleShape cs;
        cs.setRadius(2.0f);
        cs.setOrigin(2.0f, 2.0f);
        cs.setFillColor(sf::Color::White);
        cs.setPosition(ballPos * 8.0f + sf::Vector2f(frameWidth * 8.0f, 0.0f));

        window.draw(cs);

        cs.setPosition((som.indices[0] / static_cast<float>(som.numUnits - 1) + 1.0f) * frameWidth * 8.0f, frameHeight * 8.0f + 10.0f);

        window.draw(cs);
        
        cs.setPosition((som.indices[1] / static_cast<float>(som.numUnits - 1) + 1.0f) * frameWidth * 8.0f, frameHeight * 8.0f + 20.0f);

        window.draw(cs);

        cs.setPosition((som.indices[2] / static_cast<float>(som.numUnits - 1) + 1.0f) * frameWidth * 8.0f, frameHeight * 8.0f + 30.0f);

        window.draw(cs);

        window.display();
    } while (!quit);

    return 0;
}
