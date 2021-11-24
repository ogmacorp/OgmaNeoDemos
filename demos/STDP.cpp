// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

class Network {
public:
    int numVisible, numHidden;
    std::vector<float> eWeights;
    std::vector<float> iWeights;
    std::vector<float> pWeights;
    std::vector<float> predictionActivations;
    std::vector<float> predictionStates;
    std::vector<float> visibleTraces;
    std::vector<float> visibleTracesPrev;
    std::vector<float> visibleTracesPrevPrev;
    std::vector<float> hiddenTraces;
    std::vector<float> hiddenTracesPrev;
    std::vector<float> hiddenTracesPrevPrev;
    std::vector<float> predictionTraces;
    std::vector<float> predictionTracesPrev;
    std::vector<float> predictionTracesPrevPrev;
    std::vector<float> hiddenActivations;
    std::vector<float> hiddenStates;
    std::vector<float> hiddenStatesPrev;
    std::vector<float> predictionThresholds;
    std::vector<float> hiddenThresholds;

    float leak;
    float iStrength;
    float traceDecay;
    float eLR, iLR, pLR;
    float targetRate;
    float tLR;
    float ptLR;

    Network()
    :
    leak(0.01f),
    iStrength(0.0f),
    traceDecay(0.98f),
    eLR(0.001f), iLR(0.001f), pLR(0.01f),
    targetRate(1.0f / 8.0f),
    tLR(0.001f),
    ptLR(0.01f)
    {}

    void init(
        int numVisible,
        int numHidden,
        std::mt19937 &rng
    ) {
        this->numVisible = numVisible;
        this->numHidden = numHidden;

        eWeights.resize(numHidden * numVisible);
        iWeights.resize(numHidden * numHidden);
        pWeights.resize(numHidden * numVisible);

        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        for (int i = 0; i < eWeights.size(); i++)
            eWeights[i] = dist01(rng);

        for (int i = 0; i < iWeights.size(); i++)
            iWeights[i] = dist01(rng);

        for (int i = 0; i < pWeights.size(); i++)
            pWeights[i] = dist01(rng);

        visibleTraces.resize(numVisible, 0.0f);
        visibleTracesPrev.resize(numVisible, 0.0f);
        visibleTracesPrevPrev.resize(numVisible, 0.0f);
        hiddenTraces.resize(numHidden, 0.0f);
        hiddenTracesPrev.resize(numHidden, 0.0f);
        hiddenTracesPrevPrev.resize(numHidden, 0.0f);
        predictionTraces.resize(numVisible, 0.0f);
        predictionTracesPrev.resize(numVisible, 0.0f);
        predictionTracesPrevPrev.resize(numVisible, 0.0f);

        predictionActivations.resize(numVisible, 0.0f);
        predictionStates.resize(numVisible, 0.0f);
        hiddenActivations.resize(numHidden, 0.0f);
        hiddenStates.resize(numHidden, 0.0f);

        hiddenStatesPrev.resize(numHidden, 0.0f);

        predictionThresholds.resize(numVisible, 1.0f);
        hiddenThresholds.resize(numHidden, 1.0f);
    }
    
    void step(
        const std::vector<float> &visibleStates
    ) {
        for (int i = 0; i < numVisible; i++) {
            float error = visibleStates[i] - predictionStates[i];

            for (int j = 0; j < numHidden; j++) {
                pWeights[j * numVisible + i] += pLR * error * hiddenStates[j];

                pWeights[j * numVisible + i] = std::min(1.0f, std::max(0.0f, pWeights[j * numVisible + i]));
            }

            predictionThresholds[i] = std::max(0.0001f, predictionThresholds[i] - ptLR * error);
        }

        for (int i = 0; i < numHidden; i++) {
            float stimulus = 0.0f;

            for (int j = 0; j < numVisible; j++)
                stimulus += eWeights[i * numVisible + j] * visibleStates[j];

            for (int j = 0; j < numHidden; j++) {
                // No self-connection
                if (j == i)
                    continue;

                stimulus -= iStrength * iWeights[i * numHidden + j] * hiddenStatesPrev[i];
            }

            hiddenActivations[i] = std::max(0.0f, hiddenActivations[i] + stimulus - leak * hiddenActivations[i]);
            //hiddenActivations[i] += stimulus - leak * hiddenActivations[i];

            if (hiddenActivations[i] > hiddenThresholds[i]) {
                hiddenStates[i] = 1.0f;

                hiddenActivations[i] = 0.0f;
            }
            else
                hiddenStates[i] = 0.0f;
        }

        // Prediction
        for (int i = 0; i < numVisible; i++) {
            float stimulus = 0.0f;

            for (int j = 0; j < numHidden; j++)
                stimulus += pWeights[j * numVisible + i] * hiddenStates[j];

            if (stimulus > predictionThresholds[i])
                predictionStates[i] = 1.0f;
            else
                predictionStates[i] = 0.0f;
        }

        // Trace update
        for (int i = 0; i < numHidden; i++)
            hiddenTraces[i] = std::max(hiddenTraces[i] * traceDecay, hiddenStates[i]);

        for (int i = 0; i < numVisible; i++)
            visibleTraces[i] = std::max(visibleTraces[i] * traceDecay, visibleStates[i]);

        // STDP
        for (int i = 0; i < numHidden; i++) {
            for (int j = 0; j < numVisible; j++)
                eWeights[i * numVisible + j] += eLR * (hiddenStates[i] * visibleTraces[j] - hiddenTracesPrev[i] * visibleStates[j]);

            for (int j = 0; j < numHidden; j++)
                iWeights[i * numHidden + j] += iLR * (hiddenStates[i] * hiddenTracesPrev[j] - hiddenTracesPrev[i] * hiddenStatesPrev[j]);
        
            hiddenThresholds[i] = std::max(0.0001f, hiddenThresholds[i] + tLR * (hiddenStates[i] - targetRate));
        }

        // Clamp weights
        for (int i = 0; i < eWeights.size(); i++)
            eWeights[i] = std::min(1.0f, std::max(0.0f, eWeights[i]));

        for (int i = 0; i < iWeights.size(); i++)
            iWeights[i] = std::min(1.0f, std::max(0.0f, iWeights[i]));

        // Normalize E weights
        for (int i = 0; i < numHidden; i++) {
            float total = 0.0f;

            for (int j = 0; j < numVisible; j++)
                total += eWeights[i * numVisible + j];

            float scale = 1.0f / std::max(0.0001f, total);

            for (int j = 0; j < numVisible; j++)
                eWeights[i * numVisible + j] *= scale;
        }

        // Normalize I weights
        for (int i = 0; i < numHidden; i++) {
            float total = 0.0f;

            for (int j = 0; j < numHidden; j++)
                total += iWeights[i * numHidden + j];

            float scale = 1.0f / std::max(0.0001f, total);

            for (int j = 0; j < numHidden; j++)
                iWeights[i * numHidden + j] *= scale;
        }

        // Buffer
        hiddenStatesPrev = hiddenStates;

        visibleTracesPrevPrev = visibleTracesPrev;
        visibleTracesPrev = visibleTraces;

        hiddenTracesPrevPrev = hiddenTracesPrev;
        hiddenTracesPrev = hiddenTraces;

        predictionTracesPrevPrev = predictionTracesPrev;
        predictionTracesPrev = predictionTraces;
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
    Network n;

    n.init(8, 16, rng);
    std::vector<float> visibleStates(n.numVisible, 0.0f);

    bool quit = false;
    int t = 0;
    int steps = 0;

    std::vector<float> randHeights(n.numHidden);

    std::normal_distribution<float> nDist(0.0f, 1.0f);

    for (int i = 0; i < n.numHidden; i++)
        randHeights[i] = nDist(rng) * 40.0f;

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
            iters = 1000;

        for (int it = 0; it < iters; it++) {
            if (steps >= 50) {
                steps = 0;

                for (int i = 0; i < visibleStates.size(); i++) {
                    if (t % visibleStates.size() == i || (t * 2) % visibleStates.size() == i)
                        visibleStates[i] = 1.0f;
                    else
                        visibleStates[i] = 0.0f;
                }

                t++;
            }
            else
                std::fill(visibleStates.begin(), visibleStates.end(), 0.0f);

            n.step(visibleStates);

            steps++;
        }

        window.clear();

        // Render
        sf::CircleShape cs;
        cs.setRadius(16.0f);
        cs.setOrigin(16.0f, 16.0f);
        cs.setOutlineThickness(2.0f);
        cs.setOutlineColor(sf::Color::White);


        const float hSpacing = 60.0f;
        const float vSpacing = 300.0f;
        const float predSpacing = 60.0f;
        const float hStart = windowWidth * 0.5f;
        const float vStart = windowHeight * 0.5f - 100.0f;

        sf::RectangleShape rs;

        for (int i = 0; i < n.numHidden; i++) {
            sf::Vector2f hPos(hStart + (-n.numHidden * 0.5f + i) * hSpacing, vStart + randHeights[i]);

            for (int j = 0; j < n.numVisible; j++) {
                sf::Vector2f vPos(hStart + (-n.numVisible * 0.5f + j) * hSpacing, vStart + vSpacing);

                sf::Vector2f delta = hPos - vPos;

                float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

                rs.setSize(sf::Vector2f(dist, 2.0f));

                rs.setRotation(std::atan2(-delta.y, -delta.x) * 180.0f / 3.1415f);

                rs.setPosition(hPos);

                int g = 255.0f * n.eWeights[i * n.numVisible + j];

                rs.setFillColor(sf::Color(g, g, g, 255));

                window.draw(rs);
            }

            for (int j = 0; j < n.numHidden; j++) {
                sf::Vector2f vPos(hStart + (-n.numHidden * 0.5f + j) * hSpacing, vStart + randHeights[j]);

                sf::Vector2f delta = hPos - vPos;

                float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

                rs.setSize(sf::Vector2f(dist, 2.0f));

                rs.setRotation(std::atan2(-delta.y, -delta.x) * 180.0f / 3.1415f);

                rs.setPosition(hPos);

                int g = 255.0f * n.iWeights[i * n.numHidden + j];

                rs.setFillColor(sf::Color(g, g, g, 255));

                window.draw(rs);
            }
        }

        for (int i = 0; i < n.numHidden; i++) {
            sf::Vector2f hPos(hStart + (-n.numHidden * 0.5f + i) * hSpacing, vStart + randHeights[i]);

            cs.setPosition(hPos);
            
            if (n.hiddenStates[i])
                cs.setFillColor(sf::Color::Green);
            else {
                int r = 255.0f * std::max(0.0f, n.hiddenActivations[i] / std::max(0.001f, n.hiddenThresholds[i]));
                int b = 255.0f * std::min(1.0f, std::max(0.0f, -n.hiddenActivations[i] / std::max(0.001f, n.hiddenThresholds[i])));
                
                cs.setFillColor(sf::Color(r, 0, b, 255));
            }

            window.draw(cs);
        }

        for (int i = 0; i < n.numVisible; i++) {
            sf::Vector2f vPos(hStart + (-n.numVisible * 0.5f + i) * hSpacing, vStart + vSpacing);

            cs.setPosition(vPos);
            
            if (visibleStates[i])
                cs.setFillColor(sf::Color::Green);
            else
                cs.setFillColor(sf::Color::Black);

            window.draw(cs);
        }

        for (int i = 0; i < n.numVisible; i++) {
            sf::Vector2f vPos(hStart + (-n.numVisible * 0.5f + i) * hSpacing, vStart + vSpacing + predSpacing);

            cs.setPosition(vPos);
            
            if (n.predictionStates[i])
                cs.setFillColor(sf::Color::Green);
            else {
                int r = 255.0f * std::max(0.0f, n.predictionActivations[i] / std::max(0.001f, n.predictionThresholds[i]));
                int b = 255.0f * std::min(1.0f, std::max(0.0f, -n.predictionActivations[i] / std::max(0.001f, n.predictionThresholds[i])));
                
                cs.setFillColor(sf::Color(r, 0, b, 255));
            }

            window.draw(cs);
        }
        
        window.display();
    } while (!quit);

    return 0;
}
