#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "evo/Lattice.h"

#include <cmath>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>

int selectIndex(const std::vector<float> &values, float greed, std::mt19937 &rng) {
    float minimum = values[0];
    float maximum = values[0];

    for (int i = 1; i < values.size(); i++) {
        minimum = std::min(minimum, values[i]);
        maximum = std::max(maximum, values[i]);
    }

    float scale = 1.0f / std::max(0.0001f, maximum - minimum);

    std::vector<float> rescaledValues(values.size());

    float total = 0.0f;

    for (int i = 0; i < values.size(); i++) {
        rescaledValues[i] = std::pow((values[i] - minimum) * scale, greed);
        total += rescaledValues[i];
    }

    std::uniform_real_distribution<float> totalDist(0.0f, total);

    float cusp = totalDist(rng);
    float sumSoFar = 0.0f;

    for (int i = 0; i < values.size(); i++) {
        sumSoFar += rescaledValues[i];

        if (sumSoFar >= cusp)
            return i;
    }

    return 0;
}

int main() {
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(1024, 1024), "Cat Mouse", sf::Style::Default);

    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    std::vector<Unit> pop(20);
    std::vector<float> fitnesses(pop.size(), 0.0f);

    for (int i = 0; i < pop.size(); i++)
        pop[i].initRandom(3, 8, 4, 6, rng);

    int individualIndex = 0;
    int generationIndex = 0;

    int stepsPerIndividual = 1000;
    int t = 0;

    int width = 4;
    int height = 4;

    std::unique_ptr<Lattice> lat = std::make_unique<Lattice>();
    lat->initRandom(pop[individualIndex], { width, height }, 4, 1, 1, 1, rng);

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    bool speedMode = false;
    int tPressedPrev = false;
    bool sPressedPrev = false;

    sf::Clock clock;

    float dt = 0.017f;

    do {
        clock.restart();

        int numSubSteps = speedMode ? 500 : 1;

        for (int ss = 0; ss < numSubSteps; ss++) {
            // ----------------------------- Input -----------------------------

            sf::Event windowEvent;

            while (window.pollEvent(windowEvent)) {
                switch (windowEvent.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                }
            }

            if (window.hasFocus()) {
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                    quit = true;

                bool tPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::T);

                if (tPressed && !tPressedPrev)
                    speedMode = !speedMode;

                tPressedPrev = tPressed;

                bool sPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::S);

                if (sPressed && !sPressedPrev) {
                    std::cout << "S" << std::endl;

                    std::cout << lat->getStates()[3][0] << std::endl;
                }

                sPressedPrev = sPressed;
            }

            int oneHotIndex = (std::sin(t * M_PI * 0.125f) * 0.5f + 0.5f) * (width * height - 1) + 0.5f;
            int oneHotIndexNext = (std::sin((t + 1) * M_PI * 0.125f) * 0.5f + 0.5f) * (width * height - 1) + 0.5f;

            std::vector<std::vector<float>> inputs(width * height, std::vector<float>{ 0.0f, 0.0f, 0.0f });

            inputs[oneHotIndex] = { 1.0f, 0.0f, 0.0f };

            lat->step(inputs, pop[individualIndex], 3);

            std::vector<std::vector<float>> preds = lat->getStates();

            int predIndex = 0;

            for (int i = 0; i < preds.size(); i++) {
                if (preds[i][1] > preds[predIndex][1])
                    predIndex = i;
            }

            if (predIndex == oneHotIndexNext && t >= stepsPerIndividual / 2) {
                fitnesses[individualIndex] += 1.0f;
            }

            t++;

            if (t >= stepsPerIndividual) {
                t = 0;

                std::cout << "Individual index " << individualIndex << " evaluated with fitness " << fitnesses[individualIndex] << std::endl;

                individualIndex++;

                if (individualIndex >= pop.size()) {
                    individualIndex = 0;

                    // Generation
                    std::vector<Unit> newPop(pop.size());

                    for (int i = 0; i < pop.size(); i++) {
                        float greed = 6.0f;

                        if (dist01(rng) < 0.5f) { // Change to just copy
                            int p1 = selectIndex(fitnesses, greed, rng);      

                            newPop[i] = pop[p1];
                        }
                        else {
                            int p1 = selectIndex(fitnesses, greed, rng);      
                            int p2 = selectIndex(fitnesses, greed, rng);      

                            newPop[i].initFromParents(pop[p1], pop[p2], rng);
                            newPop[i].mutate(0.1f, 0.1f, rng);
                        }
                    }

                    pop = newPop;
                    fitnesses = std::vector<float>(pop.size(), 0.0f);

                    std::cout << "Generation index " << generationIndex << " complete." << std::endl;
                    
                    generationIndex++;
                }

                lat = std::make_unique<Lattice>();
                lat->initRandom(pop[individualIndex], { width, height }, 4, 1, 1, 1, rng);
            }
        }

        window.clear();

        window.display();

    } while (!quit);

    return 0;
}
