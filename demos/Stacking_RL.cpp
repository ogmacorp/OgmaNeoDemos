#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

#include <aogmaneo/hierarchy.h>

using namespace aon;

const int width = 3;
const int height = 3;
const int numActions = 4;
const int numBlocks = 3;

int main() {
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::normal_distribution<float> nDist(0.0f, 1.0f);
    std::uniform_int_distribution<int> actionDist(0, numActions - 1);
    std::uniform_int_distribution<int> stackDist(0, width - 1);

    std::vector<int> actualStacks(width, 0);
    std::vector<int> targetStacks(width, 0);

    for (int i = 0; i < numBlocks; i++) {
        actualStacks[stackDist(rng)]++;
        targetStacks[stackDist(rng)]++;
    }

    std::vector<int> buildStacks = targetStacks;

    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Stacking", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    sf::Font font;
    font.loadFromFile("resources/Hack-Regular.ttf");

    bool quit = false;

    bool spacePrev = false;
    bool tPrev = false;

    bool wPrev = false;
    bool aPrev = false;
    bool sPrev = false;
    bool dPrev = false;

    bool speedMode = false;
    bool saveState = false;

    bool actualHolding = false;
    bool buildHolding = false;
    int actualPosition = 0;
    int buildPosition = 0;

    // --------------------------- Create the Hierarchy ---------------------------

    set_num_threads(8);

    Array<Hierarchy::Layer_Desc> lds(2);

    for (int i = 0; i < lds.size(); i++)
        lds[i].hidden_size = Int3(5, 5, 32);

    Array<Hierarchy::IO_Desc> ioDescs(4);
    ioDescs[0] = Hierarchy::IO_Desc(Int3(width, height, 2), IO_Type::none); // Goal
    ioDescs[1] = Hierarchy::IO_Desc(Int3(width, height, 2), IO_Type::prediction); // Actual
    ioDescs[2] = Hierarchy::IO_Desc(Int3(1, 1, numActions), IO_Type::action); // Action
    ioDescs[3] = Hierarchy::IO_Desc(Int3(1, 1, width), IO_Type::none); // Grabber position

    Hierarchy h;
    h.init_random(ioDescs, lds);

    h.params.ios[2].actor.discount = 0.9f;

    int actIndex = 0;

    int slowTimer = 0;
    int slowTime = 30;

    float randomizeChance = 0.1f;

    // ---------------------------------------------------------------------
    
    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            if (window.hasFocus()) {
                switch (event.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                }
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            bool space = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);
            bool t = sf::Keyboard::isKeyPressed(sf::Keyboard::T);

            if (space && !spacePrev) {
                targetStacks = buildStacks;

                std::cout << "State saved." << std::endl;
            }

            if (speedMode && dist01(rng) < randomizeChance) {
                std::fill(targetStacks.begin(), targetStacks.end(), 0.0f);

                for (int i = 0; i < numBlocks; i++) {
                    targetStacks[stackDist(rng)]++;
                }

                std::cout << "Randomized." << std::endl;
            }

            if (t && !tPrev) {
                speedMode = !speedMode;

                std::cout << "Speed mode: " << speedMode << std::endl;
            }

            bool w = sf::Keyboard::isKeyPressed(sf::Keyboard::W);
            bool a = sf::Keyboard::isKeyPressed(sf::Keyboard::A);
            bool s = sf::Keyboard::isKeyPressed(sf::Keyboard::S);
            bool d = sf::Keyboard::isKeyPressed(sf::Keyboard::D);

            if (s && !sPrev) {
                if (!buildHolding && buildStacks[buildPosition] > 0) {
                    buildStacks[buildPosition]--;

                    buildHolding = true;
                }
                else if (buildHolding && buildStacks[buildPosition] < height) {
                    buildStacks[buildPosition]++;

                    buildHolding = false;
                }
            }

            if (a && !aPrev) {
                if (buildPosition > 0)
                    buildPosition--;
            }

            if (d && !dPrev) {
                if (buildPosition < width - 1)
                    buildPosition++;
            }

            spacePrev = space;

            tPrev = t;
            wPrev = w;
            aPrev = a;
            sPrev = s;
            dPrev = d;
        }

        if (speedMode || slowTimer >= slowTime) {
            slowTimer = 0;

            int numSubTicks = speedMode ? 300 : 1;

            for (int subTick = 0; subTick < numSubTicks; subTick++) {
                Int_Buffer actualStates(width * height, 0);

                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < actualStacks[i]; j++)
                        actualStates[j + i * height] = 1;
                }

                Int_Buffer goalStates(width * height, 0);

                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < targetStacks[i]; j++)
                        goalStates[j + i * height] = 1;
                }

                Int_Buffer actions(1);
                actions[0] = actIndex;

                Int_Buffer positions(1);
                positions[0] = actualPosition;

                Array<Int_Buffer_View> inputCIs(ioDescs.size());
                inputCIs[0] = goalStates;
                inputCIs[1] = actualStates;
                inputCIs[2] = actions;
                inputCIs[3] = positions;

                float reward = 0.0f;

                for (int i = 0; i < actualStates.size(); i++) {
                    reward += actualStates[i] == goalStates[i];
                }

                reward /= actualStates.size();
                reward *= reward;

                h.step(inputCIs, true, reward);

                actIndex = h.get_prediction_cis(2)[0];

                if (speedMode) {
                    if (dist01(rng) < (speedMode ? 1.0f : 0.0f))
                        actIndex = actionDist(rng);
                }

                if (actIndex == 1) {
                    if (!actualHolding && actualStacks[actualPosition] > 0) {
                        actualStacks[actualPosition]--;

                        actualHolding = true;
                    }
                    else if (actualHolding && actualStacks[actualPosition] < height) {
                        actualStacks[actualPosition]++;

                        actualHolding = false;
                    }
                }

                if (actIndex == 2) {
                    if (actualPosition > 0)
                        actualPosition--;
                }

                if (actIndex == 3) {
                    if (actualPosition < width - 1)
                        actualPosition++;
                }
            }
        }

        slowTimer++;

        window.clear(sf::Color::Black);

        sf::RectangleShape rs;
        rs.setSize(sf::Vector2f(windowWidth, windowWidth * 0.5f));
        rs.setPosition(windowWidth * 0.5f, windowHeight * 0.5f);
        rs.setOrigin(windowWidth * 0.5f, windowWidth * 0.25f);
        rs.setFillColor(sf::Color::White);

        window.draw(rs);

        float blockScale = windowWidth * 0.5f / width;

        // Predictions
        std::cout << actIndex << std::endl;

        sf::Image predImg;
        predImg.create(width, height);

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                float intensity = h.get_prediction_cis(1)[y + x * height];

                sf::Color c;
                c.r = 255;
                c.g = c.b = static_cast<sf::Uint8>(std::min(1.0f, std::max(0.0f, 1.0f - intensity)) * 255.0f);
                
                predImg.setPixel(x, y, c);
            }

        sf::Texture predTex;
        predTex.loadFromImage(predImg);

        // Left side
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < actualStacks[i]; j++) {
                sf::RectangleShape bs;
                bs.setSize(sf::Vector2f(blockScale, blockScale));
                bs.setPosition(i * blockScale, windowHeight * 0.5f + windowWidth * 0.25f - (j + 1) * blockScale);
                bs.setFillColor(sf::Color::Green);

                window.draw(bs);
            }
        }

        if (window.hasFocus() && sf::Keyboard::isKeyPressed(sf::Keyboard::P)) {
            sf::Sprite s;
            s.setTexture(predTex);
            s.setPosition(0.0f, windowHeight * 0.5f + windowWidth * 0.25f);
            s.setScale(blockScale, -blockScale);

            window.draw(s);
        }

        // Right side
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < buildStacks[i]; j++) {
                sf::RectangleShape bs;
                bs.setSize(sf::Vector2f(blockScale, blockScale));
                bs.setPosition((width + i) * blockScale, windowHeight * 0.5f + windowWidth * 0.25f - (j + 1) * blockScale);
                bs.setFillColor(sf::Color::Blue);

                window.draw(bs);
            }
        }

        // Show position
        sf::RectangleShape as;
        as.setSize(sf::Vector2f(blockScale * 0.2f, blockScale * 0.4f));
        as.setOrigin(blockScale * 0.05f - blockScale * 0.5f, blockScale * 0.1f);

        as.setFillColor(actualHolding ? sf::Color::Red : sf::Color::Yellow);
        as.setPosition(actualPosition * blockScale, windowHeight * 0.5f - windowWidth * 0.25f);

        window.draw(as);

        as.setFillColor(buildHolding ? sf::Color::Red : sf::Color::Yellow);
        as.setPosition((width + buildPosition) * blockScale, windowHeight * 0.5f - windowWidth * 0.25f);

        window.draw(as);

        // Divider
        sf::RectangleShape div;
        div.setSize(sf::Vector2f(windowWidth * 0.01f, windowHeight));
        div.setPosition(windowWidth * 0.5f, windowHeight * 0.5f);
        div.setOrigin(div.getSize().x * 0.5f, div.getSize().y * 0.5f);

        div.setFillColor(sf::Color::Black);

        window.draw(div);

        window.display();
    } while (!quit);

    return 0;
}

