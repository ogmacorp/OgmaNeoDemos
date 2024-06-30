#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/ImageEncoder.h>

using namespace aon;

const int width = 3;
const int height = 3;
const int numActions = 4;
const int numBlocks = 3;

const float exploration = 0.05f;

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
    
    setNumThreads(1);

    Int3 hiddenSize(1, 1, 16);

    Array<ImageEncoder::VisibleLayerDesc> vlds(1);
    vlds[0].size = Int3(width, height, 1);
    vlds[0].radius = 2;

    ImageEncoder enc;
    enc.initRandom(hiddenSize, vlds);

    Array<Hierarchy::InputDesc> ids(3);
    ids[0].size = hiddenSize;
    ids[1].size = Int3(1, 1, numActions);
    ids[1].type = aon::action;
    ids[2].size = Int3(1, 1, width);

    Array<Hierarchy::HiddenDesc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].size = Int3(4, 4, 64);
        lds[i].rRadius = -1;
    }

    Array<Hierarchy::OutputDesc> ods(1);
    ods[0].size = hiddenSize;

    Hierarchy h;
    h.initRandom(ids, lds, ods);

    int actIndex = 0;

    int slowTimer = 0;
    int slowTime = 30;

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

            int numSubTicks = speedMode ? 100 : 1;

            for (int subTick = 0; subTick < numSubTicks; subTick++) {
                FloatBuffer img(width * height, 0.0f);

                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < actualStacks[i]; j++)
                        img[i * height + j] = 1.0f;
                }

                Array<const FloatBuffer*> encInputs(1);
                encInputs[0] = &img;

                enc.step(encInputs, speedMode);

                IntBuffer imgCIs = enc.getHiddenCIs();

                IntBuffer actions(1);
                actions[0] = actIndex;

                IntBuffer positions(1);
                positions[0] = actualPosition;

                Array<const IntBuffer*> inputCIs(ids.size());
                inputCIs[0] = &imgCIs;
                inputCIs[1] = &actions;
                inputCIs[2] = &positions;

                FloatBuffer goalImg(width * height, 0.0f);

                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < targetStacks[i]; j++)
                        goalImg[i * height + j] = 1.0f;
                }

                encInputs[0] = &goalImg;

                enc.step(encInputs, false);

                IntBuffer goalImgCIs = enc.getHiddenCIs();

                Array<const IntBuffer*> goalCIs(1);
                goalCIs[0] = &goalImgCIs;

                Array<const IntBuffer*> actualCIs(1);
                actualCIs[0] = &imgCIs;

                h.step(inputCIs, actualCIs, goalCIs, speedMode);

                IntBuffer outputCIs = h.getOutputCIs(0);

                for (int i = 0; i < outputCIs.size(); i++)
                    std::cout << imgCIs[i] << " ";

                std::cout << "||||||| ";

                for (int i = 0; i < outputCIs.size(); i++)
                    std::cout << outputCIs[i] << " ";

                std::cout << std::endl;

                actIndex = h.getActionCIs(1)[0];

                if (speedMode)
                    actIndex = actionDist(rng);

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

