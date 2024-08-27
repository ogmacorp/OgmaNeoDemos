#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/image_encoder.h>

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
    
    set_num_threads(8);

    Int3 hiddenSize(3, 3, 8);

    Array<Image_Encoder::Visible_Layer_Desc> vlds(1);
    vlds[0].size = Int3(width, height, 1);
    vlds[0].radius = 0;

    Image_Encoder enc;
    enc.init_random(hiddenSize, vlds);

    Array<Hierarchy::Layer_Desc> lds(5);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(5, 5, 16);

        lds[i].ticks_per_update = 2;
        lds[i].temporal_horizon = 4;
    }

    Array<Hierarchy::IO_Desc> ioDescs(3);
    ioDescs[0] = Hierarchy::IO_Desc(hiddenSize);
    ioDescs[1] = Hierarchy::IO_Desc(Int3(1, 1, numActions));
    ioDescs[2] = Hierarchy::IO_Desc(Int3(1, 1, width));

    Hierarchy h;
    h.init_random(ioDescs, lds);

    Int_Buffer goalCIs = h.get_top_hidden_cis();
    Int_Buffer randomCIs = h.get_top_hidden_cis();

    int actIndex = 0;

    int slowTimer = 0;
    int slowTime = 30;

    bool stateSaved = false;

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

                stateSaved = true;
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
                Byte_Buffer img(width * height, 0);

                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < actualStacks[i]; j++)
                        img[i * height + j] = 255;
                }

                Array<Byte_Buffer_View> encInputs(1);
                encInputs[0] = img;

                enc.step(encInputs, speedMode);

                Int_Buffer imgCIs = enc.get_hidden_cis();

                Int_Buffer actions(1);
                actions[0] = actIndex;

                Int_Buffer positions(1);
                positions[0] = actualPosition;

                Array<Int_Buffer_View> inputCIs(ioDescs.size());
                inputCIs[0] = imgCIs;
                inputCIs[1] = actions;
                inputCIs[2] = positions;

                Byte_Buffer goalImg(width * height, 0);

                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < targetStacks[i]; j++)
                        goalImg[i * height + j] = 255;
                }

                encInputs[0] = goalImg;

                enc.step(encInputs, false);

                Int_Buffer goalImgCIs = enc.get_hidden_cis();

                if (stateSaved) {
                    stateSaved = false;

                    // TP
                    aon::Hierarchy copy = h;

                    for (int ss = 0; ss < 32; ss++) {
                        Int_Buffer noAction(1);
                        noAction[0] = 0;
                        Int_Buffer noPosition(1);
                        noPosition[0] = 0;

                        Array<Int_Buffer_View> copyInputCIs(ioDescs.size());
                        copyInputCIs[0] = goalImgCIs;
                        copyInputCIs[1] = noAction;
                        copyInputCIs[2] = noPosition;

                        copy.step(copyInputCIs, goalCIs, false);
                        
                        goalCIs = copy.get_top_hidden_cis();
                    }
                }

                if (speedMode) {
                    if (dist01(rng) < 0.1f) {
                        std::uniform_int_distribution<int> goalDist(0, h.get_top_hidden_size().z);

                        for (int i = 0; i < randomCIs.size(); i++)
                            randomCIs[i] = goalDist(rng);
                    }

                    h.step(inputCIs, randomCIs, speedMode);
                }
                else
                    h.step(inputCIs, goalCIs, false);

                actIndex = h.get_prediction_cis(1)[0];

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

        // Predictions
        enc.reconstruct(h.get_prediction_cis(0));

        std::cout << actIndex << std::endl;

        sf::Image predImg;
        predImg.create(width, height);

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                Byte intensity = enc.get_reconstruction(0)[y + x * height];

                sf::Color c;
                c.r = 255;
                c.g = c.b = static_cast<sf::Uint8>(std::min(1.0f, std::max(0.0f, 1.0f - intensity / 255.0f)) * 255.0f);
                
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

