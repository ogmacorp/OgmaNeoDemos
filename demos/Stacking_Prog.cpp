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

    sf::RenderWindow window(sf::VideoMode(sf::Vector2u(windowWidth, windowHeight)), "Stacking", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    sf::Font font("resources/Hack-Regular.ttf");

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

    //Int3 hiddenSize(3, 3, 8);

    //Array<Image_Encoder::Visible_Layer_Desc> vlds(1);
    //vlds[0].size = Int3(width, height, 1);
    //vlds[0].radius = 0;

    //Image_Encoder enc;
    //enc.init_random(hiddenSize, vlds);

    Array<Hierarchy::Layer_Desc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(5, 5, 32);
    }

    Array<Hierarchy::IO_Desc> ioDescs(3);
    ioDescs[0] = Hierarchy::IO_Desc(Int3(width, height, 2));
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
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                quit = true;
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape))
                quit = true;

            bool space = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Space);
            bool t = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::T);

            if (space && !spacePrev) {
                targetStacks = buildStacks;

                std::cout << "State saved." << std::endl;

                stateSaved = true;
            }

            if (t && !tPrev) {
                speedMode = !speedMode;

                std::cout << "Speed mode: " << speedMode << std::endl;
            }

            bool w = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W);
            bool a = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A);
            bool s = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S);
            bool d = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D);

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
                Int_Buffer imgCIs(width * height, 0);

                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < actualStacks[i]; j++) {
                        img[i * height + j] = 255;
                        imgCIs[i * height + j] = 1;
                    }
                }

                //Array<Byte_Buffer_View> encInputs(1);
                //encInputs[0] = img;

                //enc.step(encInputs, speedMode);

                //Int_Buffer imgCIs = enc.get_hidden_cis();

                Int_Buffer actions(1);
                actions[0] = actIndex;

                Int_Buffer positions(1);
                positions[0] = actualPosition;

                Array<Int_Buffer_View> inputCIs(ioDescs.size());
                inputCIs[0] = imgCIs;
                inputCIs[1] = actions;
                inputCIs[2] = positions;

                Byte_Buffer goalImg(width * height, 0);
                Int_Buffer goalImgCIs(width * height, 0);


                for (int i = 0; i < width; i++) {
                    for (int j = 0; j < targetStacks[i]; j++) {
                        goalImg[i * height + j] = 255;
                        goalImgCIs[i * height + j] = 1;
                    }
                }

                //encInputs[0] = goalImg;

                //enc.step(encInputs, false);

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
                        std::uniform_int_distribution<int> goalDist(0, h.get_top_hidden_size().z - 1);

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
        rs.setPosition(sf::Vector2f(windowWidth * 0.5f, windowHeight * 0.5f));
        rs.setOrigin(sf::Vector2f(windowWidth * 0.5f, windowWidth * 0.25f));
        rs.setFillColor(sf::Color::White);

        window.draw(rs);

        float blockScale = windowWidth * 0.5f / width;

        // Predictions
        std::cout << actIndex << std::endl;

        sf::Image predImg(sf::Vector2u(width, height));

        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                float intensity = h.get_prediction_cis(1)[y + x * height];

                sf::Color c;
                c.r = 255;
                c.g = c.b = static_cast<std::uint8_t>(std::min(1.0f, std::max(0.0f, 1.0f - intensity)) * 255.0f);
                
                predImg.setPixel(sf::Vector2u(x, y), c);
            }

        sf::Texture predTex(predImg);

        // Left side
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < actualStacks[i]; j++) {
                sf::RectangleShape bs;
                bs.setSize(sf::Vector2f(blockScale, blockScale));
                bs.setPosition(sf::Vector2f(i * blockScale, windowHeight * 0.5f + windowWidth * 0.25f - (j + 1) * blockScale));
                bs.setFillColor(sf::Color::Green);

                window.draw(bs);
            }
        }

        if (window.hasFocus() && sf::Keyboard::isKeyPressed(sf::Keyboard::Key::P)) {
            sf::Sprite s(predTex);
            s.setPosition(sf::Vector2f(0.0f, windowHeight * 0.5f + windowWidth * 0.25f));
            s.setScale(sf::Vector2f(blockScale, -blockScale));

            window.draw(s);
        }

        // Right side
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < buildStacks[i]; j++) {
                sf::RectangleShape bs;
                bs.setSize(sf::Vector2f(blockScale, blockScale));
                bs.setPosition(sf::Vector2f((width + i) * blockScale, windowHeight * 0.5f + windowWidth * 0.25f - (j + 1) * blockScale));
                bs.setFillColor(sf::Color::Blue);

                window.draw(bs);
            }
        }

        // Show position
        sf::RectangleShape as;
        as.setSize(sf::Vector2f(blockScale * 0.2f, blockScale * 0.4f));
        as.setOrigin(sf::Vector2f(blockScale * 0.05f - blockScale * 0.5f, blockScale * 0.1f));

        as.setFillColor(actualHolding ? sf::Color::Red : sf::Color::Yellow);
        as.setPosition(sf::Vector2f(actualPosition * blockScale, windowHeight * 0.5f - windowWidth * 0.25f));

        window.draw(as);

        as.setFillColor(buildHolding ? sf::Color::Red : sf::Color::Yellow);
        as.setPosition(sf::Vector2f((width + buildPosition) * blockScale, windowHeight * 0.5f - windowWidth * 0.25f));

        window.draw(as);

        // Divider
        sf::RectangleShape div;
        div.setSize(sf::Vector2f(windowWidth * 0.01f, windowHeight));
        div.setPosition(sf::Vector2f(windowWidth * 0.5f, windowHeight * 0.5f));
        div.setOrigin(sf::Vector2f(div.getSize().x * 0.5f, div.getSize().y * 0.5f));

        div.setFillColor(sf::Color::Black);

        window.draw(div);

        window.display();
    } while (!quit);

    return 0;
}

