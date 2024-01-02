#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "catmouse/CatMouseEnv.h"

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/helpers.h>
//#include <aogmaneo/image_encoder.h>
#include <cmath>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>

using namespace aon;

class CustomStreamReader : public aon::Stream_Reader {
public:
    std::ifstream ins;

    void read(
        void* data,
        int len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        int len
    ) override {
        outs.write(static_cast<const char*>(data), len);
    }
};

int main() {
    bool load = false;
    bool manualControl = false;
    float epsilon = 0.05f;

    const std::string hCatFileName = "hCat.ohr";
    const std::string hMouseFileName = "hMouse.ohr";
    
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(1024, 1024), "Cat Mouse", sf::Style::Default);

    window.setFramerateLimit(120);

    float aiDT = 1.0f / 30.0f;
    float aiTimer = 0.0f;

    float catRewardTotal = 0.0f;
    float mouseRewardTotal = 0.0f;

    sf::Image map;
    map.loadFromFile("resources/map0.png");

    CatMouseEnv env;
    env.init(map);

    // --------------------------- Create the Hierarchy ---------------------------

    // Create hierarchy
    set_num_threads(8);

    Array<Hierarchy::Layer_Desc> lds(5);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(5, 5, 32);
        //lds[i].eRadius = 2;
        //lds[i].dRadius = 2;
        //lds[i].ticksPerUpdate = 4;
        //lds[i].temporalHorizon = 4;
    }

    int obsRes = 32;
    int actionRes = 5;

    Array<Hierarchy::IO_Desc> ioDescs(2);
    ioDescs[0] = Hierarchy::IO_Desc(Int3(4, 3, obsRes), IO_Type::prediction, 2, 2);
    ioDescs[1] = Hierarchy::IO_Desc(Int3(1, 3, actionRes), IO_Type::action, 1, 2);

    Hierarchy hCat;
    Hierarchy hMouse;

    if (load) {
        {
            CustomStreamReader reader;
            reader.ins.open(hCatFileName.c_str(), std::ios::out | std::ios::binary);
            hCat.read(reader);
        }

        {
            CustomStreamReader reader;
            reader.ins.open(hMouseFileName.c_str(), std::ios::out | std::ios::binary);
            hMouse.read(reader);
        }

        std::cout << "Loaded" << std::endl;
    }
    else {
        hCat.init_random(ioDescs, lds);
        hMouse.init_random(ioDescs, lds);

        hCat.params.ios[1].importance = 0.0f;
        hMouse.params.ios[1].importance = 0.0f;

        std::cout << "Random init" << std::endl;
    }

    Int_Buffer catObsi(ioDescs[0].size.x * ioDescs[0].size.y, 0);
    Int_Buffer mouseObsi(catObsi.size(), 0);
    Int_Buffer catActionsi(env.actionsSize(), 0);
    Int_Buffer mouseActionsi(env.actionsSize(), 0);
    
    Array<Int_Buffer_View> catInputs(2);
    catInputs[0] = catObsi;
    catInputs[1] = catActionsi;

    Array<Int_Buffer_View> mouseInputs(2);
    mouseInputs[0] = mouseObsi;
    mouseInputs[1] = mouseActionsi;

    std::vector<float> catObs, mouseObs;
    std::vector<float> catActions(env.actionsSize(), 0.5f);
    std::vector<float> mouseActions(env.actionsSize(), 0.5f);

    int numResets = 0;

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    bool speedMode = false;
    int tPressedPrev = false;
    bool sPressedPrev = false;

    sf::Clock clock;

    float dt = 1.0f / 120.0f;

    sf::View view;

    view.setCenter(map.getSize().x * 0.5f, map.getSize().y * 0.5f);
    view.zoom(0.0625f);

    window.setView(view);

    do {
        clock.restart();

        int numSubSteps = speedMode ? 300 : 1;

        for (int ss = 0; ss < numSubSteps; ss++) {
            // ----------------------------- Input -----------------------------

            sf::Event windowEvent;

            while (window.pollEvent(windowEvent)) {
                switch (windowEvent.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                default:
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

                if (sPressed && !sPressedPrev && !manualControl) {
                    {
                        CustomStreamWriter writer;
                        writer.outs.open(hCatFileName.c_str(), std::ios::out | std::ios::binary);
                        hCat.write(writer);
                    }

                    {
                        CustomStreamWriter writer;
                        writer.outs.open(hMouseFileName.c_str(), std::ios::out | std::ios::binary);
                        hMouse.write(writer);
                    }

                    std::cout << "Saved." << std::endl;
                }

                sPressedPrev = sPressed;
            }

            float catVisual, mouseVisual;

            env.getObs(catObs, mouseObs, catVisual, mouseVisual);

            for (int i = 0; i < catObs.size(); i++) {
                assert(catObs[i] >= 0.0f && catObs[i] <= 1.0f);
                assert(mouseObs[i] >= 0.0f && mouseObs[i] <= 1.0f);

                catObsi[i] = catObs[i] * (obsRes - 1) + 0.5f;
                mouseObsi[i] = mouseObs[i] * (obsRes - 1) + 0.5f;
            }

            //if (window.hasFocus() && sf::Keyboard::isKeyPressed(sf::Keyboard::C))
            //    std::cout << "Curiosity: " << catCuriosity << std::endl;

            float catReward = env.getDone() * 10.0f;// + catCuriosity * 0.1f + catVisual * 1.0f;
            float mouseReward = env.getDone() * -10.0f;

            catRewardTotal += catReward;
            mouseRewardTotal += mouseReward;

            if (env.getDone()) {
                env.reset();
                numResets++;
                std::cout << "Reset " << numResets << std::endl;
            }

            if (aiTimer >= aiDT) {
                aiTimer = std::fmod(aiTimer, aiDT);

                hCat.step(catInputs, true, catRewardTotal);
                hMouse.step(mouseInputs, true, mouseRewardTotal);

                catRewardTotal = 0.0f;
                mouseRewardTotal = 0.0f;
            }

            aiTimer += dt;

            std::uniform_int_distribution<int> actionDist(0, actionRes - 1);

            for (int i = 0; i < catActions.size(); i++) {
                if (dist01(rng) < epsilon)
                    catActionsi[i] = actionDist(rng);
                else
                    catActionsi[i] = hCat.get_prediction_cis(1)[i];

                if (dist01(rng) < epsilon)
                    mouseActionsi[i] = actionDist(rng);
                else
                    mouseActionsi[i] = hMouse.get_prediction_cis(1)[i];

                catActions[i] = catActionsi[i] / static_cast<float>(actionRes - 1);
                mouseActions[i] = mouseActionsi[i] / static_cast<float>(actionRes - 1);
            }

            if (manualControl) {
                catActions[0] = 0.5f;
                catActions[1] = 0.5f;
                catActions[2] = 0.5f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::K))
                    catActions[0] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::I))
                    catActions[0] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::J))
                    catActions[1] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::L))
                    catActions[1] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::U))
                    catActions[2] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::O))
                    catActions[2] = 1.0f;

                mouseActions[0] = 0.5f;
                mouseActions[1] = 0.5f;
                mouseActions[2] = 0.5f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
                    mouseActions[0] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
                    mouseActions[0] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
                    mouseActions[1] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
                    mouseActions[1] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
                    mouseActions[2] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::E))
                    mouseActions[2] = 1.0f;
            }

            env.step(catActions, mouseActions, dt);
        }

        window.clear();

        env.render(window);

        window.display();

    } while (!quit);

    return 0;
}
