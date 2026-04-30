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
        aon::u64 len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        aon::u64 len
    ) override {
        outs.write(static_cast<const char*>(data), len);
    }
};

int main() {
    bool load = false;
    bool manualControl = false;
    float epsilon = 0.0f;

    const std::string hCatFileName = "hCat.ohr";
    const std::string hMouseFileName = "hMouse.ohr";
    
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(sf::Vector2u(1024, 1024)), "Cat Mouse", sf::Style::Default);

    window.setFramerateLimit(120);

    float aiDT = 1.0f / 30.0f;
    float aiTimer = 0.0f;

    float catRewardTotal = 0.0f;
    float mouseRewardTotal = 0.0f;

    sf::Image map("resources/map0.png");

    CatMouseEnv env;
    env.init(map);

    // --------------------------- Create the Hierarchy ---------------------------

    // Create hierarchy
    set_num_threads(8);

    Array<Hierarchy::Layer_Desc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(5, 5, 128);
        //lds[i].eRadius = 2;
        //lds[i].dRadius = 2;
        //lds[i].ticksPerUpdate = 4;
        //lds[i].temporalHorizon = 4;
    }

    int obsRes = 16;
    int actionRes = 5;

    Array<Hierarchy::IO_Desc> ioDescs(2);
    ioDescs[0] = Hierarchy::IO_Desc(Int3(7, 5, obsRes), IO_Type::none);
    ioDescs[1] = Hierarchy::IO_Desc(Int3(1, 3, actionRes), IO_Type::action);

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

        hCat.params.ios[1].importance = 0.1f;
        hMouse.params.ios[1].importance = 0.1f;

        //hCat.params.ios[1].actor.discount = 0.999f;
        //hCat.params.ios[1].actor.smoothing = 0.002f;

        //hMouse.params.ios[1].actor.discount = 0.999f;
        //hMouse.params.ios[1].actor.smoothing = 0.002f;

        std::cout << "Random init" << std::endl;
    }

    S32_Array catObsi(ioDescs[0].size.x * ioDescs[0].size.y, 0);
    S32_Array mouseObsi(catObsi.size(), 0);
    S32_Array catActionsi(env.actionsSize(), 0);
    S32_Array mouseActionsi(env.actionsSize(), 0);
    
    Array<S32_Array_View> catInputs(2);
    catInputs[0] = catObsi;
    catInputs[1] = catActionsi;

    Array<S32_Array_View> mouseInputs(2);
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

    view.setCenter(sf::Vector2f(map.getSize().x * 0.5f, map.getSize().y * 0.5f));
    view.zoom(0.0625f);

    window.setView(view);

    do {
        clock.restart();

        int numSubSteps = speedMode ? 500 : 1;

        for (int ss = 0; ss < numSubSteps; ss++) {
            // ----------------------------- Input -----------------------------

            while (const std::optional event = window.pollEvent()) {
                if (event->is<sf::Event::Closed>())
                    quit = true;
            }

            if (window.hasFocus()) {
                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape))
                    quit = true;

                bool tPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::T);

                if (tPressed && !tPressedPrev)
                    speedMode = !speedMode;

                tPressedPrev = tPressed;

                bool sPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S);

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

            float catCuriosity = 0.0f;

            for (int i = 0; i < catObsi.size(); i++)
                if (catObsi[i] != hCat.get_prediction_cis(0)[i])
                    catCuriosity++;

            catCuriosity /= catObsi.size();
                
            float catReward = env.getDone() * 100.0f;// + catCuriosity * 0.1f + catVisual * 1.0f;
            float mouseReward = env.getDone() * -100.0f;

            catRewardTotal += catReward;
            mouseRewardTotal += mouseReward;

            if (env.getDone()) {
                env.reset();
                numResets++;
                std::cout << "Reset " << numResets << std::endl;
            }

            if (aiTimer >= aiDT) {
                aiTimer = std::fmod(aiTimer, aiDT);

                const float rewardScale = 1.0f;

                hCat.step(catInputs, true, catRewardTotal * rewardScale);
                hMouse.step(mouseInputs, true, mouseRewardTotal * rewardScale);

                catRewardTotal = 0.0f;
                mouseRewardTotal = 0.0f;

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

                    //std::cout << "Action " << i << ": ";

                    //for (int j = 0; j < actionRes; j++)
                    //    std::cout << hCat.get_prediction_acts(1)[j + i * actionRes] << " ";

                    //std::cout << std::endl;
                }
            }

            aiTimer += dt;

            //std::cout << "FRAME" << std::endl;

            //std::cout << std::endl;

            if (manualControl) {
                catActions[0] = 0.5f;
                catActions[1] = 0.5f;
                catActions[2] = 0.5f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::K))
                    catActions[0] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::I))
                    catActions[0] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::J))
                    catActions[1] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::L))
                    catActions[1] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::U))
                    catActions[2] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::O))
                    catActions[2] = 1.0f;

                mouseActions[0] = 0.5f;
                mouseActions[1] = 0.5f;
                mouseActions[2] = 0.5f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
                    mouseActions[0] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
                    mouseActions[0] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
                    mouseActions[1] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
                    mouseActions[1] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Q))
                    mouseActions[2] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::E))
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
