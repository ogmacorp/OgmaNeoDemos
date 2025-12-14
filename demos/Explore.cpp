#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "explore/ExploreEnv.h"

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/helpers.h>
//#include <aogmaneo/image_encoder.h>
#include "vec.h"
#include <cmath>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>
#include <valarray>

const int S = 1024;
const int L = 32;

typedef v::Vec<S, L> Vec1;
typedef v::Bundle<S, L> Bundle1;

Vec1 embedding1d(float x, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 2] * x + loc[i * 2 + 1];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

float unembedding1d(const Vec1 &v, const std::valarray<float> &loc) {
    // mat mul
    float x = 0.0f;

    for (int i = 0; i < S; i++) {
        float val = v[i] / static_cast<float>(L);

        val = (val - loc[i * 2 + 1]) / v::max(v::limit_small, loc[i * 2]);

        x += val;
    }

    return x / S;
}

Vec1 embedding2d(float x, float y, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 3] * x + loc[i * 3 + 1] * y + loc[i * 3 + 2];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

Vec1 embedding3d(float x, float y, float z, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 4] * x + loc[i * 4 + 1] * y + loc[i * 4 + 2] * z + loc[i * 4 + 3];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

void print(const Vec1 &v) {
    std::cout << "[ ";

    for (int i = 0; i < v.segments(); i++)
        std::cout << static_cast<int>(v[i]) << " ";

    std::cout << " ]" << std::endl;
}

void print(const Bundle1 &b) {
    std::cout << "[ ";

    for (int i = 1; i < b.size(); i++)
        std::cout << b[i] << " ";

    std::cout << " ]" << std::endl;
}

class CustomStreamReader : public aon::Stream_Reader {
public:
    std::ifstream ins;

    void read(
        void* data,
        long len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        long len
    ) override {
        outs.write(static_cast<const char*>(data), len);
    }
};

int main() {
    bool load = false;
    bool manualControl = false;
    float epsilon = 0.01f;

    const std::string agentFileName = "explorer.ohr";
    
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(sf::Vector2u(1024, 1024)), "Cat Mouse", sf::Style::Default);

    window.setFramerateLimit(120);

    float aiDT = 1.0f / 30.0f;
    float aiTimer = 0.0f;

    float agentRewardTotal = 0.0f;
    float mouseRewardTotal = 0.0f;

    sf::Image map("resources/map_test.png");

    ExploreEnv env;
    env.init(map);

    // --------------------------- Create the Hierarchy ---------------------------

    // Create hierarchy
    aon::set_num_threads(8);

    aon::Array<aon::Hierarchy::Layer_Desc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = aon::Int3(5, 5, 32);
        //lds[i].eRadius = 2;
        //lds[i].dRadius = 2;
        //lds[i].ticksPerUpdate = 4;
        //lds[i].temporalHorizon = 4;
    }

    int obsRes = 16;
    int actionRes = 5;

    aon::Array<aon::Hierarchy::IO_Desc> ioDescs(2);
    ioDescs[0] = aon::Hierarchy::IO_Desc(aon::Int3(7, 5, obsRes), aon::IO_Type::prediction, 8, 2, 2);
    ioDescs[1] = aon::Hierarchy::IO_Desc(aon::Int3(1, 3, actionRes), aon::IO_Type::action, 8, 1, 2);

    aon::Hierarchy h;

    if (load) {
        CustomStreamReader reader;
        reader.ins.open(agentFileName.c_str(), std::ios::out | std::ios::binary);
        h.read(reader);

        std::cout << "Loaded" << std::endl;
    }
    else {
        h.init_random(ioDescs, lds);

        h.params.ios[1].importance = 0.1f;

        std::cout << "Random init" << std::endl;
    }

    aon::Int_Buffer agentObsi(ioDescs[0].size.x * ioDescs[0].size.y, 0);
    aon::Int_Buffer agentActionsi(env.actionsSize(), 0);
    
    aon::Array<aon::Int_Buffer_View> agentInputs(2);
    agentInputs[0] = agentObsi;
    agentInputs[1] = agentActionsi;

    std::vector<float> agentObs;
    std::vector<float> agentActions(env.actionsSize(), 0.5f);

    std::vector<Vec1> poseComps(3);

    for (int i = 0; i < poseComps.size(); i++)
        poseComps[i] = Vec1::randomized();

    std::vector<Vec1> dataElems(2);

    for (int i = 0; i < dataElems.size(); i++)
        dataElems[i] = Vec1::randomized();

    std::normal_distribution<float> ndist(0.0f, 1.0f);

    std::valarray<float> Z(S * 2); // location embedding values

    for (int i = 0; i < Z.size(); i++)
        Z[i] = ndist(rng) * 1.0f;

    // ---------------------------- Game Loop -----------------------------

    int numResets = 0;

    bool quit = false;

    bool speedMode = false;
    int tPressedPrev = false;
    bool sPressedPrev = false;

    sf::Clock clock;

    float dt = 1.0f / 120.0f;

    sf::View view;

    view.setCenter(sf::Vector2f(map.getSize().x * 0.5f, map.getSize().y * 0.5f));
    view.zoom(0.0625f);

    std::vector<Vec1> traversable = { Vec1::randomized(), Vec1::randomized() };

    // find default space
    Bundle1 base_space = 0.0f;
    float scale = 1.0f / 64.0f;

    for (int x = 0; x < map.getSize().x; x++) {
        for (int y = 0; y < map.getSize().y; y++) {
            Vec1 pose = (poseComps[0] * embedding1d(x * scale, Z) + poseComps[1] * embedding1d(y * scale, Z).permute(1) + poseComps[2] * embedding1d(x * scale, Z) * embedding1d(y * scale, Z).permute(1)).thin();

            Vec1 data = dataElems[1] * traversable[0];

            Vec1 item = data * pose;

            base_space += item;
        }
    }

    base_space *= 1.0f / (map.getSize().x * map.getSize().y);

    window.setView(view);

    Bundle1 space = 0.0f;

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
                    CustomStreamWriter writer;
                    writer.outs.open(agentFileName.c_str(), std::ios::out | std::ios::binary);
                    h.write(writer);

                    std::cout << "Saved." << std::endl;
                }

                sPressedPrev = sPressed;
            }

            env.getObs(agentObs);

            for (int i = 0; i < agentObs.size(); i++) {
                assert(agentObs[i] >= 0.0f && agentObs[i] <= 1.0f);

                agentObsi[i] = agentObs[i] * (obsRes - 1) + 0.5f;
            }

            //if (window.hasFocus() && sf::Keyboard::isKeyPressed(sf::Keyboard::C))
            //    std::cout << "Curiosity: " << agentCuriosity << std::endl;

            float agentCuriosity = 0.0f;

            for (int i = 0; i < agentObsi.size(); i++)
                if (agentObsi[i] != h.get_prediction_cis(0)[i])
                    agentCuriosity++;

            agentCuriosity /= agentObsi.size();
                
            float agentReward = agentCuriosity;// + env.getDone() * 100.0f;// + agentCuriosity * 0.1f + agentVisual * 1.0f;

            agentRewardTotal += agentReward;

            if (env.getDone()) {
                env.reset();
                numResets++;
                std::cout << "Reset " << numResets << std::endl;
            }

            if (aiTimer >= aiDT) {
                aiTimer = std::fmod(aiTimer, aiDT);

                const float rewardScale = 1.0f;

                h.step(agentInputs, true, agentRewardTotal * rewardScale);

                const aon::Int_Buffer &csdr = h.get_encoder(h.get_num_layers() - 1).get_hidden_cis();

                Vec1 vcsdr = 0;
                assert(csdr.size() >= S);

                for (int i = 0; i < csdr.size(); i++) {
                    assert(csdr[i] < L);
                    vcsdr[i] = csdr[i];
                }

                // bind
                Vec1 pose = (poseComps[0] * embedding1d(env.getAgent().pos.x * scale, Z) + poseComps[1] * embedding1d(env.getAgent().pos.y * scale, Z).permute(1) + poseComps[2] * embedding1d(env.getAgent().pos.x * scale, Z) * embedding1d(env.getAgent().pos.y * scale, Z).permute(1)).thin();

                Vec1 data = (dataElems[0] * vcsdr + dataElems[1] * traversable[1]).thin();

                Vec1 item = data * pose;

                space += 0.0001f * (item * 1.0f - space);

                agentRewardTotal = 0.0f;

                std::uniform_int_distribution<int> actionDist(0, actionRes - 1);

                for (int i = 0; i < agentActions.size(); i++) {
                    if (dist01(rng) < epsilon)
                        agentActionsi[i] = actionDist(rng);
                    else
                        agentActionsi[i] = h.get_prediction_cis(1)[i];

                    agentActions[i] = agentActionsi[i] / static_cast<float>(actionRes - 1);

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
                agentActions[0] = 0.5f;
                agentActions[1] = 0.5f;
                agentActions[2] = 0.5f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
                    agentActions[0] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
                    agentActions[0] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
                    agentActions[1] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
                    agentActions[1] = 1.0f;

                if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Q))
                    agentActions[2] = 0.0f;
                else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::E))
                    agentActions[2] = 1.0f;

            }

            env.step(agentActions, dt);
        }

        window.clear();

        env.render(window);

        sf::Image img(map.getSize());

        Vec1 thinned = (base_space * 4.0f + space).thin();

        for (int x = 0; x < img.getSize().x; x++) {
            for (int y = 0; y < img.getSize().y; y++) {
                Vec1 pose = (poseComps[0] * embedding1d(x * scale, Z) + poseComps[1] * embedding1d(y * scale, Z).permute(1) + poseComps[2] * embedding1d(x * scale, Z) * embedding1d(y * scale, Z).permute(1)).thin();

                Vec1 data = thinned / pose;

                Vec1 t = data / dataElems[1];

                bool trav = (t.dot(traversable[1]) > t.dot(traversable[0]));

                sf::Color c = (trav ? sf::Color::Green : sf::Color::Red);

                c.a = 64;

                img.setPixel(sf::Vector2u(x, y), c);
            }
        }

        sf::Texture t(img);

        sf::Sprite s(t);
        s.setScale(sf::Vector2f(4.0f, 4.0f));

        window.setView(window.getDefaultView());
        window.draw(s);
        window.setView(view);

        window.display();

    } while (!quit);

    return 0;
}
