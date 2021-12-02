#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/StateAdapter.h>
//#include <aogmaneo/ImageEncoder.h>
#include <cmath>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>

using namespace aon;

class CustomStreamReader : public aon::StreamReader {
public:
    std::ifstream ins;

    void read(
        void* data,
        int len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::StreamWriter {
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
    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(1024, 1024), "Pusher", sf::Style::Default);

    window.setFramerateLimit(20);

    std::string encFileName = "pusher.oenc";
    std::string hFileName = "pusher.ohr";

    // --------------------------- Create the Hierarchy ---------------------------

    // Create hierarchy
    setNumThreads(8);

    Array<Hierarchy::LayerDesc> lds(3);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(4, 4, 16);
        lds[i].ticksPerUpdate = 2;
        lds[i].temporalHorizon = 2;
    }

    int sensorRes = 33;
    int actionRes = 10;

    Array<Hierarchy::IODesc> ioDescs(2);
    ioDescs[0] = Hierarchy::IODesc(Int3(2, 2, sensorRes), IOType::prediction, 2, 2, 64);
    ioDescs[1] = Hierarchy::IODesc(Int3(1, 2, actionRes), IOType::action, 2, 2, 64);

    Hierarchy h;
    h.initRandom(ioDescs, lds);

    //CustomStreamReader reader;
    //reader.ins.open(hFileName.c_str(), std::ios::out | std::ios::binary);
    //h.read(reader);

    // -------------------------- Game Resources --------------------------

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    bool speedMode = false;
    bool tPressedPrev = false;
    bool sPressedPrev = false;
    bool lPressedPrev = false;

    sf::Clock clock;

    float dt = 0.017f;

    sf::View view;

    view.setCenter(0.0f, 0.0f);

    view.setSize(sf::Vector2f(2.0f, 2.0f));

    window.setView(view);

    sf::Texture whitenedTex;

    // Used for speed mode to render slower
    int renderCounter = 0;

    float averageReward = 0.0f;

    float distPrev = -1.0f;

    sf::Vector2f objectPos(0.3f, 0.3f);
    sf::Vector2f pusherPos(0.0f, 0.0f);

    sf::Vector2f targetPos(0.0f, 0.0f);

    float objectRad = 0.08f;
    float pusherRad = 0.08f;

    bool learnMode = true;

    do {
        clock.restart();

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
                CustomStreamWriter writer;
                writer.outs.open(hFileName.c_str(), std::ios::out | std::ios::binary);
                h.write(writer);
            }

            sPressedPrev = sPressed;

            bool lPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::L);

            if (lPressed && !lPressedPrev)
                learnMode = !learnMode;

            lPressedPrev = lPressed;

            sf::Vector2i mousePos = sf::Mouse::getPosition(window);

            targetPos.x = mousePos.x / static_cast<float>(window.getSize().x) * 2.0f - 1.0f;
            targetPos.y = mousePos.y / static_cast<float>(window.getSize().y) * 2.0f - 1.0f;
        }

        sf::Vector2f deltaBetween = objectPos - pusherPos;

        float distBetween = std::sqrt(deltaBetween.x * deltaBetween.x + deltaBetween.y * deltaBetween.y);

        if (distBetween < objectRad + pusherRad) {
            float push = objectRad + pusherRad - distBetween;

            objectPos += push * deltaBetween / distBetween;
        }

        sf::Vector2f delta = targetPos - pusherPos;

        float mag = std::sqrt(delta.x * delta.x + delta.y * delta.y);

        float maxSpeed = 0.05f;

        if (mag > maxSpeed)
            delta *= maxSpeed / mag;

        delta.x = maxSpeed * (h.getPredictionCIs(1)[0] / static_cast<float>(actionRes - 1) * 2.0f - 1.0f);
        delta.y = maxSpeed * (h.getPredictionCIs(1)[1] / static_cast<float>(actionRes - 1) * 2.0f - 1.0f);

        pusherPos += delta;

        if (pusherPos.x > 1.0f)
            pusherPos.x = 1.0f;
        else if (pusherPos.x < -1.0f)
            pusherPos.x = -1.0f;

        if (pusherPos.y > 1.0f)
            pusherPos.y = 1.0f;
        else if (pusherPos.y < -1.0f)
            pusherPos.y = -1.0f;

        float distToCenter = std::sqrt(objectPos.x * objectPos.x + objectPos.y * objectPos.y);

        if (distPrev == -1.0f)
            distPrev = distToCenter;

        float reward = -(distToCenter - distPrev);

        distPrev = distToCenter;

        if (distToCenter < 0.04f || objectPos.x < -1.0f || objectPos.x > 1.0f || objectPos.y < -1.0f || objectPos.y > 1.0f) {
            // Reset
            objectPos = sf::Vector2f(dist01(rng) * 2.0f - 1.0f, dist01(rng) * 2.0f - 1.0f) * 0.6f;

            reward = 10.0f;

            distPrev = -1.0f;
        }
        //std::cout << reward << std::endl;

        IntBuffer sensorCIs(4);
        sensorCIs[0] = (pusherPos.x * 0.5f + 0.5f) * (sensorRes - 1) + 0.5f;
        sensorCIs[1] = (pusherPos.y * 0.5f + 0.5f) * (sensorRes - 1) + 0.5f;
        sensorCIs[2] = (objectPos.x * 0.5f + 0.5f) * (sensorRes - 1) + 0.5f;
        sensorCIs[3] = (objectPos.y * 0.5f + 0.5f) * (sensorRes - 1) + 0.5f;

        Array<const IntBuffer*> inputCIs(2);
        inputCIs[0] = &sensorCIs;
        inputCIs[1] = &h.getPredictionCIs(1);

        h.step(inputCIs, true, reward);

        if (!speedMode || renderCounter >= 100) {
            window.clear();

            renderCounter = 0;
            
            sf::CircleShape cs;

            cs.setRadius(objectRad);
            cs.setOrigin(objectRad, objectRad);
            cs.setPosition(objectPos);
            cs.setFillColor(sf::Color::Red);

            window.draw(cs);

            cs.setRadius(pusherRad);
            cs.setOrigin(pusherRad, pusherRad);
            cs.setPosition(pusherPos);
            cs.setFillColor(sf::Color::Blue);

            window.draw(cs);

            cs.setRadius(0.01f);
            cs.setOrigin(0.01f, 0.01f);
            cs.setPosition(0.0f, 0.0f);
            cs.setFillColor(sf::Color::Green);

            window.draw(cs);

            window.display();
        }

        renderCounter++;
    } while (!quit);

    return 0;
}
