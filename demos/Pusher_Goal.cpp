#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/ImageEncoder.h>
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

    window.create(sf::VideoMode(1024, 1024), "Pusher_Goal", sf::Style::Default);

    window.setFramerateLimit(60);

    std::string encFileName = "pusher_goal.oenc";
    std::string hFileName = "pusher_goal.ohr";

    // --------------------------- Create the Hierarchy ---------------------------

    // Create hierarchy
    setNumThreads(8);

    Int3 imgSize(32, 32, 3); // 3 channels, one for center, one for pusher, one for object

    Array<ImageEncoder::VisibleLayerDesc> vlds(1);
    vlds[0].size = imgSize;
    vlds[0].radius = 3;

    ImageEncoder enc;
    enc.initRandom(Int3(8, 8, 16), vlds);

    Array<Hierarchy::LayerDesc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(5, 5, 16);
        lds[i].ticksPerUpdate = 2;
        lds[i].temporalHorizon = 2;
    }

    int actionRes = 3;

    Array<Hierarchy::IODesc> ioDescs(2);
    ioDescs[0] = Hierarchy::IODesc(enc.getHiddenSize(), IOType::prediction, 2, 2, 32);
    ioDescs[1] = Hierarchy::IODesc(Int3(1, 2, actionRes), IOType::prediction, 2, 2, 32);

    Array<Hierarchy::GDesc> gDescs(1);
    gDescs[0].size = enc.getHiddenSize();
    gDescs[0].radius = 2;

    Hierarchy h;
    h.initRandom(ioDescs, gDescs, lds);

    IntBuffer actionCIs = h.getPredictionCIs(1);

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
    float objectDistPrev = -1.0f;

    sf::Vector2f objectPos(0.3f, 0.3f);
    sf::Vector2f pusherPos(0.0f, 0.0f);

    sf::Vector2f targetPos(0.0f, 0.0f);

    sf::Vector2f centerPos(0.0f, 0.0f);

    float objectRad = 0.12f;
    float pusherRad = 0.12f;

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

        float maxSpeed = 0.08f;

        if (mag > maxSpeed)
            delta *= maxSpeed / mag;

        actionCIs = h.getPredictionCIs(1);

        // Exploration
        for (int i = 0; i < actionCIs.size(); i++) {
            if (dist01(rng) < (speedMode ? 0.5f : 0.0f)) {
                std::uniform_int_distribution<int> actionDist(0, actionRes - 1);

                actionCIs[i] = actionDist(rng);
            }
        }

        delta.x = maxSpeed * (actionCIs[0] / static_cast<float>(actionRes - 1) * 2.0f - 1.0f);
        delta.y = maxSpeed * (actionCIs[1] / static_cast<float>(actionRes - 1) * 2.0f - 1.0f);

        pusherPos += delta;

        if (pusherPos.x > 1.0f)
            pusherPos.x = 1.0f;
        else if (pusherPos.x < -1.0f)
            pusherPos.x = -1.0f;

        if (pusherPos.y > 1.0f)
            pusherPos.y = 1.0f;
        else if (pusherPos.y < -1.0f)
            pusherPos.y = -1.0f;

        sf::Vector2f deltaCenter = centerPos - objectPos;

        float distToCenter = std::sqrt(deltaCenter.x * deltaCenter.x + deltaCenter.y * deltaCenter.y);

        sf::Vector2f objectDelta = objectPos - pusherPos;

        float distToObject = std::sqrt(objectDelta.x * objectDelta.x + objectDelta.y * objectDelta.y);

        if (distPrev == -1.0f)
            distPrev = distToCenter;

        if (objectDistPrev == -1.0f)
            objectDistPrev = distToObject;

        float reward = -5.0f * (distToCenter - distPrev) - 1.0f * (distToObject - objectDistPrev);

        distPrev = distToCenter;
        objectDistPrev = distToObject;

        bool outOfBounds = objectPos.x < -1.0f || objectPos.x > 1.0f || objectPos.y < -1.0f || objectPos.y > 1.0f;

        if (distToCenter < 0.06f || outOfBounds) {
            // Reset
            objectPos = sf::Vector2f(dist01(rng) * 2.0f - 1.0f, dist01(rng) * 2.0f - 1.0f) * 0.6f;
            centerPos = sf::Vector2f(dist01(rng) * 2.0f - 1.0f, dist01(rng) * 2.0f - 1.0f) * 0.6f;

            reward = outOfBounds ? -10.0f : 100.0f;

            if (reward == 100.0f) {
                std::cout << "Made it!" << std::endl;
            }
            else {
                std::cout << "Out of bounds!" << std::endl;
            }

            distPrev = -1.0f;
            objectDistPrev = -1.0f;
        }
        //std::cout << reward << std::endl;
        
        FloatBuffer img(imgSize.x * imgSize.y * imgSize.z);
        FloatBuffer gimg(img.size());
        float falloff = 0.01f;

        for (int x = 0; x < imgSize.x; x++)
            for (int y = 0; y < imgSize.y; y++) {
                sf::Vector2f dc = sf::Vector2f((centerPos.x * 0.5f + 0.5f) * imgSize.x + 0.5f, (centerPos.y * 0.5f + 0.5f) * imgSize.y + 0.5f) - sf::Vector2f(x, y);
                sf::Vector2f dobj = sf::Vector2f((objectPos.x * 0.5f + 0.5f) * imgSize.x + 0.5f, (objectPos.y * 0.5f + 0.5f) * imgSize.y + 0.5f) - sf::Vector2f(x, y);
                sf::Vector2f dpush = sf::Vector2f((pusherPos.x * 0.5f + 0.5f) * imgSize.x + 0.5f, (pusherPos.y * 0.5f + 0.5f) * imgSize.y + 0.5f) - sf::Vector2f(x, y);

                img[0 + 3 * (y + x * imgSize.y)] = std::exp(-falloff * (dc.x * dc.x + dc.y * dc.y)) * 1.0f;
                img[1 + 3 * (y + x * imgSize.y)] = std::exp(-falloff * (dobj.x * dobj.x + dobj.y * dobj.y)) * 1.0f;
                img[2 + 3 * (y + x * imgSize.y)] = std::exp(-falloff * (dpush.x * dpush.x + dpush.y * dpush.y)) * 1.0f;
                gimg[0 + 3 * (y + x * imgSize.y)] = std::exp(-falloff * (dc.x * dc.x + dc.y * dc.y)) * 1.0f;
                gimg[1 + 3 * (y + x * imgSize.y)] = std::exp(-falloff * (dc.x * dc.x + dc.y * dc.y)) * 1.0f;
                gimg[2 + 3 * (y + x * imgSize.y)] = std::exp(-falloff * (dobj.x * dobj.x + dobj.y * dobj.y)) * 1.0f;
            }

        Array<const FloatBuffer*> imgs(1);
        imgs[0] = &img;

        enc.step(imgs, true);

        IntBuffer actualHiddenCIs = enc.getHiddenCIs();

        imgs[0] = &gimg;

        enc.step(imgs, false);

        IntBuffer goalHiddenCIs = enc.getHiddenCIs();

        Array<const IntBuffer*> inputCIs(2);
        inputCIs[0] = &actualHiddenCIs;
        inputCIs[1] = &actionCIs;

        Array<const IntBuffer*> goalCIs(1);
        goalCIs[0] = &goalHiddenCIs;

        Array<const IntBuffer*> actualCIs(1);
        actualCIs[0] = &actualHiddenCIs;

        h.step(inputCIs, goalCIs, actualCIs, true);

        sf::Image imgImg;

        imgImg.create(imgSize.x, imgSize.y);

        for (int x = 0; x < imgSize.x; x++)
            for (int y = 0; y < imgSize.y; y++) {
                sf::Color c;
                c.r = img[0 + 3 * (y + x * imgSize.y)] * 255.0f;
                c.g = img[1 + 3 * (y + x * imgSize.y)] * 255.0f;
                c.b = img[2 + 3 * (y + x * imgSize.y)] * 255.0f;
                c.a = 255;
                imgImg.setPixel(x, y, c);
            }

        if (!speedMode || renderCounter >= 300) {
            window.clear();

            renderCounter = 0;

            sf::Texture imgTex;
            imgTex.loadFromImage(imgImg);

            sf::Sprite s;
            s.setTexture(imgTex);
            s.setScale(2.0f / imgSize.x, 2.0f / imgSize.y);
            s.setColor(sf::Color(225, 255, 255, 127));
            s.setPosition(-1.0f, -1.0f);
            window.draw(s);
            
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

            cs.setRadius(0.06f);
            cs.setOrigin(0.01f, 0.01f);
            cs.setPosition(centerPos);
            cs.setFillColor(sf::Color::Green);

            window.draw(cs);

            window.display();
        }

        renderCounter++;
    } while (!quit);

    return 0;
}
