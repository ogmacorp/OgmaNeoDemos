#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/hierarchy.h>
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
    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(sf::Vector2u(1024, 1024)), "Pusher", sf::Style::Default);

    window.setFramerateLimit(20);

    std::string encFileName = "pusher.oenc";
    std::string hFileName = "pusher.ohr";

    // --------------------------- Create the Hierarchy ---------------------------

    // Create hierarchy
    set_num_threads(8);

    Array<Hierarchy::Layer_Desc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(7, 7, 128);
    }

    int sensorRes = 16;
    int actionRes = 5;

    Array<Hierarchy::IO_Desc> ioDescs(2);
    ioDescs[0] = Hierarchy::IO_Desc(Int3(2, 2, sensorRes), IO_Type::prediction, 4, 2, 3);
    ioDescs[1] = Hierarchy::IO_Desc(Int3(1, 2, actionRes), IO_Type::action, 4, 0, 3);

    Hierarchy h;
    h.init_random(ioDescs, lds);

    h.params.ios[1].importance = 0.0f;

    S32_Array actionCIs = h.get_prediction_cis(1);

    //CustomStreamReader reader;
    //reader.ins.open(hFileName.c_str(), std::ios::out | std::ios::binary);
    //h.read(reader);

    // -------------------------- Game Resources --------------------------

    sf::VertexArray arrow(sf::PrimitiveType::Lines, 6);

    arrow[0].position = sf::Vector2f(0.0f, 0.0f);
    arrow[1].position = sf::Vector2f(1.0f, 0.0f);
    arrow[2].position = sf::Vector2f(1.0f, 0.0f);
    arrow[3].position = sf::Vector2f(0.8f, -0.2f);
    arrow[4].position = sf::Vector2f(1.0f, 0.0f);
    arrow[5].position = sf::Vector2f(0.8f, 0.2f);

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    bool speedMode = false;
    bool tPressedPrev = false;
    bool sPressedPrev = false;
    bool lPressedPrev = false;

    sf::Clock clock;

    float dt = 0.017f;

    sf::View view;

    view.setCenter(sf::Vector2f(0.0f, 0.0f));

    view.setSize(sf::Vector2f(2.0f, 2.0f));

    window.setView(view);

    sf::Texture whitenedTex;

    int vwidth = 64;
    int vheight = 64;
    float vrate = 0.1f;
    float extra_exploration = 0.0f;

    std::vector<sf::Vector2f> vecs(vwidth * vheight, { 0.0f, 0.0f });

    for (int i = 0; i < vecs.size(); i++)
        vecs[i] = sf::Vector2f(dist01(rng) * 2.0f - 1.0f, dist01(rng) * 2.0f - 1.0f) * 0.001f;

    // Used for speed mode to render slower
    int renderCounter = 0;

    float average_reward = 0.0f;

    float distPrev = -1.0f;
    float objectDistPrev = -1.0f;

    sf::Vector2f objectPos(0.3f, 0.3f);
    sf::Vector2f pusherPos(0.0f, 0.0f);

    sf::Vector2f targetPos(0.0f, 0.0f);

    float objectRad = 0.1f;
    float pusherRad = 0.1f;

    bool learnMode = true;

    long steps = 0;

    do {
        clock.restart();

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

            if (sPressed && !sPressedPrev) {
                CustomStreamWriter writer;
                writer.outs.open(hFileName.c_str(), std::ios::out | std::ios::binary);
                h.write(writer);
            }

            sPressedPrev = sPressed;

            bool lPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::Key::L);

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

        actionCIs = h.get_prediction_cis(1);

        // Exploration
        for (int i = 0; i < actionCIs.size(); i++) {
            if (dist01(rng) < extra_exploration) {
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

        sf::Vector2f pusherPosVec((pusherPos.x * 0.5f + 0.5f) * 0.9999f * vwidth, (pusherPos.y * 0.5f + 0.5f) * 0.9999f * vheight);

        int px = static_cast<int>(pusherPosVec.x);
        int py = static_cast<int>(pusherPosVec.y);
        int npx = std::ceil(pusherPosVec.x);
        int npy = std::ceil(pusherPosVec.y);

        float lx = pusherPosVec.x - px;
        float ly = pusherPosVec.y - py;

        vecs[py + vheight * px] += vrate * (1.0f - ly) * (1.0f - lx) * (delta - vecs[py + vheight * px]);

        if (npx < vwidth)
            vecs[py + vheight * npx] += vrate * (1.0f - ly) * lx * (delta - vecs[py + vheight * npx]);

        if (npy < vheight)
            vecs[npy + vheight * px] += vrate * ly * (1.0f - lx) * (delta - vecs[npy + vheight * px]);

        if (npx < vwidth && npy < vheight)
            vecs[npy + vheight * npx] += vrate * ly * lx * (delta - vecs[npy + vheight * npx]);

        float distToCenter = std::sqrt(objectPos.x * objectPos.x + objectPos.y * objectPos.y);

        sf::Vector2f objectDelta = objectPos - pusherPos;

        float distToObject = std::sqrt(objectDelta.x * objectDelta.x + objectDelta.y * objectDelta.y);

        if (distPrev == -1.0f)
            distPrev = distToCenter;

        if (objectDistPrev == -1.0f)
            objectDistPrev = distToObject;

        float reward = -5.0f * (distToCenter - distPrev) - 2.0f * (distToObject - objectDistPrev);

        distPrev = distToCenter;
        objectDistPrev = distToObject;

        bool outOfBounds = objectPos.x < -1.0f || objectPos.x > 1.0f || objectPos.y < -1.0f || objectPos.y > 1.0f;

        if (distToCenter < 0.08f || outOfBounds) {
            // Reset
            objectPos = sf::Vector2f(dist01(rng) * 2.0f - 1.0f, dist01(rng) * 2.0f - 1.0f) * 0.6f;

            reward = outOfBounds ? -0.5f : 100.0f;

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

        S32_Array sensorCIs(4);
        sensorCIs[0] = (pusherPos.x * 0.5f + 0.5f) * (sensorRes - 1) + 0.5f;
        sensorCIs[1] = (pusherPos.y * 0.5f + 0.5f) * (sensorRes - 1) + 0.5f;
        sensorCIs[2] = (objectDelta.x * 0.5f + 0.5f) * (sensorRes - 1) + 0.5f;
        sensorCIs[3] = (objectDelta.y * 0.5f + 0.5f) * (sensorRes - 1) + 0.5f;

        // clamp
        for (int i = 0; i < sensorCIs.size(); i++)
            sensorCIs[i] = std::min(sensorRes - 1, std::max(0, sensorCIs[i]));

        Array<S32_Array_View> inputCIs(2);
        inputCIs[0] = sensorCIs;
        inputCIs[1] = actionCIs;

        h.step(inputCIs, true, reward * 1.0f);

        //for (int i = 0; i < h.get_encoder(0).get_hidden_cis().size(); i++) {
        //    std::cout << h.get_encoder(0).get_hidden_cis()[i] << " ";
        //}
        //std::cout << std::endl;

        average_reward += 0.0001f * (reward - average_reward);

        if (!speedMode || renderCounter >= 300) {
            window.clear();

            renderCounter = 0;


            sf::RenderStates rs;
            rs.coordinateType = sf::CoordinateType::Normalized;
            rs.transform = sf::Transform::Identity;

            // draw vectors
            for (int x = 0; x < vwidth; x++)
                for (int y = 0; y < vheight; y++) {
                    rs.transform = sf::Transform::Identity;
                    rs.transform.translate(sf::Vector2f((x + 0.5f) / vwidth * 2.0f - 1.0f, (y + 0.5f) / vheight * 2.0f - 1.0f));

                    sf::Vector2f v = vecs[y + x * vwidth];

                    float angle = std::atan2(v.y, v.x);
                    float mag = v.length();

                    rs.transform.rotate(sf::radians(angle));
                    rs.transform.scale(sf::Vector2f(mag, mag) * 0.9f);

                    window.draw(arrow, rs);
                }
            
            sf::CircleShape cs;

            cs.setRadius(objectRad);
            cs.setOrigin(sf::Vector2f(objectRad, objectRad));
            cs.setPosition(objectPos);
            cs.setFillColor(sf::Color::Red);

            window.draw(cs);

            cs.setRadius(pusherRad);
            cs.setOrigin(sf::Vector2f(pusherRad, pusherRad));
            cs.setPosition(pusherPos);
            cs.setFillColor(sf::Color::Blue);

            window.draw(cs);

            cs.setRadius(0.01f);
            cs.setOrigin(sf::Vector2f(0.01f, 0.01f));
            cs.setPosition(sf::Vector2f(0.0f, 0.0f));
            cs.setFillColor(sf::Color::Green);

            window.draw(cs);

            window.display();
        }

        renderCounter++;

        steps++;

        if (steps % 10000 == 9999)
            std::cout << "Steps: " << steps << " Avg. Reward: " << average_reward << std::endl;
    } while (!quit);

    return 0;
}
