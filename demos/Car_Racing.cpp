#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/Helpers.h>
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

struct Car {
    sf::Vector2f position;
    
    float speed;

    float rotation;

    Car()
        : position(0.0f, 0.0f),
        speed(0.0f),
        rotation(0.0f)
    {}
};

float magnitude(const sf::Vector2f &v) {
    float d = v.x * v.x + v.y * v.y;

    return std::sqrt(d);
}

// Calculate the maximal distance from the begin point "start" to the road boundary in the direction
// defined by a ray from "start" to "end" point
float rayCast(const sf::Image &mask, const sf::Vector2f &start, const sf::Vector2f &end) {
    const float castIncrement = 2.0f;

    sf::Vector2f point = start;

    int steps = magnitude(end - start) / castIncrement;

    sf::Vector2f dir = end - start;

    dir /= std::max(0.00001f, magnitude(dir));

    float d = 0.0f;

    for (int i = 0; i < steps; i++) {
        sf::Color c = mask.getPixel(point.x, point.y);

        if (c == sf::Color::White)
            return d;

        point += dir * castIncrement;
        d += castIncrement;
    }

    return d;
}

// checkPoints is a list of points in image, which have c.a != 0, the 1st point is starting and the last is for the routing end
void getCheckpoints(const sf::Image &checkpointsImg, std::vector<sf::Vector2f> &checkpoints) {
    for (int x = 0; x < checkpointsImg.getSize().x; x++)
        for (int y = 0; y < checkpointsImg.getSize().y; y++) {
            sf::Color c = checkpointsImg.getPixel(x, y);

            if (c.a != 0) {
                if (c.r >= checkpoints.size())
                    checkpoints.resize(c.r + 1);

                checkpoints[c.r] = sf::Vector2f(x, y);
            }
        }
}

int main() {
    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::RenderWindow window;

    window.create(sf::VideoMode(1000, 956), "Racing", sf::Style::Default);

    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    std::string encFileName = "Car_Racing.oenc";
    std::string hFileName = "Car_Racing.ohr";

    int numSensors = 16;
    int rootNumSensors = std::ceil(std::sqrt(numSensors));
    int sensorResolution = 32;
    int steerResolution = 9;

    // --------------------------- Create the Hierarchy ---------------------------

    // Create hierarchy
    setNumThreads(8);

    Array<Hierarchy::LayerDesc> lds(2);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(4, 4, 16);
        lds[i].errorSize = Int3(4, 4, 16);

        lds[i].hRadius = 2;
        lds[i].eRadius = 2;
        lds[i].dRadius = 2;
        lds[i].bRadius = 2;
    }

    // Two IODescs, for sensors and for actions
    // types none and action (no prediction and reinforcement learning)
    Array<Hierarchy::IODesc> ioDescs(2);
    ioDescs[0] = Hierarchy::IODesc(Int3(rootNumSensors, rootNumSensors, sensorResolution), IOType::prediction, 4, 4, 2, 2, 32);
    ioDescs[1] = Hierarchy::IODesc(Int3(1, 1, steerResolution), IOType::action, 2, 2, 2, 2, 32);

    Hierarchy h;
    h.initRandom(ioDescs, lds);
    
    //CustomStreamReader reader;
    //reader.ins.open(hFileName.c_str(), std::ios::out | std::ios::binary);
    //h.read(reader);

    // -------------------------- Game Resources --------------------------

    sf::Texture backgroundTex;
    backgroundTex.loadFromFile("resources/racingBackground.png");

    sf::Texture foregroundTex;
    foregroundTex.loadFromFile("resources/racingForeground.png");

    sf::Image collisionImg;
    collisionImg.loadFromFile("resources/racingCollision.png"); // Driving route to navigate (racing path with defined street boundaries / lane)

    sf::Image checkpointsImg;
    checkpointsImg.loadFromFile("resources/racingCheckpoints.png"); // Points of the lane middle

    std::vector<sf::Vector2f> checkpoints;

    getCheckpoints(checkpointsImg, checkpoints);
    
    sf::Texture carTex;
    carTex.loadFromFile("resources/racingCar.png");

    Car car;

    // Reset
    car.position = checkpoints[0];
    car.rotation = std::atan2(checkpoints[1].y - checkpoints[0].y, checkpoints[1].x - checkpoints[0].x);

    int curCheckpoint = 0;

    int laps = 0;

    float prevDistance = 0.0f;

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    bool speedMode = false;
    int tPressedPrev = false;
    bool sPressedPrev = false;

    sf::Clock clock;

    float dt = 0.017f;

    sf::View view;

    view.setCenter(250, 239);

    view.zoom(0.5f);

    window.setView(view);

    sf::Texture whitenedTex;

    // Used for speed mode to render slower
    int renderCounter = 0;

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

            sPressedPrev =  sPressed;
        }

        window.clear();

        const float maxSpeed = 30.0f;
        const float accel = 0.1f;
        const float spinRate = 0.16f;

        // Get action prediction
        int actionIndex = h.getPredictionCIs(1)[0];

        if (dist01(rng) < 0.1f) {
            std::uniform_int_distribution<int> steerDist(0, steerResolution - 1);
            actionIndex = steerDist(rng);
        }

        // Tranformation action (column index) -> steering value
        float steer = static_cast<float>(actionIndex) / static_cast<float>(steerResolution - 1) * 2.0f - 1.0f;

        // Physics update
        sf::Vector2f prevPosition = car.position;

        car.position += sf::Vector2f(std::cos(car.rotation), std::sin(car.rotation)) * car.speed;
        car.speed *= 0.95f;

        car.speed = std::min(maxSpeed, std::max(-maxSpeed, car.speed + accel));// * (actionIndex * 0.5f + 0.5f)
        car.rotation = std::fmod(car.rotation + steer * spinRate, 3.141596f * 2.0f);

        sf::Color curColor = collisionImg.getPixel(car.position.x, car.position.y);

        bool reset = false;

        if (curColor == sf::Color::White || sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
            // Reset
            car.position = checkpoints[0];
            car.speed = 0.0f;
            car.rotation = std::atan2(checkpoints[1].y - checkpoints[0].y, checkpoints[1].x - checkpoints[0].x);
            curCheckpoint = 0;
            laps = 0;
            prevDistance = 0.0f;

            reset = true;
        }

        sf::Vector2f vec = checkpoints[(curCheckpoint + 1) % static_cast<int>(checkpoints.size())] - checkpoints[curCheckpoint];

        // Project position onto vec
        sf::Vector2f relPos = car.position - checkpoints[curCheckpoint];

        sf::Vector2f proj = static_cast<float>((relPos.x * vec.x + relPos.y * vec.y) / std::pow(magnitude(vec), 2)) * vec;

        float addDist = magnitude(proj) * ((vec.x * proj.x + vec.y * proj.y) > 0.0f ? 1.0f : -1.0f);

        // If past checkpoint (before or after current segment)
        if (addDist >= magnitude(vec)) {
            curCheckpoint = (curCheckpoint + 1) % static_cast<int>(checkpoints.size());

            if (curCheckpoint == 0)
                laps++;
        }
        else if (addDist < 0.0f) {
            curCheckpoint = (curCheckpoint - 1) % static_cast<int>(checkpoints.size());

            if (curCheckpoint < 0) {
                curCheckpoint += checkpoints.size();
                laps--;
            }
        }

        // Re-do projection in case checkpoint changed
        vec = checkpoints[(curCheckpoint + 1) % static_cast<int>(checkpoints.size())] - checkpoints[curCheckpoint];

        // Project position onto vec
        relPos = car.position - checkpoints[curCheckpoint];

        proj = static_cast<float>((relPos.x * vec.x + relPos.y * vec.y) / std::pow(magnitude(vec), 2)) * vec;

        addDist = magnitude(proj) * ((vec.x * proj.x + vec.y * proj.y) > 0.0f ? 1.0f : -1.0f);

        // Car distance
        float distance = 0.0f;

        // Count up to current checkpoint
        for (int i = 0; i < curCheckpoint; i++)
            distance += magnitude(checkpoints[(i + 1) % static_cast<int>(checkpoints.size())] - checkpoints[i]);

        // Add laps
        float totalDist = 0.0f;

        for (int i = 0; i < checkpoints.size(); i++)
            totalDist += magnitude(checkpoints[(i + 1) % static_cast<int>(checkpoints.size())] - checkpoints[i]);

        distance += laps * totalDist;

        distance += addDist;

        float deltaDistance = reset ? 0.0f : distance - prevDistance;

        prevDistance = distance;

        sf::Vector2f carDir(car.position - prevPosition);

        carDir /= std::max(0.00001f, magnitude(carDir));

        sf::Vector2f trackDir = vec / magnitude(vec);
        sf::Vector2f trackPerp(-trackDir.y, trackDir.x);

        // Define reward as orientation difference between 2 vectors: car direction and road direction
        // because they are already normalized, so that cos(alpha) = carDir.x * trackDir.x + carDir.y * trackDir.y
        // this reward will be higher by higher speed
        float reward = 0.1f * std::abs(car.speed) * (carDir.x * trackDir.x + carDir.y * trackDir.y) + (reset ? -50.0f : 0.0f);

        // Sensors as laser multiple beams (number of beams = numSensors) for checking collision
        std::vector<float> sensors(numSensors);

        const float sensorAngle = 0.16f;
        const float sensorRange = 70.0f;

        for (int s = 0; s < sensors.size(); s++) {
            float d = sensorAngle * (s - sensors.size() * 0.5f) + car.rotation;

            sf::Vector2f dir = sf::Vector2f(std::cos(d), std::sin(d));

            sf::Vector2f begin = car.position;
            sf::Vector2f end = car.position + dir * sensorRange;

            float v = rayCast(collisionImg, begin, end);

            sensors[s] = v / sensorRange;
        }

        if (!speedMode || renderCounter >= 100) {
            renderCounter = 0;

            sf::Sprite backgroundS;
            backgroundS.setTexture(backgroundTex);

            window.draw(backgroundS);

            sf::VertexArray va;

            va.setPrimitiveType(sf::Lines);
            va.resize(sensors.size() * 2);

            for (int s = 0; s < sensors.size(); s++) {
                float d = sensorAngle * (s - sensors.size() * 0.5f) + car.rotation;

                sf::Vector2f dir = sf::Vector2f(std::cos(d), std::sin(d));

                va[s * 2 + 0] = car.position;
                va[s * 2 + 1] = car.position + dir * sensors[s] * sensorRange;
            }

            window.draw(va);

            sf::Sprite carS;
            carS.setTexture(carTex);

            carS.setOrigin(carTex.getSize().x * 0.5f, carTex.getSize().y * 0.5f);
            carS.setPosition(car.position);
            carS.setRotation(car.rotation * 180.0f / 3.141596f + 90.0f);
            carS.setScale(0.5f, 0.5f);

            window.draw(carS);

            // Draw checkpoints if desired
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::C)) {
                sf::VertexArray ar;

                ar.setPrimitiveType(sf::LinesStrip);
                ar.resize(checkpoints.size() + 1);

                for (int i = 0; i < checkpoints.size(); i++)
                    ar[i] = sf::Vertex(checkpoints[i], sf::Color::Green);

                ar[checkpoints.size()] = sf::Vertex(checkpoints.front(), sf::Color::Green);

                window.draw(ar);
            }

            sf::Sprite foregroundS;
            foregroundS.setTexture(foregroundTex);
            window.draw(foregroundS);

            window.display();
        }

        renderCounter++;

        Array<const IntBuffer*> inputCIs(ioDescs.size());

        IntBuffer sensorCIs(rootNumSensors * rootNumSensors, 0);

        for (int i = 0; i < sensors.size(); i++)
            sensorCIs[i] = static_cast<int>(std::min(1.0f, std::max(0.0f, sensors[i])) * (sensorResolution - 1) + 0.5f);

        inputCIs[0] = &sensorCIs;

        IntBuffer actionCIs(1);
        actionCIs[0] = actionIndex;

        inputCIs[1] = &actionCIs;

        h.step(inputCIs, true, reward);

    } while (!quit);

    return 0;
}
