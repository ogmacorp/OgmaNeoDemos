#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

const std::vector<std::vector<int>> graph = {
    { 0, 1, 2, 6 },
    { 1, 0, 2, 3 },
    { 2, 3, 4, 0 },
    { 3, 1, 2, 4 },
    { 4, 0, 1, 5 },
    { 5, 2, 3, 4, 6 },
    { 6, 3, 5 }
};

const int numStates = graph.size();
const int numActions = 5;

int main() {
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::normal_distribution<float> nDist(0.0f, 1.0f);
    std::uniform_int_distribution<int> actionDist(0, numActions - 1);

    std::vector<sf::Vector2f> positions(graph.size());

    for (int i = 0; i < positions.size(); i++) {
        float angle = static_cast<float>(i) / static_cast<float>(positions.size()) * 2.0f * M_PI;
        positions[i] = sf::Vector2f(std::cos(angle) * 100.0f, std::sin(angle) * 100.0f);
    }

    std::vector<float> qTable(numStates * numStates * numActions, 0.0f);
    int currentP = 0;
    int prevP = 0;
    int goalP = 0;

    const float dt = 0.017f;
    const float zoomRate = 0.4f;
    const float viewInterpolateRate = 20.0f;

    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Gen Demo", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    sf::View view = window.getDefaultView();
    view.setCenter(0.0f, 0.0f);
    sf::View newView = view;

    sf::Font font;
    font.loadFromFile("resources/Hack-Regular.ttf");

    bool quit = false;

    bool spacePrev = false;
    bool mPrev = false;

    bool moveMode = true;

    int actIndex = 0;
    
    float rate = 0.5f;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            if (window.hasFocus()) {
                switch (event.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                case sf::Event::MouseWheelMoved:
                    int dWheel = event.mouseWheel.delta;

                    newView = view;

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta0 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    newView.setSize(view.getSize() + newView.getSize() * (-zoomRate * dWheel));

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta1 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    window.setView(view);

                    newView.setCenter(view.getCenter() + mouseZoomDelta0 - mouseZoomDelta1);

                    break;
                }
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            for (int p = 0; p < graph.size(); p++) {
                if (sf::Keyboard::isKeyPressed(static_cast<sf::Keyboard::Key>(static_cast<int>(sf::Keyboard::Num0) + p)))
                    goalP = p;
            }

            bool space = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);

            if (space && !spacePrev || sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
                // Advance simulation
                int maxIndex = 0;
                float maxQ = -999999.0f;

                for (int i = 0; i < numActions; i++) {
                    float q = qTable[i + numActions * (goalP + numStates * currentP)];

                    if (q > maxQ) {
                        maxQ = q;
                        maxIndex = i;
                    }
                }

                // Learn
                for (int g = 0; g < positions.size(); g++) {
                    int qiPrev = actIndex + numActions * (g + numStates * prevP);

                    float reward = (currentP == g);

                    float target = std::max(reward, 0.9f * maxQ);

                    qTable[qiPrev] += rate * (target - qTable[qiPrev]);
                }

                rate *= 0.999f;

                actIndex = maxIndex;

                if (moveMode)
                    actIndex = actionDist(rng);

                prevP = currentP;

                if (actIndex < graph[currentP].size()) {
                    currentP = graph[currentP][actIndex];
                }
            }

            spacePrev = space;

            bool m = sf::Keyboard::isKeyPressed(sf::Keyboard::M);

            if (m && !mPrev) {
                moveMode = !moveMode;

                std::cout << "Move mode: " << moveMode << std::endl;
            }

            mPrev = m;
        }

        view.setCenter(view.getCenter() + (newView.getCenter() - view.getCenter()) * viewInterpolateRate * dt);
        view.setSize(view.getSize() + (newView.getSize() - view.getSize()) * viewInterpolateRate * dt);

        window.setView(view);

        window.clear(sf::Color::Black);

        // Draw edges
        sf::VertexArray lines;
        lines.setPrimitiveType(sf::Lines);

        for (int p = 0; p < positions.size(); p++) {
            for (int e = 0; e < graph[p].size(); e++) {
                int other = graph[p][e];

                bool hasReverse = false;

                for (int oe = 0; oe < graph[other].size(); oe++) {
                    if (graph[other][oe] == p) {
                        hasReverse = true;

                        break;
                    }
                }

                sf::Vertex start;

                start.position = positions[p];
                start.color = hasReverse ? sf::Color::Blue : sf::Color::Red;

                sf::Vertex end;

                end.position = positions[other];
                end.color = sf::Color::Blue;

                lines.append(start);
                lines.append(end);
            }
        }

        window.draw(lines);

        // Render nodes
        for (int p = 0; p < positions.size(); p++) {
            sf::CircleShape cs;
            cs.setPosition(positions[p]);
            cs.setRadius(4.0f);
            cs.setOrigin(4.0f, 4.0f);
            cs.setFillColor(p == currentP ? sf::Color::Green : sf::Color::Yellow);
            window.draw(cs);

            sf::Text t;
            t.setFont(font);
            t.setCharacterSize(16);
            t.setPosition(cs.getPosition() + sf::Vector2f(5.0f, 5.0f));
            t.setString(std::to_string(p));
            window.draw(t);
        }

        sf::View oldView = window.getView();

        window.setView(window.getDefaultView());

        sf::Text t;
        t.setFont(font);
        t.setCharacterSize(32);
        t.setPosition(window.getSize().x - 20.0f, 10.0f);
        t.setString(std::to_string(goalP));
        window.draw(t);

        // Render matrix
        t.setCharacterSize(16);

        for (int y = 0; y < positions.size(); y++) {
            for (int x = 0; x < positions.size(); x++) {
                int maxIndex = 0;
                float maxQ = -999999.0f;

                for (int i = 0; i < numActions; i++) {
                    float q = qTable[i + numActions * (y + numStates * x)];

                    if (q > maxQ) {
                        maxQ = q;
                        maxIndex = i;
                    }
                }

                t.setPosition(10.0f + x * 20.0f, 10.0f + y * 20.0f);
                
                if (maxIndex >= graph[x].size())
                    t.setString(std::to_string(x));
                else
                    t.setString(std::to_string(graph[x][maxIndex]));

                window.draw(t);
            }
        }
            
        window.setView(oldView);

        window.display();
    } while (!quit);

    return 0;
}

