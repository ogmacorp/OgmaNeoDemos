#pragma once

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <vector>
#include <tuple>
#include <random>

class CatMouseEnv {
public:
    struct Agent {
        sf::Vector2f pos;
        sf::Vector2f posVel;
        float angle;
        float angleVel;

        Agent()
        :
        pos(0.0f, 0.0f),
        posVel(0.0f, 0.0f),
        angle(0.0f),
        angleVel(0.0f)
        {}
    };

private:
    std::mt19937 rng;

    sf::Image map;
    sf::Texture mapTex;

    Agent cat;
    Agent mouse;

    bool done;
    float distance;

    float getDepth(
        const sf::Vector2f &start,
        float angle
    ) const;

    sf::Vector2f collide(
        Agent &agent
    );

public:
    void init(
        const sf::Image &map,
        unsigned long seed = 1234
    );

    int obsSize() const;
    int actionsSize() const {
        return 3;
    }

    void reset();

    bool getDone() const {
        return done;
    }

    float getDistance() const {
        return distance;
    }

    void getObs(
        std::vector<float> &catObs,
        std::vector<float> &mouseObs,
        float &catVisual,
        float &mouseVisual
    ) const;

    void step(
        const std::vector<float> &catActions,
        const std::vector<float> &mouseActions,
        float dt
    );

    void render(
        sf::RenderWindow &window
    );
};
