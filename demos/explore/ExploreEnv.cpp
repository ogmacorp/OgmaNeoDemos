#include "ExploreEnv.h"

#include <assert.h>

#include <iostream>

const float pi = 3.141592f;

const float fov = pi * 0.7f;
const int scanRays = 30;
const float angleSpeed = 8.0f;
const float accel = 80.0f;
const float deccel = 8.0f;
const float range = 20.0f;
const float depthStep = 0.2f;
const float radius = 0.4f;
//const float squashScale = 0.2f;

float angleWrap(float delta) {
    float offset = delta + pi;

    float m = offset - std::floor(offset / (2.0f * pi)) * (2.0f * pi); // Mod with sign of numerator

    return m - pi;
}

float ExploreEnv::getDepth(
    const sf::Vector2f &start,
    float angle
) const {
    sf::Vector2f dir(std::cos(angle), std::sin(angle));

    int numSteps = std::ceil(range / depthStep);

    for (int s = 1; s <= numSteps; s++) {
        sf::Vector2f checkPos = start + dir * (depthStep * s);

        int xi = std::floor(checkPos.x);
        int yi = std::floor(checkPos.y);

        if (xi < 0 || xi >= map.getSize().x || yi < 0 || yi >= map.getSize().y)
            return depthStep * s;

        sf::Color color = map.getPixel(sf::Vector2u(xi, yi));

        if (color == sf::Color::Black)
            return depthStep * s;
    }

    return range;
}

void ExploreEnv::init(
    const sf::Image &map,
    unsigned long seed
) {
    rng.seed(seed);

    this->map = map;
    mapTex = sf::Texture(map);

    reset();
}

void ExploreEnv::reset() {
    // Find possible spawn loagentions
    std::vector<sf::Vector2i> openPositions;
    
    for (int x = 0; x < map.getSize().x; x++)
        for (int y = 0; y < map.getSize().y; y++) {
            sf::Color color = map.getPixel(sf::Vector2u(x, y));

            if (color != sf::Color::Black)
                openPositions.push_back(sf::Vector2i(x, y));
        }

    std::uniform_int_distribution<int> openPositionsDist(0, openPositions.size() - 1);

    int agentSpawnIndex = openPositionsDist(rng);
    
    std::uniform_int_distribution<int> offsetDist(openPositions.size() / 3, openPositions.size() / 3 * 2);

    agent.pos = sf::Vector2f(openPositions[agentSpawnIndex].x + 0.5f, openPositions[agentSpawnIndex].y + 0.5f);

    agent.posVel = sf::Vector2f(0.0f, 0.0f);

    std::uniform_real_distribution<float> angleDist(0.0f, pi * 2.0f);

    agent.angle = angleDist(rng);

    agent.angleVel = 0.0f;

    done = false;
}

int ExploreEnv::obsSize() const {
    return scanRays + 5;
}

void ExploreEnv::getObs(
    std::vector<float> &obs
) const {
    if (obs.size() != obsSize())
        obs.resize(obsSize());

    int index = 0;

    float startAngle = agent.angle - fov * 0.5f;
    float endAngle = agent.angle + fov * 0.5f;

    float angleStep = (endAngle - startAngle) / std::max(1, scanRays - 1);

    for (int s = 0; s < scanRays; s++) {
        float angle = startAngle + s * angleStep;

        float depth = getDepth(agent.pos, angle);

        obs[index] = depth / range;
        index++;
    }

    assert(index == scanRays);

    obs[index] = agent.angle / (2.0f * pi);

    if (obs[index] < 0.0f)
        obs[index] += 1.0f;

    index++;

    //agentObs[index] = std::tanh(squashScale * agent.posVel.x) * 0.5f + 0.5f;
    //index++;
    //agentObs[index] = std::tanh(squashScale * agent.posVel.y) * 0.5f + 0.5f;
    //index++;

    // Position
    obs[index] = static_cast<float>(agent.pos.x) / static_cast<float>(map.getSize().x);
    index++;
    obs[index] = static_cast<float>(agent.pos.y) / static_cast<float>(map.getSize().y);
}

sf::Vector2f ExploreEnv::collide(
    Agent &agent
) {
    sf::Vector2f offset(0.0f, 0.0f);

    int xi = std::floor(agent.pos.x);
    int yi = std::floor(agent.pos.y);

    {
        int checkxi = xi - 1;
        int checkyi = yi;

        if (checkxi < 0 || map.getPixel(sf::Vector2u(checkxi, checkyi)) == sf::Color::Black) {
            if (agent.pos.x - radius < xi) {
                offset.x = xi + radius - agent.pos.x;

                agent.pos.x = xi + radius;
                agent.posVel.x = 0.0f;
            }
        }
    }

    {
        int checkxi = xi + 1;
        int checkyi = yi;

        if (checkxi >= map.getSize().x || map.getPixel(sf::Vector2u(checkxi, checkyi)) == sf::Color::Black) {
            if (agent.pos.x + radius > xi + 1) {
                offset.x = xi + 1 - radius - agent.pos.x;

                agent.pos.x = xi + 1 - radius;
                agent.posVel.x = 0.0f;
            }
        }
    }

    {
        int checkxi = xi;
        int checkyi = yi - 1;

        if (checkyi < 0 || map.getPixel(sf::Vector2u(checkxi, checkyi)) == sf::Color::Black) {
            if (agent.pos.y - radius < yi) {
                offset.y = yi + radius - agent.pos.y;

                agent.pos.y = yi + radius;
                agent.posVel.y = 0.0f;
            }
        }
    }

    {
        int checkxi = xi;
        int checkyi = yi + 1;

        if (checkyi >= map.getSize().y || map.getPixel(sf::Vector2u(checkxi, checkyi)) == sf::Color::Black) {
            if (agent.pos.y + radius > yi + 1) {
                offset.y = yi + 1 - radius - agent.pos.y;

                agent.pos.y = yi + 1 - radius;
                agent.posVel.y = 0.0f;
            }
        }
    }

    return offset;
}

void ExploreEnv::step(
    const std::vector<float> &actions,
    float dt
) {
    // Cat action
    agent.angleVel = (actions[2] * 2.0f - 1.0f) * angleSpeed;
    agent.angle += agent.angleVel * dt;

    agent.angle = std::fmod(agent.angle, 2.0f * pi);

    sf::Vector2f agentDir(std::cos(agent.angle), std::sin(agent.angle));
    sf::Vector2f agentStrafe(-agentDir.y, agentDir.x);

    agent.posVel += ((agentDir * (actions[0] * 2.0f - 1.0f) + agentStrafe * (actions[1] * 2.0f - 1.0f)) * accel - agent.posVel * deccel) * dt;
    agent.pos += agent.posVel * dt;

    // collision
    collide(agent);

    done = false;
}

void ExploreEnv::render(
    sf::RenderWindow &window
) {
    sf::Sprite s(mapTex);

    window.draw(s);

    sf::CircleShape cs;
    cs.setRadius(radius);
    cs.setOrigin(sf::Vector2f(radius, radius));

    cs.setPosition(agent.pos);
    cs.setFillColor(sf::Color::Magenta);

    window.draw(cs);

    // Scan lines
    {
        sf::VertexArray vs;
        vs.setPrimitiveType(sf::PrimitiveType::Lines);
        vs.resize(scanRays * 2);

        float startAngle = agent.angle - fov * 0.5f;
        float endAngle = agent.angle + fov * 0.5f;

        float angleStep = (endAngle - startAngle) / std::max(1, scanRays - 1);

        int index = 0;

        for (int s = 0; s < scanRays; s++) {
            float angle = startAngle + s * angleStep;

            sf::Vector2f dir(std::cos(angle), std::sin(angle));

            float depth = getDepth(agent.pos, angle);

            vs[index].position = agent.pos + dir * radius; 
            vs[index].color = sf::Color::Green;
            index++;
            vs[index].position = agent.pos + dir * depth; 
            vs[index].color = sf::Color::Green;
            index++;
        }

        window.draw(vs);
    }
}
