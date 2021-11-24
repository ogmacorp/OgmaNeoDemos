#include "CatMouseEnv.h"

#include <assert.h>

#include <iostream>

const float pi = 3.141592f;

const float fov = pi * 0.7f;
const int scanRays = 7;
const float angleSpeed = 8.0f;
const float accel = 80.0f;
const float deccel = 8.0f;
const float range = 12.0f;
const float depthStep = 0.2f;
const float radius = 0.4f;
//const float squashScale = 0.2f;

float angleWrap(float delta) {
    float offset = delta + pi;

    float m = offset - std::floor(offset / (2.0f * pi)) * (2.0f * pi); // Mod with sign of numerator

    return m - pi;
}

float CatMouseEnv::getDepth(
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

        sf::Color color = map.getPixel(xi, yi);

        if (color == sf::Color::Black)
            return depthStep * s;
    }

    return range;
}

void CatMouseEnv::init(
    const sf::Image &map,
    unsigned long seed
) {
    rng.seed(seed);

    this->map = map;
    mapTex.loadFromImage(map);

    reset();
}

void CatMouseEnv::reset() {
    // Find possible spawn locations
    std::vector<sf::Vector2i> openPositions;
    
    for (int x = 0; x < map.getSize().x; x++)
        for (int y = 0; y < map.getSize().y; y++) {
            sf::Color color = map.getPixel(x, y);

            if (color != sf::Color::Black)
                openPositions.push_back(sf::Vector2i(x, y));
        }

    std::uniform_int_distribution<int> openPositionsDist(0, openPositions.size() - 1);

    int catSpawnIndex = openPositionsDist(rng);
    
    std::uniform_int_distribution<int> offsetDist(openPositions.size() / 3, openPositions.size() / 3 * 2);

    int mouseSpawnIndex = (catSpawnIndex + offsetDist(rng)) % openPositions.size();
    
    cat.pos = sf::Vector2f(openPositions[catSpawnIndex].x + 0.5f, openPositions[catSpawnIndex].y + 0.5f);
    mouse.pos = sf::Vector2f(openPositions[mouseSpawnIndex].x + 0.5f, openPositions[mouseSpawnIndex].y + 0.5f);

    cat.posVel = sf::Vector2f(0.0f, 0.0f);
    mouse.posVel = sf::Vector2f(0.0f, 0.0f);

    std::uniform_real_distribution<float> angleDist(0.0f, pi * 2.0f);

    cat.angle = angleDist(rng);
    mouse.angle = angleDist(rng);

    cat.angleVel = 0.0f;
    mouse.angleVel = 0.0f;

    done = false;

    sf::Vector2f posDelta = cat.pos - mouse.pos;

    distance = std::sqrt(posDelta.x * posDelta.x + posDelta.y * posDelta.y);
}

int CatMouseEnv::obsSize() const {
    return scanRays + 5;
}

void CatMouseEnv::getObs(
    std::vector<float> &catObs,
    std::vector<float> &mouseObs,
    float &catVisual,
    float &mouseVisual
) const {
    {
        if (catObs.size() != obsSize())
            catObs.resize(obsSize());

        int index = 0;

        float startAngle = cat.angle - fov * 0.5f;
        float endAngle = cat.angle + fov * 0.5f;

        float angleStep = (endAngle - startAngle) / std::max(1, scanRays - 1);

        for (int s = 0; s < scanRays; s++) {
            float angle = startAngle + s * angleStep;

            float depth = getDepth(cat.pos, angle);

            catObs[index] = depth / range;
            index++;
        }

        assert(index == scanRays);

        catObs[index] = cat.angle / (2.0f * pi);

        if (catObs[index] < 0.0f)
            catObs[index] += 1.0f;

        index++;

        //catObs[index] = std::tanh(squashScale * cat.posVel.x) * 0.5f + 0.5f;
        //index++;
        //catObs[index] = std::tanh(squashScale * cat.posVel.y) * 0.5f + 0.5f;
        //index++;

        // To mouse vector
        sf::Vector2f dirSense(0.0f, 0.0f);
        sf::Vector2f toVec = mouse.pos - cat.pos;
        float toDist = std::sqrt(toVec.x * toVec.x + toVec.y * toVec.y);

        sf::Vector2f forward(std::cos(cat.angle), std::sin(cat.angle));

        float toDot = forward.x * toVec.x / std::max(0.0001f, toDist) + forward.y * toVec.y / std::max(0.0001f, toDist);

        if (std::abs(std::acos(toDot)) < fov * 0.5f) {
            float toAngle = std::atan2(toVec.y, toVec.x);
            float depth = getDepth(cat.pos, toAngle);

            if (depth >= toDist) {
                float deltaAngle = toAngle - cat.angle;

                deltaAngle = angleWrap(deltaAngle);

                dirSense = sf::Vector2f(std::cos(deltaAngle), std::sin(deltaAngle));
            }
        }

        if (dirSense == sf::Vector2f(0.0f, 0.0f))
            catVisual = 0.0f;
        else
            catVisual = std::max(0.0f, range - toDist) / range;

        catObs[index] = dirSense.x * 0.5f + 0.5f;
        index++;
        catObs[index] = dirSense.y * 0.5f + 0.5f;
        index++;

        // Position
        catObs[index] = static_cast<float>(cat.pos.x) / static_cast<float>(map.getSize().x);
        index++;
        catObs[index] = static_cast<float>(cat.pos.y) / static_cast<float>(map.getSize().y);
    }

    {
        if (mouseObs.size() != obsSize())
            mouseObs.resize(obsSize());

        int index = 0;

        float startAngle = mouse.angle - fov * 0.5f;
        float endAngle = mouse.angle + fov * 0.5f;

        float angleStep = (endAngle - startAngle) / std::max(1, scanRays - 1);

        for (int s = 0; s < scanRays; s++) {
            float angle = startAngle + s * angleStep;

            float depth = getDepth(mouse.pos, angle);

            mouseObs[index] = depth / range;
            index++;
        }

        assert(index == scanRays);

        mouseObs[index] = mouse.angle / (2.0f * pi);

        if (mouseObs[index] < 0.0f)
            mouseObs[index] += 1.0f;

        index++;

        //mouseObs[index] = std::tanh(squashScale * mouse.posVel.x) * 0.5f + 0.5f;
        //index++;
        //mouseObs[index] = std::tanh(squashScale * mouse.posVel.y) * 0.5f + 0.5f;
        //index++;

        // To mouse vector
        sf::Vector2f dirSense(0.0f, 0.0f);
        sf::Vector2f toVec = cat.pos - mouse.pos;
        float toDist = std::sqrt(toVec.x * toVec.x + toVec.y * toVec.y);

        sf::Vector2f forward(std::cos(mouse.angle), std::sin(mouse.angle));

        float toDot = std::acos(forward.x * toVec.x / std::max(0.0001f, toDist) + forward.y * toVec.y / std::max(0.0001f, toDist));

        if (std::abs(toDot) < fov * 0.5f) {
            float toAngle = std::atan2(toVec.y, toVec.x);
            float depth = getDepth(mouse.pos, toAngle);

            if (depth >= toDist) {
                float deltaAngle = toAngle - mouse.angle;

                deltaAngle = angleWrap(deltaAngle);

                dirSense = sf::Vector2f(std::cos(deltaAngle), std::sin(deltaAngle));
            }
        }

        if (dirSense == sf::Vector2f(0.0f, 0.0f))
            mouseVisual = 0.0f;
        else
            mouseVisual = std::max(0.0f, range - toDist) / range;

        mouseObs[index] = dirSense.x * 0.5f + 0.5f;
        index++;
        mouseObs[index] = dirSense.y * 0.5f + 0.5f;
        index++;

        // Position
        mouseObs[index] = static_cast<float>(mouse.pos.x) / static_cast<float>(map.getSize().x);
        index++;
        mouseObs[index] = static_cast<float>(mouse.pos.y) / static_cast<float>(map.getSize().y);
    }
}

sf::Vector2f CatMouseEnv::collide(
    Agent &agent
) {
    sf::Vector2f offset(0.0f, 0.0f);

    int xi = std::floor(agent.pos.x);
    int yi = std::floor(agent.pos.y);

    {
        int checkxi = xi - 1;
        int checkyi = yi;

        if (checkxi < 0 || map.getPixel(checkxi, checkyi) == sf::Color::Black) {
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

        if (checkxi >= map.getSize().x || map.getPixel(checkxi, checkyi) == sf::Color::Black) {
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

        if (checkyi < 0 || map.getPixel(checkxi, checkyi) == sf::Color::Black) {
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

        if (checkyi >= map.getSize().y || map.getPixel(checkxi, checkyi) == sf::Color::Black) {
            if (agent.pos.y + radius > yi + 1) {
                offset.y = yi + 1 - radius - agent.pos.y;

                agent.pos.y = yi + 1 - radius;
                agent.posVel.y = 0.0f;
            }
        }
    }

    return offset;
}

void CatMouseEnv::step(
    const std::vector<float> &catActions,
    const std::vector<float> &mouseActions,
    float dt
) {
    // Cat action
    cat.angleVel = (catActions[2] * 2.0f - 1.0f) * angleSpeed;
    cat.angle += cat.angleVel * dt;

    cat.angle = std::fmod(cat.angle, 2.0f * pi);

    sf::Vector2f catDir(std::cos(cat.angle), std::sin(cat.angle));
    sf::Vector2f catStrafe(-catDir.y, catDir.x);

    cat.posVel += ((catDir * (catActions[0] * 2.0f - 1.0f) + catStrafe * (catActions[1] * 2.0f - 1.0f)) * accel - cat.posVel * deccel) * dt;
    cat.pos += cat.posVel * dt;

    // Mouse action
    mouse.angleVel = (mouseActions[2] * 2.0f - 1.0f) * angleSpeed;
    mouse.angle += mouse.angleVel * dt;

    mouse.angle = std::fmod(mouse.angle, 2.0f * pi);

    sf::Vector2f mouseDir(std::cos(mouse.angle), std::sin(mouse.angle));
    sf::Vector2f mouseStrafe(-mouseDir.y, mouseDir.x);

    mouse.posVel += ((mouseDir * (mouseActions[0] * 2.0f - 1.0f) + mouseStrafe * (mouseActions[1] * 2.0f - 1.0f)) * accel - mouse.posVel * deccel) * dt;
    mouse.pos += mouse.posVel * dt;

    // Cat collision
    collide(cat);
    collide(mouse);

    sf::Vector2f posDelta = cat.pos - mouse.pos;

    distance = std::sqrt(posDelta.x * posDelta.x + posDelta.y * posDelta.y);

    if (distance < 2.0f * radius)
        done = true;
}

void CatMouseEnv::render(
    sf::RenderWindow &window
) {
    sf::Sprite s;

    s.setTexture(mapTex);

    window.draw(s);

    sf::CircleShape cs;
    cs.setRadius(radius);
    cs.setOrigin(radius, radius);

    cs.setPosition(cat.pos);
    cs.setFillColor(sf::Color::Magenta);

    window.draw(cs);

    cs.setPosition(mouse.pos);
    cs.setFillColor(sf::Color::Cyan);

    window.draw(cs);

    // Scan lines
    {
        sf::VertexArray vs;
        vs.setPrimitiveType(sf::Lines);
        vs.resize(scanRays * 2);

        float startAngle = cat.angle - fov * 0.5f;
        float endAngle = cat.angle + fov * 0.5f;

        float angleStep = (endAngle - startAngle) / std::max(1, scanRays - 1);

        int index = 0;

        for (int s = 0; s < scanRays; s++) {
            float angle = startAngle + s * angleStep;

            sf::Vector2f dir(std::cos(angle), std::sin(angle));

            float depth = getDepth(cat.pos, angle);

            vs[index].position = cat.pos + dir * radius; 
            vs[index].color = sf::Color::Green;
            index++;
            vs[index].position = cat.pos + dir * depth; 
            vs[index].color = sf::Color::Green;
            index++;
        }

        window.draw(vs);
    }

    {
        sf::VertexArray vs;
        vs.setPrimitiveType(sf::Lines);
        vs.resize(scanRays * 2);

        float startAngle = mouse.angle - fov * 0.5f;
        float endAngle = mouse.angle + fov * 0.5f;

        float angleStep = (endAngle - startAngle) / std::max(1, scanRays - 1);

        int index = 0;

        for (int s = 0; s < scanRays; s++) {
            float angle = startAngle + s * angleStep;

            sf::Vector2f dir(std::cos(angle), std::sin(angle));

            float depth = getDepth(mouse.pos, angle);

            vs[index].position = mouse.pos + dir * radius; 
            vs[index].color = sf::Color::Green;
            index++;
            vs[index].position = mouse.pos + dir * depth; 
            vs[index].color = sf::Color::Green;
            index++;
        }

        window.draw(vs);
    }
}
