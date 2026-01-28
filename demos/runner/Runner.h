// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <box2d/box2d.h>

#include <memory>
#include <cmath>

class Runner {
public:
    struct LimbSegmentDesc {
        float relativeAngle;
        float thickness, length;
        float minAngle, maxAngle;
        float maxTorque;
        float maxSpeed;
        float density;
        float friction;
        float restitution;
        bool motorEnabled;

        LimbSegmentDesc()
        :
        relativeAngle(0.0f),
        thickness(0.03f), length(0.125f),
        minAngle(-1.1f), maxAngle(1.1f),
        maxTorque(0.2f),
        maxSpeed(50.0f),
        density(2.0f),
        friction(5.0f),
        restitution(0.001f),
        motorEnabled(true)
        {}
    };

    struct LimbSegment {
        b2ShapeId bodyShape;
        b2BodyId body;
        b2JointId joint;

        float maxSpeed;
        float minAngle;
        float maxAngle;
    };

    struct Limb {
        std::vector<LimbSegment> segments;

        void create(b2WorldId world, const std::vector<LimbSegmentDesc> &descs, b2BodyId attachBody, const b2Vec2 &localAttachPoint, std::uint16_t categoryBits, std::uint16_t maskBits);
        void remove(b2WorldId world);
    };
private:
    bool initialized;
    b2WorldId world;
    std::vector<float> whiskerResults;
    b2Vec2 lVelPrev;
    float rVelPrev;

    std::vector<float> positions;
    std::vector<float> speeds;
    
public:

    static sf::Color mulColors(const sf::Color &c1, const sf::Color &c2) {
        const float byteInv = 1.0f / 255.0f;

        return sf::Color(c1.r * c2.r * byteInv,
            c1.g * c2.g * byteInv,
            c1.b * c2.b * byteInv);
    }

    b2ShapeId bodyShape;
    b2BodyId body;

    Limb leftBackLimb;
    Limb leftFrontLimb;
    Limb rightBackLimb;
    Limb rightFrontLimb;

    Runner()
        : initialized(false)
    {}

    void createDefault(b2WorldId world, const b2Vec2 &position, float angle, int layer);
    void destroy();

    ~Runner();

    void renderDefault(sf::RenderTarget &rt, const sf::Color &color, float metersToPixels);

    void getStateVector(std::vector<float> &state);
    void motorUpdate(const std::vector<float> &actions, float propPos = 0.5f, float propSpeed = 0.4f);

    bool infrontOfWall() const {
        return whiskerResults[0] < 0.01f;
    }
};
