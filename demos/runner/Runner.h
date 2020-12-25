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

#include <Box2D/Box2D.h>

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
        minAngle(-0.6f), maxAngle(0.6f),
        maxTorque(300.0f),
        maxSpeed(4000.0f),
        density(2.0f),
        friction(2.0f),
        restitution(0.01f),
        motorEnabled(true)
        {}
    };

    struct LimbSegment {
        b2Body* body;
        b2PolygonShape bodyShape;
        b2RevoluteJoint* joint;

        float maxSpeed;
        float minAngle;
        float maxAngle;
    };

    struct Limb {
        std::vector<LimbSegment> segments;

        void create(b2World* world, const std::vector<LimbSegmentDesc> &descs, b2Body* attachBody, const b2Vec2 &localAttachPoint, uint16 categoryBits, uint16 maskBits);
        void remove(b2World* world);
    };
private:
    std::shared_ptr<b2World> world;

public:
    static sf::Color mulColors(const sf::Color &c1, const sf::Color &c2) {
        const float byteInv = 1.0f / 255.0f;

        return sf::Color(c1.r * c2.r * byteInv,
            c1.g * c2.g * byteInv,
            c1.b * c2.b * byteInv);
    }

    b2Body* body;
    b2PolygonShape bodyShape;

    Limb leftBackLimb;
    Limb leftFrontLimb;
    Limb rightBackLimb;
    Limb rightFrontLimb;

    Runner()
    :
    world(nullptr)
    {}

    ~Runner();

    void createDefault(const std::shared_ptr<b2World> &world, const b2Vec2 &position, float angle, int layer);

    void renderDefault(sf::RenderTarget &rt, const sf::Color &color, float metersToPixels);

    void getStateVector(std::vector<float> &state);
    void motorUpdate(const std::vector<float> &action, float interpolateFactor = 16.0f, float smoothIn = 0.0f, float minSmooth = 1.0f);
};
