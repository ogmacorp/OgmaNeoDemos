// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOSLICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Runner.h"

#include <iostream>

const float bodyWidth = 0.45f;
const float legInset = 0.075f;
const float bodyHeight = 0.1f;
const float bodyDensity = 2.5f;
const float bodyFriction = 1.0f;
const float bodyRestitution = 0.01f;
const int numWhiskers = 6;
const float whiskerLen = 1.5f;
const float whiskerSpread = 0.25f;

void Runner::Limb::create(b2WorldId world, const std::vector<LimbSegmentDesc> &descs, b2BodyId attachBody, const b2Vec2 &localAttachPoint, std::uint16_t categoryBits, std::uint16_t maskBits) {
    segments.resize(descs.size());

    b2BodyId prevBody = attachBody;
    b2Vec2 prevAttachPoint = localAttachPoint;

    for (int si = 0; si < segments.size(); si++) {
        b2BodyDef bodyDef = b2DefaultBodyDef();

        bodyDef.type = b2_dynamicBody;

        float offset = descs[si].length * 0.5f - descs[si].thickness * 0.5f;

        float angle = b2Rot_GetAngle(b2Body_GetRotation(prevBody)) + descs[si].relativeAngle;

        b2Vec2 p = b2Body_GetWorldPoint(prevBody, prevAttachPoint);

        bodyDef.position = (b2Vec2){p.x + std::cos(angle) * offset, p.y + std::sin(angle) * offset};
        bodyDef.rotation = b2MakeRot(angle);
        bodyDef.enableSleep = false;

        segments[si].body = b2CreateBody(world, &bodyDef);

        b2Polygon box = b2MakeBox(descs[si].length * 0.5f, descs[si].thickness * 0.5f);
        b2ShapeDef shape = b2DefaultShapeDef();
        shape.density = descs[si].density;
        shape.material.friction = descs[si].friction;
        shape.material.restitution = descs[si].restitution;
        shape.filter.categoryBits = categoryBits;
        shape.filter.maskBits = maskBits;

        segments[si].bodyShape = b2CreatePolygonShape(segments[si].body, &shape, &box);

        b2RevoluteJointDef jointDef = b2RevoluteJointDef();

        jointDef.bodyIdA = prevBody;

        jointDef.bodyIdB = segments[si].body;

        jointDef.referenceAngle = descs[si].relativeAngle;
        jointDef.localAnchorA = prevAttachPoint;
        jointDef.localAnchorB = (b2Vec2){-offset, 0.0f};
        jointDef.collideConnected = false;
        jointDef.lowerAngle = descs[si].minAngle;
        jointDef.upperAngle = descs[si].maxAngle;
        jointDef.enableLimit = true;
        jointDef.maxMotorTorque = descs[si].maxTorque;
        //jointDef.motorSpeed = descs[si].maxSpeed;
        jointDef.enableMotor = descs[si].motorEnabled;

        segments[si].maxSpeed = descs[si].maxSpeed;
        segments[si].minAngle = descs[si].minAngle;
        segments[si].maxAngle = descs[si].maxAngle;

        segments[si].joint = b2CreateRevoluteJoint(world, &jointDef);

        prevBody = segments[si].body;
        prevAttachPoint = (b2Vec2){offset, 0.0f};
    }
}

void Runner::Limb::remove(b2WorldId world) {
    for (int si = segments.size() - 1; si >= 0; si--) {
        b2DestroyJoint(segments[si].joint);
        b2DestroyBody(segments[si].body);
    }
}

Runner::~Runner() {
    destroy();
}

void Runner::destroy() {
    if (initialized) {
        leftBackLimb.remove(world);
        leftFrontLimb.remove(world);

        rightBackLimb.remove(world);
        rightFrontLimb.remove(world);

        b2DestroyBody(body);
    }
}

void Runner::createDefault(b2WorldId world, const b2Vec2 &position, float angle, int layer) {
    destroy();

    this->world = world;

    std::vector<LimbSegmentDesc> leftSegments(2);

    leftSegments[0].relativeAngle = 3.141592f * -0.75f;
    leftSegments[1].relativeAngle = 3.141592f * 0.5f;

    leftSegments[0].length = 0.15f;
    leftSegments[1].length = 0.15f;

    std::vector<LimbSegmentDesc> rightSegments(2);

    rightSegments[0].relativeAngle = 3.141592f * -0.75f;
    rightSegments[1].relativeAngle = 3.141592f * 0.5f;

    rightSegments[0].length = 0.15f;
    rightSegments[1].length = 0.15f;

    b2BodyDef bodyDef = b2DefaultBodyDef();

    bodyDef.type = b2_dynamicBody;

    bodyDef.position = position;
    bodyDef.rotation = b2MakeRot(angle);
    bodyDef.enableSleep = false;

    body = b2CreateBody(world, &bodyDef);

    b2Polygon box = b2MakeBox(bodyWidth * 0.5f, bodyHeight * 0.5f);
    b2ShapeDef shape = b2DefaultShapeDef();
    shape.density = bodyDensity;
    shape.material.friction = bodyFriction;
    shape.material.restitution = bodyRestitution;
    shape.filter.categoryBits = 1 << layer;
    shape.filter.maskBits = 1;

    bodyShape = b2CreatePolygonShape(body, &shape, &box);

    leftBackLimb.create(world, leftSegments, body, (b2Vec2){-bodyWidth * 0.5f + legInset, -bodyHeight * 0.5f}, 1 << layer, 1);
    leftFrontLimb.create(world, leftSegments, body, (b2Vec2){-bodyWidth * 0.5f + legInset, -bodyHeight * 0.5f}, 1 << layer, 1);

    rightBackLimb.create(world, rightSegments, body, (b2Vec2){bodyWidth * 0.5f - legInset, -bodyHeight * 0.5f}, 1 << (layer + 1), 1);
    rightFrontLimb.create(world, rightSegments, body, (b2Vec2){bodyWidth * 0.5f - legInset, -bodyHeight * 0.5f}, 1 << (layer + 1), 1);

    whiskerResults.resize(numWhiskers);
    std::fill(whiskerResults.begin(), whiskerResults.end(), 1.0f);

    lVelPrev = (b2Vec2){0.0f, 0.0f};
    rVelPrev = 0.0f;

    positions = std::vector<float>(8, 0.0f);
    speeds = std::vector<float>(8, 0.0f);

    initialized = true;
}

void Runner::renderDefault(sf::RenderTarget &rt, const sf::Color &color, float metersToPixels) {
    assert(initialized);

    // Render back legs
    for (int si = leftBackLimb.segments.size() - 1; si >= 0; si--) {
        b2Polygon poly = b2Shape_GetPolygon(leftBackLimb.segments[si].bodyShape);
        int numVertices = poly.count;

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(poly.vertices[i].x, poly.vertices[i].y));

        b2Vec2 p = b2Body_GetPosition(leftBackLimb.segments[si].body);
        shape.setPosition(metersToPixels * sf::Vector2f(p.x, -p.y));
        shape.setRotation(sf::radians(-b2Rot_GetAngle(b2Body_GetRotation(leftBackLimb.segments[si].body))));
        shape.setScale(sf::Vector2f(metersToPixels, -metersToPixels));

        shape.setFillColor(mulColors(sf::Color(200, 200, 200), color));
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }

    for (int si = rightBackLimb.segments.size() - 1; si >= 0; si--) {
        b2Polygon poly = b2Shape_GetPolygon(rightBackLimb.segments[si].bodyShape);
        int numVertices = poly.count;

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(poly.vertices[i].x, poly.vertices[i].y));

        b2Vec2 p = b2Body_GetPosition(rightBackLimb.segments[si].body);
        shape.setPosition(metersToPixels * sf::Vector2f(p.x, -p.y));
        shape.setRotation(sf::radians(-b2Rot_GetAngle(b2Body_GetRotation(rightBackLimb.segments[si].body))));
        shape.setScale(sf::Vector2f(metersToPixels, -metersToPixels));

        shape.setFillColor(mulColors(sf::Color(200, 200, 200), color));
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }

    // Render body
    {
        b2Polygon poly = b2Shape_GetPolygon(bodyShape);
        int numVertices = poly.count;

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(poly.vertices[i].x, poly.vertices[i].y));

        b2Vec2 p = b2Body_GetPosition(body);
        shape.setPosition(metersToPixels * sf::Vector2f(p.x, -p.y));
        shape.setRotation(sf::radians(-b2Rot_GetAngle(b2Body_GetRotation(body))));
        shape.setScale(sf::Vector2f(metersToPixels, -metersToPixels));

        shape.setFillColor(mulColors(sf::Color::White, color));
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }

    for (int si = leftFrontLimb.segments.size() - 1; si >= 0; si--) {
        b2Polygon poly = b2Shape_GetPolygon(leftFrontLimb.segments[si].bodyShape);
        int numVertices = poly.count;

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(poly.vertices[i].x, poly.vertices[i].y));

        b2Vec2 p = b2Body_GetPosition(leftFrontLimb.segments[si].body);
        shape.setPosition(metersToPixels * sf::Vector2f(p.x, -p.y));
        shape.setRotation(sf::radians(-b2Rot_GetAngle(b2Body_GetRotation(leftFrontLimb.segments[si].body))));
        shape.setScale(sf::Vector2f(metersToPixels, -metersToPixels));

        shape.setFillColor(color);
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }

    for (int si = rightFrontLimb.segments.size() - 1; si >= 0; si--) {
        b2Polygon poly = b2Shape_GetPolygon(rightFrontLimb.segments[si].bodyShape);
        int numVertices = poly.count;

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(poly.vertices[i].x, poly.vertices[i].y));

        b2Vec2 p = b2Body_GetPosition(rightFrontLimb.segments[si].body);
        shape.setPosition(metersToPixels * sf::Vector2f(p.x, -p.y));
        shape.setRotation(sf::radians(-b2Rot_GetAngle(b2Body_GetRotation(rightFrontLimb.segments[si].body))));
        shape.setScale(sf::Vector2f(metersToPixels, -metersToPixels));

        shape.setFillColor(color);
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }

    b2Vec2 whiskersStart(b2Body_GetWorldPoint(body, (b2Vec2){bodyWidth * 0.5f, 0.0f}));
    float whiskersBaseAngle = b2Rot_GetAngle(b2Body_GetRotation(body));

    for (int i = 0; i < numWhiskers; i++) {
        float angle = whiskersBaseAngle - whiskerSpread * i;

        sf::RectangleShape rs;
        rs.setSize(sf::Vector2f(whiskerLen * whiskerResults[i], 0.01f) * metersToPixels);
        rs.setPosition(sf::Vector2f(whiskersStart.x, -whiskersStart.y) * metersToPixels);
        rs.setRotation(sf::radians(-angle));
        rs.setFillColor(sf::Color(0, 255, 0, 50));

        //float d = std::sqrt(std::pow(vertices[0].position.x - vertices[1].position.x, 2) + std::pow(vertices[0].position.y - vertices[1].position.y, 2));

        rt.draw(rs);
    }
}

void Runner::getStateVector(std::vector<float> &state) {
    assert(initialized);

    const int stateSize = 2 + 2 + 2 + 2 + 1 + 2 + 2 + numWhiskers + 3;

    if (state.size() != stateSize)
        state.resize(stateSize);

    int si = 0;

    for (int i = 0; i < 2; i++)
        state[si++] = b2RevoluteJoint_GetAngle(leftBackLimb.segments[i].joint);

    for (int i = 0; i < 2; i++)
        state[si++] = b2RevoluteJoint_GetAngle(leftFrontLimb.segments[i].joint);

    for (int i = 0; i < 2; i++)
        state[si++] = b2RevoluteJoint_GetAngle(rightBackLimb.segments[i].joint);

    for (int i = 0; i < 2; i++)
        state[si++] = b2RevoluteJoint_GetAngle(rightFrontLimb.segments[i].joint);

    state[si++] = b2Rot_GetAngle(b2Body_GetRotation(body));

    {
        state[si] = 0.0f;

        std::vector<b2ContactData> cds(16);
        std::vector<b2ShapeId> shapes(16);
        
        int c = b2Body_GetContactData(leftBackLimb.segments.back().body, cds.data(), cds.size());

        for (int i = 0; i < c; c++) {
            b2Filter f = b2Shape_GetFilter(cds[i].shapeIdA);

            if (f.categoryBits != 0x0002 && f.categoryBits != 0x0004) {
                state[si++] = 1.0f;

                break;
            }
        }
    }

    {
        state[si] = 0.0f;

        std::vector<b2ContactData> cds(16);
        std::vector<b2ShapeId> shapes(16);
        
        int c = b2Body_GetContactData(leftFrontLimb.segments.back().body, cds.data(), cds.size());

        for (int i = 0; i < c; c++) {
            b2Filter f = b2Shape_GetFilter(cds[i].shapeIdA);

            if (f.categoryBits != 0x0002 && f.categoryBits != 0x0004) {
                state[si++] = 1.0f;

                break;
            }
        }
    }

    {
        state[si] = 0.0f;

        std::vector<b2ContactData> cds(16);
        std::vector<b2ShapeId> shapes(16);
        
        int c = b2Body_GetContactData(rightBackLimb.segments.back().body, cds.data(), cds.size());

        for (int i = 0; i < c; c++) {
            b2Filter f = b2Shape_GetFilter(cds[i].shapeIdA);

            if (f.categoryBits != 0x0002 && f.categoryBits != 0x0004) {
                state[si++] = 1.0f;

                break;
            }
        }
    }

    {
        state[si] = 0.0f;

        std::vector<b2ContactData> cds(16);
        std::vector<b2ShapeId> shapes(16);
        
        int c = b2Body_GetContactData(rightFrontLimb.segments.back().body, cds.data(), cds.size());

        for (int i = 0; i < c; c++) {
            b2Filter f = b2Shape_GetFilter(cds[i].shapeIdA);

            if (f.categoryBits != 0x0002 && f.categoryBits != 0x0004) {
                state[si++] = 1.0f;

                break;
            }
        }
    }

    // Whiskers
    b2Vec2 whiskersStart(b2Body_GetWorldPoint(body, (b2Vec2){bodyWidth * 0.5f, 0.0f}));
    float whiskersBaseAngle = b2Rot_GetAngle(b2Body_GetRotation(body));

    for (int i = 0; i < numWhiskers; i++) {
        float angle = whiskersBaseAngle - whiskerSpread * i;

        b2QueryFilter f = b2DefaultQueryFilter();

        b2RayResult res = b2World_CastRayClosest(world, whiskersStart, (b2Vec2){std::cos(angle) * whiskerLen, whiskersStart.y + std::sin(angle) * whiskerLen}, f);

        whiskerResults[i] = state[si++] = res.fraction;
    }

    // IMU
    b2Vec2 lVel = b2Body_GetLinearVelocity(body);
    b2Vec2 lAccel = (b2Vec2){lVel.x - lVelPrev.x, lVel.y - lVelPrev.y};
    lVelPrev = lVel;

    float rVel = b2Body_GetAngularVelocity(body);
    float rAccel = rVel - rVelPrev;
    rVelPrev = rVel;

    state[si++] = lAccel.x;
    state[si++] = lAccel.y;
    state[si++] = rAccel;
}

void Runner::motorUpdate(const std::vector<float> &actions, float propPos, float propSpeed) {
    assert(initialized);

    int ai = 0;

    for (int i = 0; i < 2; i++) {
        Limb &limb = leftBackLimb;
        LimbSegment &seg = limb.segments[i];

        float pos = actions[ai] * (seg.maxAngle - seg.minAngle) + seg.minAngle;

        positions[ai] += propPos * (pos - positions[ai]);
        
        float speed = positions[ai] - b2RevoluteJoint_GetAngle(seg.joint);

        speeds[ai] += propSpeed * (speed - speeds[ai]);

        b2RevoluteJoint_SetMotorSpeed(seg.joint, speeds[ai] * seg.maxSpeed);

        ai++;
    }

    for (int i = 0; i < 2; i++) {
        Limb &limb = leftFrontLimb;
        LimbSegment &seg = limb.segments[i];

        float pos = actions[ai] * (seg.maxAngle - seg.minAngle) + seg.minAngle;

        positions[ai] += propPos * (pos - positions[ai]);
        
        float speed = positions[ai] - b2RevoluteJoint_GetAngle(seg.joint);

        speeds[ai] += propSpeed * (speed - speeds[ai]);

        b2RevoluteJoint_SetMotorSpeed(seg.joint, speeds[ai] * seg.maxSpeed);

        ai++;
    }

    for (int i = 0; i < 2; i++) {
        Limb &limb = rightBackLimb;
        LimbSegment &seg = limb.segments[i];

        float pos = actions[ai] * (seg.maxAngle - seg.minAngle) + seg.minAngle;

        positions[ai] += propPos * (pos - positions[ai]);
        
        float speed = positions[ai] - b2RevoluteJoint_GetAngle(seg.joint);

        speeds[ai] += propSpeed * (speed - speeds[ai]);

        b2RevoluteJoint_SetMotorSpeed(seg.joint, speeds[ai] * seg.maxSpeed);

        ai++;
    }

    for (int i = 0; i < 2; i++) {
        Limb &limb = rightFrontLimb;
        LimbSegment &seg = limb.segments[i];

        float pos = actions[ai] * (seg.maxAngle - seg.minAngle) + seg.minAngle;

        positions[ai] += propPos * (pos - positions[ai]);
        
        float speed = positions[ai] - b2RevoluteJoint_GetAngle(seg.joint);

        speeds[ai] += propSpeed * (speed - speeds[ai]);

        b2RevoluteJoint_SetMotorSpeed(seg.joint, speeds[ai] * seg.maxSpeed);

        ai++;
    }
}
