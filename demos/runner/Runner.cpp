// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOSLICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <cmath>
#include "Runner.h"

void Runner::Limb::create(b2World* world, const std::vector<LimbSegmentDesc> &descs, b2Body* attachBody, const b2Vec2 &localAttachPoint, uint16 categoryBits, uint16 maskBits) {
    segments.resize(descs.size());

    b2Body* prevBody = attachBody;
    b2Vec2 prevAttachPoint = localAttachPoint;

    for (int si = 0; si < segments.size(); si++) {
        b2BodyDef bodyDef;

        bodyDef.type = b2_dynamicBody;

        float offset = descs[si].length * 0.5f - descs[si].thickness * 0.5f;

        float angle = prevBody->GetAngle() + descs[si].relativeAngle;

        bodyDef.position = prevBody->GetWorldPoint(prevAttachPoint) + b2Vec2(std::cos(angle) * offset, std::sin(angle) * offset);
        bodyDef.angle = angle;
        bodyDef.allowSleep = false;

        segments[si].body = world->CreateBody(&bodyDef);

        segments[si].bodyShape.SetAsBox(descs[si].length * 0.5f, descs[si].thickness * 0.5f);

        b2FixtureDef fixtureDef;

        fixtureDef.shape = &segments[si].bodyShape;

        fixtureDef.density = descs[si].density;

        fixtureDef.friction = descs[si].friction;

        fixtureDef.restitution = descs[si].restitution;

        fixtureDef.filter.categoryBits = categoryBits;
        fixtureDef.filter.maskBits = maskBits;

        segments[si].body->CreateFixture(&fixtureDef);

        b2RevoluteJointDef jointDef;

        jointDef.bodyA = prevBody;

        jointDef.bodyB = segments[si].body;

        jointDef.referenceAngle = descs[si].relativeAngle;
        jointDef.localAnchorA = prevAttachPoint;
        jointDef.localAnchorB = b2Vec2(-offset, 0.0f);
        jointDef.collideConnected = false;
        jointDef.lowerAngle = descs[si].minAngle;
        jointDef.upperAngle = descs[si].maxAngle;
        jointDef.enableLimit = true;
        jointDef.maxMotorTorque = descs[si].maxTorque;
        jointDef.motorSpeed = descs[si].maxSpeed;
        jointDef.enableMotor = descs[si].motorEnabled;

        segments[si].maxSpeed = descs[si].maxSpeed;
        segments[si].minAngle = descs[si].minAngle;
        segments[si].maxAngle = descs[si].maxAngle;

        segments[si].joint = static_cast<b2RevoluteJoint*>(world->CreateJoint(&jointDef));

        prevBody = segments[si].body;
        prevAttachPoint = b2Vec2(offset, 0.0f);
    }
}

void Runner::Limb::remove(b2World* world) {
    for (int si = segments.size() - 1; si >= 0; si--) {
        world->DestroyJoint(segments[si].joint);
        world->DestroyBody(segments[si].body);
    }
}

Runner::~Runner() {
    if (world != nullptr) {
        leftBackLimb.remove(world.get());
        leftFrontLimb.remove(world.get());

        rightBackLimb.remove(world.get());
        rightFrontLimb.remove(world.get());

        world->DestroyBody(body);
    }
}

void Runner::createDefault(const std::shared_ptr<b2World> &world, const b2Vec2 &position, float angle, int layer) {
    const float bodyWidth = 0.45f;
    const float legInset = 0.075f;
    const float bodyHeight = 0.1f;
    const float bodyDensity = 1.0f;
    const float bodyFriction = 1.0f;
    const float bodyRestitution = 0.01f;

    std::vector<LimbSegmentDesc> leftSegments(3);

    leftSegments[0].relativeAngle = 3.141592f * -0.25f;
    leftSegments[1].relativeAngle = 3.141592f * -0.5f;
    leftSegments[2].relativeAngle = 3.141592f * 0.5f;

    leftSegments[0].length = 0.08f;
    leftSegments[1].length = 0.13f;
    leftSegments[2].length = 0.13f;

    std::vector<LimbSegmentDesc> rightSegments(2);

    rightSegments[0].relativeAngle = 3.141592f * -0.75f;
    rightSegments[1].relativeAngle = 3.141592f * 0.5f;

    rightSegments[0].length = 0.15f;
    rightSegments[1].length = 0.15f;

    this->world = world;

    b2BodyDef bodyDef;

    bodyDef.type = b2_dynamicBody;

    bodyDef.position = position;
    bodyDef.angle = angle;
    bodyDef.allowSleep = false;

    body = world->CreateBody(&bodyDef);

    bodyShape.SetAsBox(bodyWidth * 0.5f, bodyHeight * 0.5f);

    b2FixtureDef fixtureDef;

    fixtureDef.shape = &bodyShape;

    fixtureDef.density = bodyDensity;

    fixtureDef.friction = bodyFriction;

    fixtureDef.restitution = bodyRestitution;

    fixtureDef.filter.categoryBits = 1 << layer;
    fixtureDef.filter.maskBits = 1;

    body->CreateFixture(&fixtureDef);

    leftBackLimb.create(world.get(), leftSegments, body, b2Vec2(-bodyWidth * 0.5f + legInset, -bodyHeight * 0.5f), 1 << layer, 1);
    leftFrontLimb.create(world.get(), leftSegments, body, b2Vec2(-bodyWidth * 0.5f + legInset, -bodyHeight * 0.5f), 1 << layer, 1);

    rightBackLimb.create(world.get(), rightSegments, body, b2Vec2(bodyWidth * 0.5f - legInset, -bodyHeight * 0.5f), 1 << (layer + 1), 1);
    rightFrontLimb.create(world.get(), rightSegments, body, b2Vec2(bodyWidth * 0.5f - legInset, -bodyHeight * 0.5f), 1 << (layer + 1), 1);
}

void Runner::renderDefault(sf::RenderTarget &rt, const sf::Color &color, float metersToPixels) {
    // Render back legs
    for (int si = leftBackLimb.segments.size() - 1; si >= 0; si--) {
        int numVertices = leftBackLimb.segments[si].bodyShape.GetVertexCount();

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(leftBackLimb.segments[si].bodyShape.GetVertex(i).x, leftBackLimb.segments[si].bodyShape.GetVertex(i).y));

        shape.setPosition(metersToPixels * sf::Vector2f(leftBackLimb.segments[si].body->GetPosition().x, -leftBackLimb.segments[si].body->GetPosition().y));
        shape.setRotation(-leftBackLimb.segments[si].body->GetAngle() * 180.0f / 3.141592f);
        shape.setScale(metersToPixels, -metersToPixels);

        shape.setFillColor(mulColors(sf::Color(200, 200, 200), color));
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }

    for (int si = rightBackLimb.segments.size() - 1; si >= 0; si--) {
        int numVertices = rightBackLimb.segments[si].bodyShape.GetVertexCount();

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(rightBackLimb.segments[si].bodyShape.GetVertex(i).x, rightBackLimb.segments[si].bodyShape.GetVertex(i).y));

        shape.setPosition(metersToPixels * sf::Vector2f(rightBackLimb.segments[si].body->GetPosition().x, -rightBackLimb.segments[si].body->GetPosition().y));
        shape.setRotation(-rightBackLimb.segments[si].body->GetAngle() * 180.0f / 3.141592f);
        shape.setScale(metersToPixels, -metersToPixels);

        shape.setFillColor(mulColors(sf::Color(200, 200, 200), color));
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }

    // Render body
    {
        int numVertices = bodyShape.GetVertexCount();

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(bodyShape.GetVertex(i).x, bodyShape.GetVertex(i).y));

        shape.setPosition(metersToPixels * sf::Vector2f(body->GetPosition().x, -body->GetPosition().y));
        shape.setRotation(-body->GetAngle() * 180.0f / 3.141592f);
        shape.setScale(metersToPixels, -metersToPixels);

        shape.setFillColor(mulColors(sf::Color::White, color));
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }

    // Render front legs
    for (int si = 0; si < leftFrontLimb.segments.size(); si++) {
        int numVertices = leftFrontLimb.segments[si].bodyShape.GetVertexCount();

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(leftFrontLimb.segments[si].bodyShape.GetVertex(i).x, leftFrontLimb.segments[si].bodyShape.GetVertex(i).y));

        shape.setPosition(metersToPixels * sf::Vector2f(leftFrontLimb.segments[si].body->GetPosition().x, -leftFrontLimb.segments[si].body->GetPosition().y));
        shape.setRotation(-leftFrontLimb.segments[si].body->GetAngle() * 180.0f / 3.141592f);
        shape.setScale(metersToPixels, -metersToPixels);

        shape.setFillColor(mulColors(sf::Color::White, color));
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }

    for (int si = 0; si < rightFrontLimb.segments.size(); si++) {
        int numVertices = rightFrontLimb.segments[si].bodyShape.GetVertexCount();

        sf::ConvexShape shape;

        shape.setPointCount(numVertices);

        for (int i = 0; i < numVertices; i++)
            shape.setPoint(i, sf::Vector2f(rightFrontLimb.segments[si].bodyShape.GetVertex(i).x, rightFrontLimb.segments[si].bodyShape.GetVertex(i).y));

        shape.setPosition(metersToPixels * sf::Vector2f(rightFrontLimb.segments[si].body->GetPosition().x, -rightFrontLimb.segments[si].body->GetPosition().y));
        shape.setRotation(-rightFrontLimb.segments[si].body->GetAngle() * 180.0f / 3.141592f);
        shape.setScale(metersToPixels, -metersToPixels);

        shape.setFillColor(mulColors(sf::Color::White, color));
        shape.setOutlineColor(sf::Color::Black);
        shape.setOutlineThickness(0.01f);

        rt.draw(shape);
    }
}

void Runner::getStateVector(std::vector<float> &state) {
    const int stateSize = 3 + 3 + 2 + 2 + 1 + 2 + 2;

    if (state.size() != stateSize)
        state.resize(stateSize);

    int si = 0;

    for (int i = 0; i < 3; i++)
        state[si++] = leftBackLimb.segments[i].joint->GetJointAngle();

    for (int i = 0; i < 3; i++)
        state[si++] = leftFrontLimb.segments[i].joint->GetJointAngle();

    for (int i = 0; i < 2; i++)
        state[si++] = rightBackLimb.segments[i].joint->GetJointAngle();

    for (int i = 0; i < 2; i++)
        state[si++] = rightFrontLimb.segments[i].joint->GetJointAngle();

    state[si++] = body->GetAngle();

    b2ContactEdge* edge;

    state[si] = 0.0f;
    
    edge = leftBackLimb.segments.back().body->GetContactList();

    while (edge != nullptr) {
        if (edge->contact->IsTouching() && edge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0002 && edge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0004) {
            state[si++] = 1.0f;

            break;
        }

        edge = edge->next;
    }

    state[si] = 0.0f;

    edge = leftFrontLimb.segments.back().body->GetContactList();

    while (edge != nullptr) {
        if (edge->contact->IsTouching() && edge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0002 && edge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0004) {
            state[si++] = 1.0f;

            break;
        }

        edge = edge->next;
    }

    state[si] = 0.0f;

    edge = rightBackLimb.segments.back().body->GetContactList();

    while (edge != nullptr) {
        if (edge->contact->IsTouching() && edge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0002 && edge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0004) {
            state[si++] = 1.0f;

            break;
        }

        edge = edge->next;
    }

    state[si] = 0.0f;

    edge = rightFrontLimb.segments.back().body->GetContactList();

    while (edge != nullptr) {
        if (edge->contact->IsTouching() && edge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0002 && edge->contact->GetFixtureA()->GetFilterData().categoryBits != 0x0004) {
            state[si++] = 1.0f;

            break;
        }

        edge = edge->next;
    }
}

void Runner::motorUpdate(const std::vector<float> &action, float interpolateFactor, float smoothIn, float minSmooth) {
    int ai = 0;

    for (int i = 0; i < 3; i++) {
        float target = action[ai++] * (leftBackLimb.segments[i].maxAngle - leftBackLimb.segments[i].minAngle) + leftBackLimb.segments[i].minAngle;

        float speed = interpolateFactor * (target - leftBackLimb.segments[i].joint->GetJointAngle()) * (minSmooth + 1.0f - std::exp(-smoothIn * std::abs(leftBackLimb.segments[i].joint->GetJointSpeed())));

        //if (std::abs(speed) > leftBackLimb.segments[i].maxSpeed)
        //	speed = speed > 0.0f ? leftBackLimb.segments[i].maxSpeed : -leftBackLimb.segments[i].maxSpeed;

        leftBackLimb.segments[i].joint->SetMotorSpeed(speed);
    }

    for (int i = 0; i < 3; i++) {
        float target = action[ai++] * (leftFrontLimb.segments[i].maxAngle - leftFrontLimb.segments[i].minAngle) + leftFrontLimb.segments[i].minAngle;

        float speed = interpolateFactor * (target - leftFrontLimb.segments[i].joint->GetJointAngle()) * (minSmooth + 1.0f - std::exp(-smoothIn * std::abs(leftFrontLimb.segments[i].joint->GetJointSpeed())));

        //if (std::abs(speed) > leftFrontLimb.segments[i].maxSpeed)
        //	speed = speed > 0.0f ? leftFrontLimb.segments[i].maxSpeed : -leftFrontLimb.segments[i].maxSpeed;

        leftFrontLimb.segments[i].joint->SetMotorSpeed(speed);
    }

    for (int i = 0; i < 2; i++) {
        float target = action[ai++] * (rightBackLimb.segments[i].maxAngle - rightBackLimb.segments[i].minAngle) + rightBackLimb.segments[i].minAngle;

        float speed = interpolateFactor * (target - rightBackLimb.segments[i].joint->GetJointAngle()) * (minSmooth + 1.0f - std::exp(-smoothIn * std::abs(rightBackLimb.segments[i].joint->GetJointSpeed())));

        //if (std::abs(speed) > rightBackLimb.segments[i].maxSpeed)
        //	speed = speed > 0.0f ? rightBackLimb.segments[i].maxSpeed : -rightBackLimb.segments[i].maxSpeed;

        rightBackLimb.segments[i].joint->SetMotorSpeed(speed);
    }

    for (int i = 0; i < 2; i++) {
        float target = action[ai++] * (rightFrontLimb.segments[i].maxAngle - rightFrontLimb.segments[i].minAngle) + rightFrontLimb.segments[i].minAngle;

        float speed = interpolateFactor * (target - rightFrontLimb.segments[i].joint->GetJointAngle()) * (minSmooth + 1.0f - std::exp(-smoothIn * std::abs(rightFrontLimb.segments[i].joint->GetJointSpeed())));

        //if (std::abs(speed) > rightFrontLimb.segments[i].maxSpeed)
        //	speed = speed > 0.0f ? rightFrontLimb.segments[i].maxSpeed : -rightFrontLimb.segments[i].maxSpeed;

        rightFrontLimb.segments[i].joint->SetMotorSpeed(speed);
    }
}
