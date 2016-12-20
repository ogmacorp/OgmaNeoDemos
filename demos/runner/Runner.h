// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <Box2D/Box2D.h>

class Runner {
public:
	struct LimbSegmentDesc {
		float _relativeAngle;
		float _thickness, _length;
		float _minAngle, _maxAngle;
		float _maxTorque;
		float _maxSpeed;
		float _density;
		float _friction;
		float _restitution;
		bool _motorEnabled;

		LimbSegmentDesc()
			: _relativeAngle(0.0f),
			_thickness(0.03f), _length(0.125f),
			_minAngle(-0.5f), _maxAngle(0.5f),
			_maxTorque(10.0f),
			_maxSpeed(500.0f),
			_density(2.0f),
			_friction(2.0f),
			_restitution(0.01f),
			_motorEnabled(true)
		{}
	};

	struct LimbSegment {
		b2Body* _pBody;
		b2PolygonShape _bodyShape;
		b2RevoluteJoint* _pJoint;

		float _maxSpeed;
		float _minAngle;
		float _maxAngle;
	};

	struct Limb {
		std::vector<LimbSegment> _segments;

		void create(b2World* pWorld, const std::vector<LimbSegmentDesc> &descs, b2Body* pAttachBody, const b2Vec2 &localAttachPoint, uint16 categoryBits, uint16 maskBits);
		void remove(b2World* pWorld);
	};
private:
	std::shared_ptr<b2World> _world;

public:
	static sf::Color mulColors(const sf::Color &c1, const sf::Color &c2) {
		const float byteInv = 1.0f / 255.0f;

		return sf::Color(c1.r * c2.r * byteInv,
			c1.g * c2.g * byteInv,
			c1.b * c2.b * byteInv);
	}

	b2Body* _pBody;
	b2PolygonShape _bodyShape;

	Limb _leftBackLimb;
	Limb _leftFrontLimb;
	Limb _rightBackLimb;
	Limb _rightFrontLimb;

	Runner()
		: _world(nullptr)
	{}

	~Runner();

	void createDefault(const std::shared_ptr<b2World> &world, const b2Vec2 &position, float angle, int layer);

	void renderDefault(sf::RenderTarget &rt, const sf::Color &color, float metersToPixels);

	void getStateVector(std::vector<float> &state);
	void motorUpdate(const std::vector<float> &action, float interpolateFactor = 8.0f, float smoothIn = 0.0f, float minSmooth = 1.0f);
};