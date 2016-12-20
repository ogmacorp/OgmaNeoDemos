// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <SFML/Graphics.hpp>

namespace vis {
	struct Point {
		sf::Vector2f _position;

		sf::Color _color;

		Point() :
			_color(sf::Color::Black) {
        }
	};

	struct Curve {
		std::string _name;

		float _shadow;
		sf::Vector2f _shadowOffset;

		std::vector<Point> _points;

		Curve() :
			_shadow(0.5f), _shadowOffset(-4.0f, 4.0f) {
        }
	};

	struct Plot {
        bool _plotXAxisTicks;        
        sf::Color _axesColor;

        sf::Color _backgroundColor;
        sf::Color _plotBackgroundColor;

        std::vector<Curve> _curves;

		Plot() :
			_axesColor(sf::Color::Black), _backgroundColor(sf::Color::White),
            _plotBackgroundColor(sf::Color::White), _plotXAxisTicks(false) {
        }

		void draw(sf::RenderTarget &target, const sf::Texture &lineGradientTexture, const sf::Font &tickFont, float tickTextScale,
			const sf::Vector2f &domain, const sf::Vector2f &range, const sf::Vector2f &margins, const sf::Vector2f &tickIncrements,
            float axesSize, float lineSize, float tickSize, float tickLength, float textTickOffset, int precision);
	};

	float vectorMagnitude(const sf::Vector2f &vector);
	sf::Vector2f vectorNormalize(const sf::Vector2f &vector);
	float vectorDot(const sf::Vector2f &left, const sf::Vector2f &right);
}