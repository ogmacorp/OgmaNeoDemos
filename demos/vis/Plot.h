// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <SFML/Graphics.hpp>

namespace vis {
	struct Point {
		sf::Vector2f position;

		sf::Color color;

		Point() :
			color(sf::Color::Black) {
        }
	};

	struct Curve {
		std::string name;

		float shadow;
		sf::Vector2f shadowOffset;

		std::vector<Point> points;

		Curve() :
			shadow(0.5f), shadowOffset(-4.0f, 4.0f) {
        }
	};

	struct Plot {
        bool plotXAxisTicks;        
        sf::Color axesColor;

        sf::Color backgroundColor;
        sf::Color plotBackgroundColor;

        std::vector<Curve> curves;

		Plot() :
			axesColor(sf::Color::Black), backgroundColor(sf::Color::White),
            plotBackgroundColor(sf::Color::White), plotXAxisTicks(false) {
        }

		void draw(sf::RenderTarget &target, const sf::Texture &lineGradientTexture, const sf::Font &tickFont, float tickTextScale,
			const sf::Vector2f &domain, const sf::Vector2f &range, const sf::Vector2f &margins, const sf::Vector2f &tickIncrements,
            float axesSize, float lineSize, float tickSize, float tickLength, float textTickOffset, int precision);
	};

	float vectorMagnitude(const sf::Vector2f &vector);
	sf::Vector2f vectorNormalize(const sf::Vector2f &vector);
	float vectorDot(const sf::Vector2f &left, const sf::Vector2f &right);
}