#ifndef _SFML_PLOT_UTILS_HPP_
#define _SFML_PLOT_UTILS_HPP_

#include <SFML/Graphics.hpp>
#include <sstream>
#include <cmath>


namespace vis {
	float vectorMagnitude(const sf::Vector2f &vector)
	{
		return std::sqrt(vector.x * vector.x + vector.y * vector.y);
	};

	sf::Vector2f vectorNormalize(const sf::Vector2f &vector)
	{
		float magnitude = vectorMagnitude(vector);
		return vector / magnitude;
	};

	float vectorDot(const sf::Vector2f &left, const sf::Vector2f &right)
	{
		return left.x * right.x + left.y * right.y;
	};


	struct Point {
		sf::Vector2f _position;

		sf::Color _color;

		Point()
			: _color(sf::Color::Black)
		{}
	};

	struct Curve {
		std::string _name;

		float _shadow;
		sf::Vector2f _shadowOffset;

		std::vector<Point> _points;

		Curve()
			: _shadow(0.5f), _shadowOffset(-4.0f, 4.0f)
		{}
	};

	struct Plot
	{
		sf::Color 				_axesColor;
		sf::Color 				_backgroundColor;
		sf::RenderTexture _plotRT;
		sf::Sprite 				_plotSprite;
		sf::Texture 			_lineGradient;
		sf::Font 					_tickFont;
		sf::Text 					_xlabel, _ylabel;

		sf::Vector2f 			_pos;
		sf::Vector2i 			_size;
		int 							_numCurves;

		std::vector<Curve> _curves;
		int maxBufferSize;				// buffer length for Curve

		Plot(sf::Vector2f pos=sf::Vector2f(20,20), sf::Vector2i size=sf::Vector2i(20,20), int numCurves=0, std::string xlabel="", std::string ylabel="", int bufferSize = 300)
			: _pos(pos), _size(size), _numCurves(numCurves), _axesColor(sf::Color::Black), _backgroundColor(sf::Color::White), maxBufferSize(bufferSize)
		{
			_plotRT.create(_size.x, _size.y, false);
			_plotRT.setActive();
  		_plotRT.clear(_backgroundColor);

      _plotSprite.setPosition(_pos.x, _pos.y);

		  _lineGradient.loadFromFile("resources/lineGradient.png");
		  _tickFont.loadFromFile("resources/Hack-Regular.ttf");

		  const int xlabelHeight = 20;
		  //set up text for xlabel display		  
		  _xlabel.setFont(_tickFont);
		  _xlabel.setCharacterSize(16);
		  _xlabel.setColor(sf::Color::Black);
		  _xlabel.setPosition(_pos.x + 0.5*_size.x, _pos.y + _size.y - xlabelHeight);
		  _xlabel.setString(xlabel.c_str());

		  //set up text for ylabel display		  
		  _ylabel.setFont(_tickFont);
		  _ylabel.setCharacterSize(16);
		  _ylabel.setColor(sf::Color::Black);
		  _ylabel.rotate(-90.f);
		  _ylabel.setString(ylabel.c_str());
		  sf::FloatRect rect = _ylabel.getLocalBounds();
		  _ylabel.setPosition(_pos.x + 10, _pos.y +.5*(_size.y + rect.width));

		  _curves.resize(numCurves);
		  for (auto i=0; i < numCurves; ++i) 	_curves[i]._shadow = 0.0;

		}

		void updateBuffer(int pIndx, vis::Point p, int firstIndex = 0)
		{
			if (pIndx >= _curves.size())
			{
				//std::cout << "Plot: invalid indx of the curve" << std::endl;
				return;
			}

		  _curves[pIndx]._points.push_back(p);
		  if (_curves[pIndx]._points.size() > maxBufferSize)
		    _curves[pIndx]._points.erase(_curves[pIndx]._points.begin());

		  for (auto it = _curves[pIndx]._points.begin(); it != _curves[pIndx]._points.end(); ++it, ++firstIndex)
		    it->_position.x = firstIndex;
		};

		void draw_(sf::RenderTexture &target, const sf::Texture &lineGradientTexture, const sf::Font &tickFont, float tickTextScale,
			const sf::Vector2f &domain, const sf::Vector2f &range, const sf::Vector2f &margins, const sf::Vector2f &tickIncrements, float axesSize, float lineSize, float tickSize, float tickLength, float textTickOffset, int precision, int firstIndex=-1)
		{
			target.clear(_backgroundColor);
			
			sf::Vector2f plotSize = sf::Vector2f(target.getSize().x - margins.x, target.getSize().y - margins.y);

			sf::Vector2f origin = sf::Vector2f(margins.x, target.getSize().y - margins.y);

			// Draw curves
			for (int c = 0; c < _curves.size(); c++) {
				if (_curves[c]._points.empty())
					continue;

				sf::VertexArray vertexArray;

				vertexArray.resize((_curves[c]._points.size() - 1) * 6);

				int index = 0;

				// Go through points
				for (int p = 0; p < _curves[c]._points.size() - 1; p++) {
					Point &point = _curves[c]._points[p];
					Point &pointNext = _curves[c]._points[p + 1];

					sf::Vector2f difference = pointNext._position - point._position;
					sf::Vector2f direction = vectorNormalize(difference);

					sf::Vector2f renderPointFirst, renderPointSecond;

					bool pointVisible = point._position.x >= domain.x && point._position.x <= domain.y &&
						point._position.y >= range.x && point._position.y <= range.y;

					bool pointNextVisible = pointNext._position.x >= domain.x && pointNext._position.x <= domain.y &&
						pointNext._position.y >= range.x && pointNext._position.y <= range.y;

					if (pointVisible || pointNextVisible)
					{
						sf::Vector2f renderPoint = sf::Vector2f(origin.x + (point._position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
							origin.y - (point._position.y - range.x) / (range.y - range.x) * plotSize.y);

						sf::Vector2f renderPointNext = sf::Vector2f(origin.x + (pointNext._position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
							origin.y - (pointNext._position.y - range.x) / (range.y - range.x) * plotSize.y);

						sf::Vector2f renderDirection = vectorNormalize(renderPointNext - renderPoint);

						sf::Vector2f sizeOffset;
						sf::Vector2f sizeOffsetNext;

						if (p > 0) {
							sf::Vector2f renderPointPrev = sf::Vector2f(origin.x + (_curves[c]._points[p - 1]._position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
								origin.y - (_curves[c]._points[p - 1]._position.y - range.x) / (range.y - range.x) * plotSize.y);

							sf::Vector2f averageDirection = (renderDirection + vectorNormalize(renderPoint - renderPointPrev)) * 0.5f;
							
							sizeOffset = vectorNormalize(sf::Vector2f(-averageDirection.y, averageDirection.x));
						}
						else
							sizeOffset = vectorNormalize(sf::Vector2f(-renderDirection.y, renderDirection.x));

						if (p < _curves[c]._points.size() - 2) {
							sf::Vector2f renderPointNextNext = sf::Vector2f(origin.x + (_curves[c]._points[p + 2]._position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
								origin.y - (_curves[c]._points[p + 2]._position.y - range.x) / (range.y - range.x) * plotSize.y);

							sf::Vector2f averageDirection = (renderDirection + vectorNormalize(renderPointNextNext - renderPointNext)) * 0.5f;

							sizeOffsetNext = vectorNormalize(sf::Vector2f(-averageDirection.y, averageDirection.x));
						}
						else
							sizeOffsetNext = vectorNormalize(sf::Vector2f(-renderDirection.y, renderDirection.x));

						sf::Vector2f perpendicular = vectorNormalize(sf::Vector2f(-renderDirection.y, renderDirection.x));

						sizeOffset *= 1.0f / vectorDot(perpendicular, sizeOffset) * lineSize * 0.5f;
						sizeOffsetNext *= 1.0f / vectorDot(perpendicular, sizeOffsetNext) * lineSize * 0.5f;

						vertexArray[index].position = renderPoint - sizeOffset;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, 0.0f);
						vertexArray[index].color = point._color;

						index++;

						vertexArray[index].position = renderPointNext - sizeOffsetNext;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, 0.0f);
						vertexArray[index].color = pointNext._color;

						index++;

						vertexArray[index].position = renderPointNext + sizeOffsetNext;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, lineGradientTexture.getSize().y);
						vertexArray[index].color = pointNext._color;

						index++;

						vertexArray[index].position = renderPoint - sizeOffset;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, 0.0f);
						vertexArray[index].color = point._color;

						index++;

						vertexArray[index].position = renderPointNext + sizeOffsetNext;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, lineGradientTexture.getSize().y);
						vertexArray[index].color = pointNext._color;

						index++;

						vertexArray[index].position = renderPoint + sizeOffset;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, lineGradientTexture.getSize().y);
						vertexArray[index].color = point._color;

						index++;
					}
				}

				vertexArray.resize(index);

				vertexArray.setPrimitiveType(sf::PrimitiveType::Triangles);

				if (_curves[c]._shadow != 0.0f) {
					sf::VertexArray shadowArray = vertexArray;

					for (int v = 0; v < shadowArray.getVertexCount(); v++) {
						shadowArray[v].position += _curves[c]._shadowOffset;
						shadowArray[v].color = sf::Color(0, 0, 0, _curves[c]._shadow * 255.0f);
					}

					target.draw(shadowArray, &lineGradientTexture);
				}

				target.draw(vertexArray, &lineGradientTexture);
			}

			// Mask off parts of the curve that go beyond bounds
			sf::RectangleShape leftMask;
			leftMask.setSize(sf::Vector2f(margins.x, target.getSize().y));
			leftMask.setFillColor(_backgroundColor);

			target.draw(leftMask);

			sf::RectangleShape rightMask;
			rightMask.setSize(sf::Vector2f(target.getSize().x, margins.y));
			rightMask.setPosition(sf::Vector2f(0.0f, target.getSize().y - margins.y));
			rightMask.setFillColor(_backgroundColor);

			target.draw(rightMask);

			// Draw axes
			sf::RectangleShape xAxis;
			xAxis.setSize(sf::Vector2f(plotSize.x + axesSize * 0.5f, axesSize));
			xAxis.setPosition(sf::Vector2f(origin.x - axesSize * 0.5f, origin.y - axesSize * 0.5f));
			xAxis.setFillColor(_axesColor);

			target.draw(xAxis);

			sf::RectangleShape yAxis;
			yAxis.setSize(sf::Vector2f(axesSize, plotSize.y + axesSize * 0.5f));
			yAxis.setPosition(sf::Vector2f(origin.x - axesSize * 0.5f, origin.y - axesSize * 0.5f - plotSize.y));
			yAxis.setFillColor(_axesColor);

			target.draw(yAxis);

			// BINH: draw firstIndex
			std::ostringstream os1;
			os1.str(" ");
			if (firstIndex >= 0) os1 << firstIndex;
			sf::Text titleText;
			titleText.setString(os1.str());
			titleText.setFont(tickFont);
			titleText.setPosition(sf::Vector2f(origin.x - axesSize * 0.5f, origin.y));
			titleText.setColor(_axesColor);
			target.draw(titleText);

			// Draw ticks
			{
				float xDistance = domain.y - domain.x;
				int xTicks = std::floor(xDistance / tickIncrements.x);
				float xTickOffset = std::fmod(domain.x, tickIncrements.x);

				if (xTickOffset < 0.0f)
					xTickOffset += tickIncrements.x;

				float xTickRenderOffset = xTickOffset / xDistance;

				float xTickRenderDistance = tickIncrements.x / xDistance * plotSize.x;

				std::ostringstream os;

				os.precision(precision);

				for (int t = 0; t < xTicks; t++) {
					sf::RectangleShape xTick;
					xTick.setSize(sf::Vector2f(axesSize, tickLength));
					xTick.setPosition(sf::Vector2f(origin.x + xTickRenderOffset + xTickRenderDistance * t - tickSize * 0.5f, origin.y));
					xTick.setFillColor(_axesColor);

					target.draw(xTick);

					float value = domain.x + xTickOffset + t * tickIncrements.x;

					os.str("");
					os << value;

					sf::Text xTickText;
					xTickText.setString(os.str());
					xTickText.setFont(tickFont);
					xTickText.setPosition(sf::Vector2f(xTick.getPosition().x, xTick.getPosition().y + tickLength + textTickOffset));
					xTickText.setRotation(45.0f);
					xTickText.setColor(_axesColor);
					xTickText.setScale(sf::Vector2f(tickTextScale, tickTextScale));

					target.draw(xTickText);
				}
			}

			{
				float yDistance = range.y - range.x;
				int yTicks = std::floor(yDistance / tickIncrements.y);
				float yTickOffset = std::fmod(range.x, tickIncrements.y);

				if (yTickOffset < 0.0f)
					yTickOffset += tickIncrements.y;

				float yTickRenderOffset = yTickOffset / yDistance;

				float yTickRenderDistance = tickIncrements.y / yDistance * plotSize.y;

				std::ostringstream os;

				os.precision(precision);

				for (int t = 0; t < yTicks; t++) {
					sf::RectangleShape yTick;
					yTick.setSize(sf::Vector2f(tickLength, axesSize));
					yTick.setPosition(sf::Vector2f(origin.x - tickLength, origin.y - yTickRenderOffset - yTickRenderDistance * t - tickSize * 0.5f));
					yTick.setFillColor(_axesColor);

					target.draw(yTick);

					float value = range.x + yTickOffset + t * tickIncrements.y;

					os.str("");
					os << value;

					sf::Text yTickText;
					yTickText.setString(os.str());
					yTickText.setFont(tickFont);
					sf::FloatRect bounds = yTickText.getLocalBounds();
					yTickText.setPosition(sf::Vector2f(yTick.getPosition().x - bounds.width * 0.5f - tickLength * 0.5f - textTickOffset, yTick.getPosition().y - bounds.height * 0.5f));
					yTickText.setRotation(0.0f);
					yTickText.setColor(_axesColor);
					yTickText.setScale(sf::Vector2f(tickTextScale, tickTextScale));

					target.draw(yTickText);
				}
			}
		}

		void draw(sf::RenderWindow &renderWindow, const sf::Vector2f &range, const int nYTicks, float tickTextScale = 0.5f, float axesSize=2.f, float lineSize=4.f, float tickSize=2.f, float tickLength=6.f, float textTickOffset=2.f, int precision=4, int firstIndex=-1)
		{
			const int npts     = _curves[0]._points.size();
			const float ydelta = (range.y - range.x) / nYTicks; 
			const sf::Vector2f margins(64.0f, 64.0f);
			const sf::Vector2f tickIncrements(npts / 10.0f, ydelta);
			const sf::Vector2f domain(0.f, npts);

			_plotRT.clear(_backgroundColor);

			sf::Vector2f plotSize = sf::Vector2f(_plotRT.getSize().x - margins.x, _plotRT.getSize().y - margins.y);

			sf::Vector2f origin = sf::Vector2f(margins.x, _plotRT.getSize().y - margins.y);

			// Draw curves
			for (int c = 0; c < _curves.size(); c++)
			{
				if (_curves[c]._points.empty())
					continue;

				sf::VertexArray vertexArray;

				vertexArray.resize((_curves[c]._points.size() - 1) * 6);

				int index = 0;

				// Go through points
				for (int p = 0; p < _curves[c]._points.size() - 1; p++)
				{
					const Point &point     = _curves[c]._points[p];
					const Point &pointNext = _curves[c]._points[p + 1];

					sf::Vector2f difference = pointNext._position - point._position;
					sf::Vector2f direction  = vectorNormalize(difference);

					sf::Vector2f renderPointFirst, renderPointSecond;

					bool pointVisible = point._position.x >= domain.x && point._position.x <= domain.y &&
						point._position.y >= range.x && point._position.y <= range.y;

					bool pointNextVisible = pointNext._position.x >= domain.x && pointNext._position.x <= domain.y &&
						pointNext._position.y >= range.x && pointNext._position.y <= range.y;

					if (pointVisible || pointNextVisible)
					{
						sf::Vector2f renderPoint = sf::Vector2f(origin.x + (point._position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
							origin.y - (point._position.y - range.x) / (range.y - range.x) * plotSize.y);

						sf::Vector2f renderPointNext = sf::Vector2f(origin.x + (pointNext._position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
							origin.y - (pointNext._position.y - range.x) / (range.y - range.x) * plotSize.y);

						sf::Vector2f renderDirection = vectorNormalize(renderPointNext - renderPoint);

						sf::Vector2f sizeOffset;
						sf::Vector2f sizeOffsetNext;

						if (p > 0) {
							sf::Vector2f renderPointPrev = sf::Vector2f(origin.x + (_curves[c]._points[p - 1]._position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
								origin.y - (_curves[c]._points[p - 1]._position.y - range.x) / (range.y - range.x) * plotSize.y);

							sf::Vector2f averageDirection = (renderDirection + vectorNormalize(renderPoint - renderPointPrev)) * 0.5f;
							
							sizeOffset = vectorNormalize(sf::Vector2f(-averageDirection.y, averageDirection.x));
						}
						else
							sizeOffset = vectorNormalize(sf::Vector2f(-renderDirection.y, renderDirection.x));

						if (p < _curves[c]._points.size() - 2) {
							sf::Vector2f renderPointNextNext = sf::Vector2f(origin.x + (_curves[c]._points[p + 2]._position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
								origin.y - (_curves[c]._points[p + 2]._position.y - range.x) / (range.y - range.x) * plotSize.y);

							sf::Vector2f averageDirection = (renderDirection + vectorNormalize(renderPointNextNext - renderPointNext)) * 0.5f;

							sizeOffsetNext = vectorNormalize(sf::Vector2f(-averageDirection.y, averageDirection.x));
						}
						else
							sizeOffsetNext = vectorNormalize(sf::Vector2f(-renderDirection.y, renderDirection.x));

						sf::Vector2f perpendicular = vectorNormalize(sf::Vector2f(-renderDirection.y, renderDirection.x));

						sizeOffset *= 1.0f / vectorDot(perpendicular, sizeOffset) * lineSize * 0.5f;
						sizeOffsetNext *= 1.0f / vectorDot(perpendicular, sizeOffsetNext) * lineSize * 0.5f;

						vertexArray[index].position = renderPoint - sizeOffset;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, 0.0f);
						vertexArray[index].color = point._color;

						index++;

						vertexArray[index].position = renderPointNext - sizeOffsetNext;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, 0.0f);
						vertexArray[index].color = pointNext._color;

						index++;

						vertexArray[index].position = renderPointNext + sizeOffsetNext;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, _lineGradient.getSize().y);
						vertexArray[index].color = pointNext._color;

						index++;

						vertexArray[index].position = renderPoint - sizeOffset;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, 0.0f);
						vertexArray[index].color = point._color;

						index++;

						vertexArray[index].position = renderPointNext + sizeOffsetNext;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, _lineGradient.getSize().y);
						vertexArray[index].color = pointNext._color;

						index++;

						vertexArray[index].position = renderPoint + sizeOffset;
						vertexArray[index].texCoords = sf::Vector2f(0.0f, _lineGradient.getSize().y);
						vertexArray[index].color = point._color;

						index++;
					}
				}

				vertexArray.resize(index);

				vertexArray.setPrimitiveType(sf::PrimitiveType::Triangles);

				if (_curves[c]._shadow != 0.0f) {
					sf::VertexArray shadowArray = vertexArray;

					for (int v = 0; v < shadowArray.getVertexCount(); v++) {
						shadowArray[v].position += _curves[c]._shadowOffset;
						shadowArray[v].color = sf::Color(0, 0, 0, _curves[c]._shadow * 255.0f);
					}

					_plotRT.draw(shadowArray, &_lineGradient);
				}

				_plotRT.draw(vertexArray, &_lineGradient);
			}

			// Mask off parts of the curve that go beyond bounds
			sf::RectangleShape leftMask;
			leftMask.setSize(sf::Vector2f(margins.x, _plotRT.getSize().y));
			leftMask.setFillColor(_backgroundColor);

			_plotRT.draw(leftMask);

			sf::RectangleShape rightMask;
			rightMask.setSize(sf::Vector2f(_plotRT.getSize().x, margins.y));
			rightMask.setPosition(sf::Vector2f(0.0f, _plotRT.getSize().y - margins.y));
			rightMask.setFillColor(_backgroundColor);

			_plotRT.draw(rightMask);

			// Draw axes
			sf::RectangleShape xAxis;
			xAxis.setSize(sf::Vector2f(plotSize.x + axesSize * 0.5f, axesSize));
			xAxis.setPosition(sf::Vector2f(origin.x - axesSize * 0.5f, origin.y - axesSize * 0.5f));
			xAxis.setFillColor(_axesColor);

			_plotRT.draw(xAxis);

			sf::RectangleShape yAxis;
			yAxis.setSize(sf::Vector2f(axesSize, plotSize.y + axesSize * 0.5f));
			yAxis.setPosition(sf::Vector2f(origin.x - axesSize * 0.5f, origin.y - axesSize * 0.5f - plotSize.y));
			yAxis.setFillColor(_axesColor);

			_plotRT.draw(yAxis);

			// BINH: draw firstIndex
			std::ostringstream os1;
			os1.str(" ");
			if (firstIndex >= 0) os1 << firstIndex;
			sf::Text titleText;
			titleText.setString(os1.str());
			titleText.setFont(_tickFont);
			titleText.setPosition(sf::Vector2f(origin.x - axesSize * 0.5f, origin.y));
			titleText.setColor(_axesColor);
			_plotRT.draw(titleText);

			// Draw ticks
			{
				float xDistance = domain.y - domain.x;
				int xTicks = std::floor(xDistance / tickIncrements.x);
				float xTickOffset = std::fmod(domain.x, tickIncrements.x);

				if (xTickOffset < 0.0f)
					xTickOffset += tickIncrements.x;

				float xTickRenderOffset = xTickOffset / xDistance;

				float xTickRenderDistance = tickIncrements.x / xDistance * plotSize.x;

				std::ostringstream os;

				os.precision(precision);

				for (int t = 0; t < xTicks; t++) {
					sf::RectangleShape xTick;
					xTick.setSize(sf::Vector2f(axesSize, tickLength));
					xTick.setPosition(sf::Vector2f(origin.x + xTickRenderOffset + xTickRenderDistance * t - tickSize * 0.5f, origin.y));
					xTick.setFillColor(_axesColor);

					_plotRT.draw(xTick);

					float value = domain.x + xTickOffset + t * tickIncrements.x;

					os.str("");
					os << value;

					sf::Text xTickText;
					xTickText.setString(os.str());
					xTickText.setFont(_tickFont);
					xTickText.setPosition(sf::Vector2f(xTick.getPosition().x, xTick.getPosition().y + tickLength + textTickOffset));
					//xTickText.setRotation(45.0f);
					xTickText.setColor(_axesColor);
					xTickText.setScale(sf::Vector2f(tickTextScale, tickTextScale));

					_plotRT.draw(xTickText);
				}
			}

			{
				float yDistance = range.y - range.x;
				int yTicks = std::floor(yDistance / tickIncrements.y);
				float yTickOffset = std::fmod(range.x, tickIncrements.y);

				if (yTickOffset < 0.0f)
					yTickOffset += tickIncrements.y;

				float yTickRenderOffset = yTickOffset / yDistance;

				float yTickRenderDistance = tickIncrements.y / yDistance * plotSize.y;

				std::ostringstream os;

				os.precision(precision);

				for (int t = 0; t < yTicks; t++) {
					sf::RectangleShape yTick;
					yTick.setSize(sf::Vector2f(tickLength, axesSize));
					yTick.setPosition(sf::Vector2f(origin.x - tickLength, origin.y - yTickRenderOffset - yTickRenderDistance * t - tickSize * 0.5f));
					yTick.setFillColor(_axesColor);

					_plotRT.draw(yTick);

					float value = range.x + yTickOffset + t * tickIncrements.y;

					os.str("");
					os << value;

					sf::Text yTickText;
					yTickText.setString(os.str());
					yTickText.setFont(_tickFont);
					sf::FloatRect bounds = yTickText.getLocalBounds();
					yTickText.setPosition(sf::Vector2f(yTick.getPosition().x - bounds.width * 0.5f - tickLength * 0.5f - textTickOffset, yTick.getPosition().y - bounds.height * 0.5f));
					yTickText.setRotation(0.0f);
					yTickText.setColor(_axesColor);
					yTickText.setScale(sf::Vector2f(tickTextScale, tickTextScale));

					_plotRT.draw(yTickText);
				}
			}

			// update
			_plotRT.display();
			_plotSprite.setTexture(_plotRT.getTexture());
			renderWindow.draw(_plotSprite);
			renderWindow.draw(_xlabel);
			renderWindow.draw(_ylabel);
		}
	};// end plot

} // namespace

#endif
