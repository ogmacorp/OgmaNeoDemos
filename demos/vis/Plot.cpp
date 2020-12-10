// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Plot.h"

#include <sstream>
#include <cmath>

using namespace vis;

void Plot::draw(
    sf::RenderTarget &target, const sf::Texture &lineGradientTexture, const sf::Font &tickFont, float tickTextScale,
    const sf::Vector2f &domain, const sf::Vector2f &range, const sf::Vector2f &margins, const sf::Vector2f &tickIncrements,
    float axesSize, float lineSize, float tickSize, float tickLength, float textTickOffset, int precision
) {
    target.clear(backgroundColor);

    sf::Vector2f plotSize = sf::Vector2f(target.getSize().x - (2.0f * margins.x), target.getSize().y - (2.0f * margins.y));

    sf::Vector2f origin = sf::Vector2f(margins.x, target.getSize().y - margins.y);

    if (plotBackgroundColor != backgroundColor) {
        sf::RectangleShape plotBackground;
        plotBackground.setSize(sf::Vector2f(plotSize.x, plotSize.y));
        plotBackground.setPosition(sf::Vector2f(margins.x, margins.y));
        plotBackground.setFillColor(plotBackgroundColor);

        target.draw(plotBackground);
    }

    // Draw curves
    for (int c = 0; c < curves.size(); c++) {
        if (curves[c].points.empty())
            continue;

        sf::VertexArray vertexArray;

        vertexArray.resize((curves[c].points.size() - 1) * 6);

        int index = 0;

        // Go through points
        for (int p = 0; p < curves[c].points.size() - 1; p++) {
            Point &point = curves[c].points[p];
            Point &pointNext = curves[c].points[p + 1];

            sf::Vector2f difference = pointNext.position - point.position;
            sf::Vector2f direction = vectorNormalize(difference);

            sf::Vector2f renderPointFirst, renderPointSecond;

            bool pointVisible = point.position.x >= domain.x && point.position.x <= domain.y &&
                point.position.y >= range.x && point.position.y <= range.y;

            bool pointNextVisible = pointNext.position.x >= domain.x && pointNext.position.x <= domain.y &&
                pointNext.position.y >= range.x && pointNext.position.y <= range.y;

            if (pointVisible || pointNextVisible) {
                sf::Vector2f renderPoint = sf::Vector2f(
                    origin.x + (point.position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
                    origin.y - (point.position.y - range.x) / (range.y - range.x) * plotSize.y
                );

                sf::Vector2f renderPointNext = sf::Vector2f(
                    origin.x + (pointNext.position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
                    origin.y - (pointNext.position.y - range.x) / (range.y - range.x) * plotSize.y
                );

                sf::Vector2f renderDirection = vectorNormalize(renderPointNext - renderPoint);

                sf::Vector2f sizeOffset;
                sf::Vector2f sizeOffsetNext;

                if (p > 0) {
                    sf::Vector2f renderPointPrev = sf::Vector2f(
                        origin.x + (curves[c].points[p - 1].position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
                        origin.y - (curves[c].points[p - 1].position.y - range.x) / (range.y - range.x) * plotSize.y
                    );

                    sf::Vector2f averageDirection = (renderDirection + vectorNormalize(renderPoint - renderPointPrev)) * 0.5f;
                    
                    sizeOffset = vectorNormalize(sf::Vector2f(-averageDirection.y, averageDirection.x));
                }
                else
                    sizeOffset = vectorNormalize(sf::Vector2f(-renderDirection.y, renderDirection.x));

                if (p < curves[c].points.size() - 2) {
                    sf::Vector2f renderPointNextNext = sf::Vector2f(
                        origin.x + (curves[c].points[p + 2].position.x - domain.x) / (domain.y - domain.x) * plotSize.x,
                        origin.y - (curves[c].points[p + 2].position.y - range.x) / (range.y - range.x) * plotSize.y
                    );

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
                vertexArray[index].color = point.color;

                index++;

                vertexArray[index].position = renderPointNext - sizeOffsetNext;
                vertexArray[index].texCoords = sf::Vector2f(0.0f, 0.0f);
                vertexArray[index].color = pointNext.color;

                index++;

                vertexArray[index].position = renderPointNext + sizeOffsetNext;
                vertexArray[index].texCoords = sf::Vector2f(0.0f, (float)lineGradientTexture.getSize().y);
                vertexArray[index].color = pointNext.color;

                index++;

                vertexArray[index].position = renderPoint - sizeOffset;
                vertexArray[index].texCoords = sf::Vector2f(0.0f, 0.0f);
                vertexArray[index].color = point.color;

                index++;

                vertexArray[index].position = renderPointNext + sizeOffsetNext;
                vertexArray[index].texCoords = sf::Vector2f(0.0f, (float)lineGradientTexture.getSize().y);
                vertexArray[index].color = pointNext.color;

                index++;

                vertexArray[index].position = renderPoint + sizeOffset;
                vertexArray[index].texCoords = sf::Vector2f(0.0f, (float)lineGradientTexture.getSize().y);
                vertexArray[index].color = point.color;

                index++;
            }
        }

        vertexArray.resize(index);

        vertexArray.setPrimitiveType(sf::PrimitiveType::Triangles);

        if (curves[c].shadow != 0.0f) {
            sf::VertexArray shadowArray = vertexArray;

            for (int v = 0; v < shadowArray.getVertexCount(); v++) {
                shadowArray[v].position += curves[c].shadowOffset;
                shadowArray[v].color = sf::Color(0, 0, 0, static_cast<sf::Uint8>(curves[c].shadow * 255.0f));
            }

            target.draw(shadowArray, &lineGradientTexture);
        }

        target.draw(vertexArray, &lineGradientTexture);
    }

    // Mask off parts of the curve that go beyond bounds
    sf::RectangleShape leftMask;
    leftMask.setSize(sf::Vector2f(margins.x, (float)target.getSize().y));
    leftMask.setFillColor(backgroundColor);

    target.draw(leftMask);

    sf::RectangleShape rightMask;
    rightMask.setSize(sf::Vector2f((float)target.getSize().x, margins.y));
    rightMask.setPosition(sf::Vector2f(0.0f, (float)(target.getSize().y - margins.y)));
    rightMask.setFillColor(backgroundColor);

    target.draw(rightMask);

    // Draw ticks
    if (plotXAxisTicks) {
        // Draw X axes
        sf::RectangleShape xAxis;
        xAxis.setSize(sf::Vector2f(plotSize.x + axesSize * 0.5f, axesSize));
        xAxis.setPosition(sf::Vector2f(origin.x - axesSize * 0.5f, origin.y - axesSize * 0.5f));
        xAxis.setFillColor(axesColor);

        target.draw(xAxis);

        float xDistance = domain.y - domain.x;
        int xTicks = (int)std::floor(xDistance / tickIncrements.x);
        float xTickOffset = 0.0f;// std::fmod(domain.x, tickIncrements.x);

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
            xTick.setFillColor(axesColor);

            target.draw(xTick);

            float value = domain.x + xTickOffset + t * tickIncrements.x;

            os.str("");
            os << value;

            sf::Text xTickText;
            xTickText.setString(os.str());
            xTickText.setFont(tickFont);
            xTickText.setPosition(sf::Vector2f(xTick.getPosition().x, xTick.getPosition().y + tickLength + textTickOffset));
            xTickText.setRotation(45.0f);
            xTickText.setFillColor(axesColor);
            xTickText.setScale(sf::Vector2f(tickTextScale, tickTextScale));

            target.draw(xTickText);
        }
    }

    {
        // Draw Y axes
        sf::RectangleShape yAxis;
        yAxis.setSize(sf::Vector2f(axesSize, plotSize.y + axesSize * 0.5f));
        yAxis.setPosition(sf::Vector2f(origin.x - axesSize * 0.5f, origin.y - axesSize * 0.5f - plotSize.y));
        yAxis.setFillColor(axesColor);

        target.draw(yAxis);

        float yDistance = range.y - range.x;
        int yTicks = (int)std::floor(yDistance / tickIncrements.y);
        float yTickOffset = 0.0f;// std::fmod(range.x, tickIncrements.y);

        if (yTickOffset < 0.0f)
            yTickOffset += tickIncrements.y;

        float yTickRenderOffset = yTickOffset / yDistance;

        float yTickRenderDistance = tickIncrements.y / yDistance * plotSize.y;

        std::ostringstream os;

        os.precision(precision);

        for (int t = 0; t < yTicks + 1; t++) {
            sf::RectangleShape yTick;
            yTick.setSize(sf::Vector2f(tickLength, axesSize));
            yTick.setPosition(sf::Vector2f(origin.x - tickLength, origin.y - yTickRenderOffset - yTickRenderDistance * t - tickSize * 0.5f));
            yTick.setFillColor(axesColor);

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
            yTickText.setFillColor(axesColor);
            yTickText.setScale(sf::Vector2f(tickTextScale, tickTextScale));

            target.draw(yTickText);
        }
    }
}

float vis::vectorMagnitude(const sf::Vector2f &vector) {
    return std::sqrt(vector.x * vector.x + vector.y * vector.y);
}

sf::Vector2f vis::vectorNormalize(const sf::Vector2f &vector) {
    float magnitude = vectorMagnitude(vector);

    return vector / magnitude;
}

float vis::vectorDot(const sf::Vector2f &left, const sf::Vector2f &right) {
    return left.x * right.x + left.y * right.y;
}
