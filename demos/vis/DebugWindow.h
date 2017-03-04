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

#include <neo/FeatureHierarchy.h>
#include <neo/Hierarchy.h>

namespace vis {
    class DebugWindow {
    private:
        class TextureWindow {
        public:
            template<class T> class MinMaxValues {
            public:
                T min, max;
            };

            sf::Texture* texture;
            sf::Sprite sprite;
            sf::Text text;
            union {
                MinMaxValues<float> f;
                MinMaxValues<int> i;
            } minmax;
        };

        sf::RenderWindow _window;

        sf::Font _debugFont;

        bool _catchUnitRanges;
        float _globalScale;
        std::vector<TextureWindow> _textureWindows;

        float _maxTileSize;
        float _xSeperation;
        float _ySeperation;

        float _xOffset;
        float _yOffset;

        int _windowIndex;

        std::shared_ptr<ogmaneo::Resources> res;
        std::shared_ptr<ogmaneo::Hierarchy> h;

        void addImage(const std::string& title, const cl::Image2D& img, const sf::Vector2f& offset, const sf::Vector2f& scale = sf::Vector2f(1.f, 1.f));
        void addValueField2D(const std::string& title, const ogmaneo::ValueField2D& valueField, const sf::Vector2f& offset, const sf::Vector2f& scale = sf::Vector2f(1.f, 1.f));

        void updateTextureFromImage2D(const cl::Image2D &img, sf::Texture *texture);
        void updateTextureFromValueField2D(ogmaneo::ValueField2D &valueField, sf::Texture *texture);

    public:
        DebugWindow() :
            _catchUnitRanges(false), _globalScale(1.0f), _maxTileSize(256.0f), _xSeperation(16.0f), _ySeperation(8.0f) {
        }

        ~DebugWindow() {
            _textureWindows.clear();
        }

        void create(const sf::String& title, const sf::Vector2i& pos, const sf::Vector2u& size);

        void registerHierarchy(std::shared_ptr<ogmaneo::Resources> resources, std::shared_ptr<ogmaneo::Hierarchy> hierarchy);

        void update();
        void display(const sf::Color& clearColor = sf::Color(64, 64, 64, 255));

        void catchUnitRanges(bool newValue) {
            _catchUnitRanges = newValue;
        }
    };
};