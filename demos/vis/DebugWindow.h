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

#include <neo/Hierarchy.h>
#include <neo/Agent.h>

namespace vis {
    class DebugWindow {
    private:
        class TextureWindow {
        public:
            sf::Texture* texture;
            sf::Sprite sprite;
            sf::Text text;
        };

        sf::RenderWindow _window;

        sf::Font _debugFont;

        bool _catchUnitRanges;
        float _globalScale;
        std::vector<TextureWindow> _textureWindows;

        std::shared_ptr<ogmaneo::Resources> res;
        std::shared_ptr<ogmaneo::Hierarchy> h;
        std::shared_ptr<ogmaneo::Agent> a;

        void addImage(const std::string& title, const cl::Image2D& img, const sf::Vector2f& offset, const sf::Vector2f& scale = sf::Vector2f(1.f, 1.f));
        void addValueField2D(const std::string& title, const ogmaneo::ValueField2D& valueField, const sf::Vector2f& offset, const sf::Vector2f& scale = sf::Vector2f(1.f, 1.f));

        void updateTextureFromImage2D(const cl::Image2D &img, sf::Texture *texture);
        void updateTextureFromValueField2D(ogmaneo::ValueField2D &valueField, sf::Texture *texture);


    public:
        DebugWindow() :
            _catchUnitRanges(false), _globalScale(1.0f) {
        }

        ~DebugWindow() {
            _textureWindows.clear();
        }

        void create(const sf::String& title, const sf::Vector2i& pos, const sf::Vector2u& size);
        
        void registerHierarchy(std::shared_ptr<ogmaneo::Resources> resources, std::shared_ptr<ogmaneo::Hierarchy> hierarchy);
        void registerAgent(std::shared_ptr<ogmaneo::Resources> resources, std::shared_ptr<ogmaneo::Agent> agent);

        void update();
        void display(const sf::Color& clearColor = sf::Color(64, 64, 64, 255));

        void catchUnitRanges(bool newValue) {
            _catchUnitRanges = newValue;
        }
    };
};