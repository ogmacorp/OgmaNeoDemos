// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <algorithm>

#include "DebugWindow.h"

#include <neo/SparseFeaturesChunk.h>
#include <neo/SparseFeaturesDistance.h>

using namespace vis;
using namespace ogmaneo;


void DebugWindow::create(const sf::String& title, const sf::Vector2i& pos, const sf::Vector2u& size) {
    _window.create(sf::VideoMode(size.x, size.y), title, sf::Style::Default);
    _window.setPosition(pos);
    _window.setFramerateLimit(0);

#if defined(_WINDOWS)
    _debugFont.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#elif defined(__APPLE__)
    _debugFont.loadFromFile("/Library/Fonts/Courier New.ttf");
#else
    _debugFont.loadFromFile("/usr/share/fonts/truetype/freefont/FreeMono.ttf");
#endif
}


void DebugWindow::addImage(const std::string& title, const cl::Image2D& img, const sf::Vector2f& offset, const sf::Vector2f& scale) {
    unsigned int width = (unsigned int)img.getImageInfo<CL_IMAGE_WIDTH>();
    unsigned int height = (unsigned int)img.getImageInfo<CL_IMAGE_HEIGHT>();

    TextureWindow rtw;

    rtw.texture = new sf::Texture();
    rtw.texture->create(width, height);

    rtw.sprite.setPosition(offset.x, offset.y);
    rtw.sprite.setScale(scale.x * _globalScale, scale.y * _globalScale);

    rtw.text.setFont(_debugFont);
    rtw.text.setCharacterSize(12);
    rtw.text.setString(title);
    rtw.text.setPosition(offset.x, offset.y);
    rtw.text.setPosition(offset.x, offset.y - rtw.text.getCharacterSize() - 4.f);
    rtw.text.setFillColor(sf::Color::White);

    _textureWindows.push_back(rtw);
}


void DebugWindow::addValueField2D(const std::string& title, const ValueField2D& valueField, const sf::Vector2f& offset, const sf::Vector2f& scale) {
    unsigned int width = (unsigned int)valueField.getSize().x;
    unsigned int height = (unsigned int)valueField.getSize().y;

    TextureWindow rtw;

    rtw.texture = new sf::Texture();
    rtw.texture->create(width, height);

    rtw.sprite.setPosition(offset.x, offset.y);
    rtw.sprite.setScale(scale.x * _globalScale, scale.y * _globalScale);

    rtw.text.setFont(_debugFont);
    rtw.text.setCharacterSize(12);
    rtw.text.setString(title);
    rtw.text.setPosition(offset.x, offset.y - rtw.text.getCharacterSize() - 4.f);
    rtw.text.setFillColor(sf::Color::White);

    _textureWindows.push_back(rtw);
}


void DebugWindow::registerHierarchy(std::shared_ptr<Resources> resources, std::shared_ptr<Hierarchy> hierarchy) {
    h = hierarchy;
    res = resources;

    Predictor& predictor = h->getPredictor();
    FeatureHierarchy& featureHierarchy = predictor.getHierarchy();

    const std::vector<cl::Image2D> &inputImages = h->getInputImagesFeed();

    float inputWidth = (float)inputImages[0].getImageInfo<CL_IMAGE_WIDTH>();
    float inputHeight = (float)inputImages[0].getImageInfo<CL_IMAGE_HEIGHT>();

    float maxTileSize = 256.f;
    _globalScale = (maxTileSize - inputWidth) / maxTileSize;

    float xSeperation = 16.f;
    float ySeperation = 8.f;

    float xOffset = 8.f;
    float yOffset = (float)_window.getSize().y - (inputHeight * _globalScale) - ySeperation;

    std::string title;

    for (size_t i = 0; i < inputImages.size(); i++) {
        const cl::Image2D& img = inputImages[i];

        title = "Input:" + std::to_string(i);
        addImage(title, img, sf::Vector2f(xOffset, yOffset));
        xOffset += (float)inputWidth * _globalScale + xSeperation;
    }
    yOffset -= ySeperation;

    _globalScale = 1.0f;

    for (int i = 0; i < (int)featureHierarchy.getNumLayers(); i++) {
        const FeatureHierarchy::Layer& featureHierarchyLayer = featureHierarchy.getLayer(i);
        std::shared_ptr<SparseFeatures> sparseFeatures = featureHierarchyLayer._sf;

        xOffset = 8.f;

        switch (sparseFeatures->_type) {
        default:
        case SparseFeaturesType::_chunk: {
            SparseFeaturesChunk* sparseFeaturesChunk = reinterpret_cast<SparseFeaturesChunk*>(sparseFeatures.get());

            const cl::Image2D& states = sparseFeaturesChunk->getHiddenStates()[_back];
            unsigned int width = (unsigned int)states.getImageInfo<CL_IMAGE_WIDTH>();
            unsigned int height = (unsigned int)states.getImageInfo<CL_IMAGE_HEIGHT>();

            _globalScale = (maxTileSize - width) / maxTileSize;

            yOffset -= (float)height * _globalScale + ySeperation;

            title = "States:" + std::to_string(i);
            addImage(title, states, sf::Vector2f(xOffset, yOffset));
            xOffset += (float)width * _globalScale + xSeperation;

            const cl::Image2D& activations = sparseFeaturesChunk->getHiddenActivations()[_back];
            width = (unsigned int)activations.getImageInfo<CL_IMAGE_WIDTH>();

            title = "Activations:" + std::to_string(i);
            addImage(title, activations, sf::Vector2f(xOffset, yOffset));
            xOffset += (float)width * _globalScale + xSeperation;

            const cl::Image2D& winners = sparseFeaturesChunk->getChunkWinners()[_back];
            width = (unsigned int)winners.getImageInfo<CL_IMAGE_WIDTH>();

            title = "Winners:" + std::to_string(i);
            addImage(title, winners, sf::Vector2f(xOffset, yOffset), sf::Vector2f(4.f, 4.f));
            xOffset += (float)width * _globalScale + xSeperation;
            break;
        }

        case SparseFeaturesType::_distance: {
            SparseFeaturesDistance* sparseFeaturesDistance = reinterpret_cast<SparseFeaturesDistance*>(sparseFeatures.get());

            const cl::Image2D& states = sparseFeaturesDistance->getHiddenStates()[_back];
            unsigned int width = (unsigned int)states.getImageInfo<CL_IMAGE_WIDTH>();
            unsigned int height = (unsigned int)states.getImageInfo<CL_IMAGE_HEIGHT>();

            _globalScale = (maxTileSize - width) / maxTileSize;

            yOffset -= (float)height * _globalScale + ySeperation;

            title = "States:" + std::to_string(i);
            addImage(title, states, sf::Vector2f(xOffset, yOffset));
            xOffset += (float)width * _globalScale + xSeperation;

            const cl::Image2D& activations = sparseFeaturesDistance->getHiddenActivations()[_back];
            width = (unsigned int)activations.getImageInfo<CL_IMAGE_WIDTH>();

            title = "Activations:" + std::to_string(i);
            addImage(title, activations, sf::Vector2f(xOffset, yOffset));
            xOffset += (float)width * _globalScale + xSeperation;

            const cl::Image2D& winners = sparseFeaturesDistance->getDistanceWinners()[_back];
            width = (unsigned int)winners.getImageInfo<CL_IMAGE_WIDTH>();

            title = "Winners:" + std::to_string(i);
            addImage(title, winners, sf::Vector2f(xOffset, yOffset));
            xOffset += (float)width * _globalScale + xSeperation;
            break;
        }
        }

        yOffset -= ySeperation;
    }

    const cl::Image2D& states = predictor.getPredLayer(0)[0].getHiddenStates()[_back];

    unsigned int predictionWidth = (unsigned int)states.getImageInfo<CL_IMAGE_WIDTH>();
    unsigned int predictionHeight = (unsigned int)states.getImageInfo<CL_IMAGE_HEIGHT>();

    _globalScale = (maxTileSize - predictionWidth) / maxTileSize;

    yOffset = (float)_window.getSize().y - (predictionHeight * _globalScale) - ySeperation;

    for (int p = 0; p < (int)predictor.getNumPredLayers(); p++) {
        const std::vector<PredictorLayer>& predictorLayers = predictor.getPredLayer(p);

        xOffset = ((float)_window.getSize().x / 2.f);

        for (int i = 0; i < (int)predictorLayers.size(); i++) {
            const cl::Image2D& states = predictorLayers[i].getHiddenStates()[_back];
            unsigned int width = (unsigned int)states.getImageInfo<CL_IMAGE_WIDTH>();
            unsigned int height = (unsigned int)states.getImageInfo<CL_IMAGE_HEIGHT>();

            _globalScale = (maxTileSize - width) / maxTileSize;

            title = "States:" + std::to_string(p) + "-" + std::to_string(i);
            addImage(title, states, sf::Vector2f(xOffset, yOffset));

            xOffset += (float)width * _globalScale + xSeperation;
        }

        yOffset -= (float)predictionHeight * _globalScale + ySeperation;
    }

}


void DebugWindow::update() {
    if (!_window.isOpen())
        return;

    sf::Event windowEvent;

    while (_window.pollEvent(windowEvent))
    {
        switch (windowEvent.type)
        {
        case sf::Event::Closed:
            _window.close();
            break;
        case sf::Event::Resized:
            break;
        default:
            break;
        }
    }

    int windowIndex = 0;

    const std::vector<cl::Image2D> &inputImages = h->getInputImagesFeed();
    for (size_t i = 0; i < inputImages.size(); i++) {
        const cl::Image2D &img = inputImages[i];

        updateTextureFromImage2D(img, _textureWindows[windowIndex++].texture);
    }

    Predictor& predictor = h->getPredictor();
    FeatureHierarchy& featureHierarchy = predictor.getHierarchy();

    for (int i = 0; i < (int)featureHierarchy.getNumLayers(); i++) {
        const FeatureHierarchy::Layer& featureHierarchyLayer = featureHierarchy.getLayer(i);
        std::shared_ptr<SparseFeatures> sparseFeatures = featureHierarchyLayer._sf;

        switch (sparseFeatures->_type) {
        default:
        case SparseFeaturesType::_chunk: {
            SparseFeaturesChunk* sparseFeaturesChunk = reinterpret_cast<SparseFeaturesChunk*>(sparseFeatures.get());

            cl::Image2D states = sparseFeaturesChunk->getHiddenStates()[_back];
            updateTextureFromImage2D(states, _textureWindows[windowIndex++].texture);

            cl::Image2D activations = sparseFeaturesChunk->getHiddenActivations()[_back];
            updateTextureFromImage2D(activations, _textureWindows[windowIndex++].texture);

            cl::Image2D winners = sparseFeaturesChunk->getChunkWinners()[_back];
            updateTextureFromImage2D(winners, _textureWindows[windowIndex++].texture);
            break;
        }

        case SparseFeaturesType::_distance: {
            SparseFeaturesDistance* sparseFeaturesDistance = reinterpret_cast<SparseFeaturesDistance*>(sparseFeatures.get());

            cl::Image2D states = sparseFeaturesDistance->getHiddenStates()[_back];
            updateTextureFromImage2D(states, _textureWindows[windowIndex++].texture);

            cl::Image2D activations = sparseFeaturesDistance->getHiddenActivations()[_back];
            updateTextureFromImage2D(activations, _textureWindows[windowIndex++].texture);

            cl::Image2D winners = sparseFeaturesDistance->getDistanceWinners()[_back];
            updateTextureFromImage2D(winners, _textureWindows[windowIndex++].texture);
            break;
        }
        }
    }

    for (int p = 0; p < (int)predictor.getNumPredLayers(); p++) {
        for (int i = 0; i < (int)predictor.getPredLayer(p).size(); i++) {
            const PredictorLayer& predictorLayer = predictor.getPredLayer(p)[i];

            const cl::Image2D& states = predictorLayer.getHiddenStates()[_back];
            updateTextureFromImage2D(states, _textureWindows[windowIndex++].texture);
        }
    }
}


void DebugWindow::display(const sf::Color& clearColor) {
    if (!_window.isOpen())
        return;

    _window.clear(clearColor);

    for (TextureWindow& rtw : _textureWindows) {
        _window.draw(rtw.text);

        rtw.sprite.setTexture(*rtw.texture);
        _window.draw(rtw.sprite);
    }

    _window.display();
}


void DebugWindow::updateTextureFromImage2D(const cl::Image2D &img, sf::Texture *texture) {

    uint32_t width = (uint32_t)img.getImageInfo<CL_IMAGE_WIDTH>();
    uint32_t height = (uint32_t)img.getImageInfo<CL_IMAGE_HEIGHT>();
    uint32_t elementSize = (uint32_t)img.getImageInfo<CL_IMAGE_ELEMENT_SIZE>();

    cl_channel_order channelOrder = img.getImageInfo<CL_IMAGE_FORMAT>().image_channel_order;
    cl_channel_type channelType = img.getImageInfo<CL_IMAGE_FORMAT>().image_channel_data_type;

    switch (channelType) {
    case CL_FLOAT:
    {
        std::vector<float> pixels(width * height * (elementSize / sizeof(float)), 0.0f);
        res->getComputeSystem()->getQueue().enqueueReadImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, 1 }, 0, 0, pixels.data());
        res->getComputeSystem()->getQueue().finish();

        float minValue = *min_element(pixels.begin(), pixels.end());
        float maxValue = *max_element(pixels.begin(), pixels.end());

        float offset = minValue;
        float scale = 1.f / (maxValue - minValue);

        if (_catchUnitRanges) {
            // Catch [-1, 1]
            if (minValue >= -1.0f && minValue < 0.0f &&
                maxValue <= 1.0f && maxValue > 0.0f) {
                offset = 1.f;
                scale = 0.5f;
            }
            // Catch [0, 1]
            if (minValue >= 0.0f && maxValue <= 1.0f) {
                offset = 0.0f;
                scale = 1.0f;
            }
        }

        uint32_t elementsPerPixel = (elementSize / sizeof(float));
        uint32_t numElements = width * height * elementsPerPixel;
        float pixelValue;
        sf::Color pixel;

        sf::Image newImg;
        newImg.create(width, height);

        for (uint32_t i = 0; i < numElements; i += elementsPerPixel) {
            for (uint32_t j = 0; j < elementsPerPixel; j++) {
                pixelValue = (pixels[i + j] + offset) * scale;

                switch (j) {
                default:
                case 0: pixel.r = (sf::Uint8)(pixelValue * 255.f); break;
                case 1: pixel.g = (sf::Uint8)(pixelValue * 255.f); break;
                case 2: pixel.b = (sf::Uint8)(pixelValue * 255.f); break;
                case 3: pixel.a = (sf::Uint8)(pixelValue * 255.f); break;
                }
            }

            newImg.setPixel((i / elementsPerPixel) % width, (i / elementsPerPixel) / width, pixel);
        }

        texture->loadFromImage(newImg);
        break;
    }
    case CL_UNSIGNED_INT8:
    case CL_SIGNED_INT8:
    {
        std::vector<unsigned char> pixels(width * height * (elementSize / sizeof(unsigned char)), 0);
        res->getComputeSystem()->getQueue().enqueueReadImage(img, CL_TRUE, { 0, 0, 0 }, { width, height, 1 }, 0, 0, pixels.data());
        res->getComputeSystem()->getQueue().finish();

        unsigned char minValue = *min_element(pixels.begin(), pixels.end());
        unsigned char maxValue = *max_element(pixels.begin(), pixels.end());

        float offset = (float)minValue;
        float scale = 255.f / (float)(maxValue - minValue);

        uint32_t elementsPerPixel = (elementSize / sizeof(unsigned char));
        uint32_t numElements = width * height * elementsPerPixel;
        sf::Uint8 pixelValue;
        sf::Color pixel;

        sf::Image newImg;
        newImg.create(width, height);

        for (uint32_t i = 0; i < numElements; i += elementsPerPixel) {
            for (uint32_t j = 0; j < elementsPerPixel; j++) {
                pixelValue = (sf::Uint8)(((float)pixels[i + j] + offset) * scale);

                switch (j) {
                default:
                case 0: pixel.r = pixelValue; break;
                case 1: pixel.g = pixelValue; break;
                case 2: pixel.b = pixelValue; break;
                case 3: pixel.a = pixelValue; break;
                }
            }

            newImg.setPixel((i / elementsPerPixel) % width, (i / elementsPerPixel) / width, pixel);
        }

        texture->loadFromImage(newImg);
        break;
    }
    default:
        assert(0);
        break;
    }
}

void DebugWindow::updateTextureFromValueField2D(ValueField2D &valueField, sf::Texture *texture) {
    const Vec2i size = valueField.getSize();

    sf::Color pixel;
    sf::Image img;
    img.create(size.x, size.y);

    uint32_t numElements = size.x * size.y;
    float pixelValue;

    float minValue = *min_element(valueField.getData().begin(), valueField.getData().end());
    float maxValue = *max_element(valueField.getData().begin(), valueField.getData().end());

    float offset = minValue;
    float scale = 1.f / (maxValue - minValue);

    if (_catchUnitRanges) {
        // Catch [-1, 1]
        if (minValue >= -1.0f && minValue < 0.0f &&
            maxValue <= 1.0f && maxValue > 0.0f) {
            offset = 1.f;
            scale = 0.5f;
        }
        // Catch [0, 1]
        if (minValue >= 0.0f && maxValue <= 1.0f) {
            offset = 0.0f;
            scale = 1.0f;
        }
    }

    for (uint32_t i = 0; i < numElements; i++) {
        pixelValue = (valueField.getData()[i] + offset) * scale;

        pixel.r = (sf::Uint8)(pixelValue * 255.f);
        img.setPixel(i % size.x, i / size.x, pixel);
    }

    texture->loadFromImage(img);
}
