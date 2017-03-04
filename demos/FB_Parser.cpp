// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#define _CRT_SECURE_NO_WARNINGS

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <iostream>
#include <algorithm>

#include <neo/Architect.h>
#include <neo/Hierarchy.h>
#include <neo/SparseFeaturesChunk.h>
#include <neo/SparseFeaturesDelay.h>
#include <neo/SparseFeaturesSTDP.h>
#include <neo/SparseFeaturesReLU.h>

using namespace ogmaneo;


void WriteImage2D(const std::string& namePrefix, const schemas::Image2D *fbImg) {
    if (fbImg == nullptr)
        return;

    uint32_t width = fbImg->width();
    uint32_t height = fbImg->height();
    uint32_t elementSize = fbImg->elementSize();

    sf::Color pixel;
    sf::Image img;
    img.create(width, height);

    std::string fileName = namePrefix;

    switch (fbImg->pixels_type())
    {
    case schemas::PixelData::PixelData_FloatArray:
    {
        const schemas::FloatArray* fbFloatArray =
            reinterpret_cast<const schemas::FloatArray*>(fbImg->pixels());
        const flatbuffers::Vector<float> *data = fbFloatArray->data();

        float minValue, maxValue;
#if defined(_DEBUG)
        minValue = FLT_MAX;
        maxValue = -FLT_MAX;
        for (flatbuffers::uoffset_t d = 0; d < data->Length(); d++) {
            if (data->Get(d) < minValue)    minValue = data->Get(d);
            if (data->Get(d) > maxValue)    maxValue = data->Get(d);
        }
#else
        minValue = *min_element(fbFloatArray->data()->begin(), fbFloatArray->data()->end());
        maxValue = *max_element(fbFloatArray->data()->begin(), fbFloatArray->data()->end());
#endif

        float offset = minValue;
        float scale = 1.f / (maxValue - minValue);

        // Catch [-1, 1]
        if (minValue >= -1.0f && minValue < 0.0f &&
            maxValue <= 1.0f && maxValue > 0.0f) {
            offset = 1.f;
            scale = 0.5f;
        }

        uint32_t elementsPerPixel = (elementSize / sizeof(float));
        uint32_t numElements = width * height * elementsPerPixel;
        float pixelValue;

        for (uint32_t i = 0; i < numElements; i += elementsPerPixel) {
            for (uint32_t j = 0; j < elementsPerPixel; j++) {
                pixelValue = (data->Get(i + j) + offset) * scale;

                switch (j) {
                default:
                case 0: pixel.r = (sf::Uint8)(pixelValue * 255.f); break;
                case 1: pixel.g = (sf::Uint8)(pixelValue * 255.f); break;
                case 2: pixel.b = (sf::Uint8)(pixelValue * 255.f); break;
                case 3: pixel.a = (sf::Uint8)(pixelValue * 255.f); break;
                }
            }

            img.setPixel((i / elementsPerPixel) % width, (i / elementsPerPixel) / width, pixel);
        }

        fileName += ".jpg";
        break;
    }
    case schemas::PixelData::PixelData_ByteArray:
    {
        const schemas::ByteArray* fbByteArray =
            reinterpret_cast<const schemas::ByteArray*>(fbImg->pixels());

        uint32_t elementsPerPixel = (elementSize / sizeof(unsigned char));
        uint32_t numElements = width * height * elementsPerPixel;
        sf::Uint8 pixelValue;

        const flatbuffers::Vector<uint8_t> *data = fbByteArray->data();
        for (uint32_t i = 0; i < numElements; i += elementsPerPixel) {
            for (uint32_t j = 0; j < elementsPerPixel; j++) {
                pixelValue = (sf::Uint8)data->Get(i + j);

                switch (j) {
                default:
                case 0: pixel.r = pixelValue; break;
                case 1: pixel.g = pixelValue; break;
                case 2: pixel.b = pixelValue; break;
                case 3: pixel.a = pixelValue; break;
                }
            }

            img.setPixel((i / elementsPerPixel) % width, (i / elementsPerPixel) / width, pixel);
        }

        fileName += ".jpg";
        break;
    }
    default:
        assert(0);
        break;
    }

    img.saveToFile(fileName);
}


void WriteImage3D(const std::string& namePrefix, const schemas::Image3D *fbImg) {
    if (fbImg == nullptr)
        return;

    /*
    uint32_t width = fbImg->width();
    uint32_t height = fbImg->height();
    uint32_t depth = fbImg->depth();
    uint32_t elementSize = fbImg->elementSize();

    sf::Color pixel;
    std::vector<sf::Image> imgs;
    std::vector<std::string> fileNames;

    for (uint32_t i = 0; i < depth; i++) {
    fileNames[i] = namePrefix + "_d" + std::to_string(i);
    imgs[i].create(width, height);
    }

    switch (fbImg->pixels_type())
    {
    case schemas::PixelData::PixelData_FloatArray:
    {
    const schemas::FloatArray* fbFloatArray =
    reinterpret_cast<const schemas::FloatArray*>(fbImg->pixels());

    float minValue = *min_element(fbFloatArray->data()->begin(), fbFloatArray->data()->end());
    float maxValue = *max_element(fbFloatArray->data()->begin(), fbFloatArray->data()->end());

    float offset = minValue;
    float scale = 1.f / (maxValue - minValue);

    // Catch [-1, 1]
    if (minValue >= -1.0f && minValue < 1.0f &&
    maxValue <= 1.0f && maxValue > -1.0f) {
    offset = 1.f;
    scale = 0.5f;
    }

    uint32_t elementsPerPixel = (elementSize / sizeof(float));
    uint32_t numElements = width * height * depth * elementsPerPixel;
    float pixelValue;

    for (uint32_t i = 0; i < depth; i++) {
    fileNames[i] += ".jpg";
    }
    break;
    }
    case schemas::PixelData::PixelData_ByteArray:
    {
    break;
    }
    default:
    assert(0);
    break;
    }

    for (uint32_t i = 0; i < depth; i++) {
    imgs[i].saveToFile(fileNames[i]);
    }
    */
}


void WriteDoubleBuffer2D(const std::string& namePrefix, const schemas::DoubleBuffer2D* fbDB) {
    //WriteImage2D(namePrefix + "_front", fbDB->_front());
    WriteImage2D(namePrefix + "_back", fbDB->_back());
}


void WriteDoubleBuffer3D(const std::string& namePrefix, const schemas::DoubleBuffer3D* fbDB) {
    //WriteImage3D(namePrefix + "_front", fbDB->_front());
    WriteImage3D(namePrefix + "_back", fbDB->_back());
}


void WriteSparseFeaturesChunk(const std::string& namePrefix, schemas::SparseFeaturesChunk* fbSparseFeaturesChunk) {

    cl_int2 _hiddenSize = cl_int2{ fbSparseFeaturesChunk->_hiddenSize()->x(), fbSparseFeaturesChunk->_hiddenSize()->y() };
    cl_int2 _chunkSize = cl_int2{ fbSparseFeaturesChunk->_chunkSize()->x(), fbSparseFeaturesChunk->_chunkSize()->y() };

    WriteDoubleBuffer2D(namePrefix + "_hiddenStates", fbSparseFeaturesChunk->_hiddenStates());
    WriteDoubleBuffer2D(namePrefix + "_hiddenActivations", fbSparseFeaturesChunk->_hiddenActivations());
    WriteDoubleBuffer2D(namePrefix + "_chunkWinners", fbSparseFeaturesChunk->_chunkWinners());
    WriteDoubleBuffer2D(namePrefix + "_hiddenSummationTemp", fbSparseFeaturesChunk->_hiddenSummationTemp());

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesChunk->_visibleLayerDescs()->Length(); i++) {
        const schemas::VisibleChunkLayerDesc* fbVisibleChunkLayerDesc = fbSparseFeaturesChunk->_visibleLayerDescs()->Get(i);

        cl_int2 _size = cl_int2{ fbVisibleChunkLayerDesc->_size().x(), fbVisibleChunkLayerDesc->_size().y() };
        int32_t _radius = fbVisibleChunkLayerDesc->_radius();
        uint8_t _ignoreMiddle = fbVisibleChunkLayerDesc->_ignoreMiddle();
        float _weightAlpha = fbVisibleChunkLayerDesc->_weightAlpha();
        float _lambda = fbVisibleChunkLayerDesc->_lambda();
    }

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesChunk->_visibleLayers()->Length(); i++) {
        const schemas::VisibleChunkLayer* fbVisibleChunkLayer = fbSparseFeaturesChunk->_visibleLayers()->Get(i);
        std::string layerPrefix = namePrefix + "_visibleLayer" + std::to_string(i);

        WriteDoubleBuffer2D(layerPrefix + "_derivedInput", fbVisibleChunkLayer->_derivedInput());
        WriteDoubleBuffer3D(layerPrefix + "_samples", fbVisibleChunkLayer->_samples());
        WriteDoubleBuffer3D(layerPrefix + "_weights", fbVisibleChunkLayer->_weights());

        cl_float2 _hiddenToVisible = cl_float2{ fbVisibleChunkLayer->_hiddenToVisible()->x(), fbVisibleChunkLayer->_hiddenToVisible()->y() };
        cl_float2 _visibleToHidden = cl_float2{ fbVisibleChunkLayer->_visibleToHidden()->x(), fbVisibleChunkLayer->_visibleToHidden()->y() };
        cl_int2 _reverseRadii = cl_int2{ fbVisibleChunkLayer->_reverseRadii()->x(), fbVisibleChunkLayer->_reverseRadii()->y() };
    }
}


void WriteSparseFeaturesDelay(const std::string& namePrefix, schemas::SparseFeaturesDelay* fbSparseFeaturesDelay) {

    cl_int2 _hiddenSize = cl_int2{ fbSparseFeaturesDelay->_hiddenSize()->x(), fbSparseFeaturesDelay->_hiddenSize()->y() };

    int32_t _inhibitionRadius = fbSparseFeaturesDelay->_inhibitionRadius();

    WriteDoubleBuffer2D(namePrefix + "_hiddenActivations", fbSparseFeaturesDelay->_hiddenActivations());
    WriteDoubleBuffer2D(namePrefix + "_hiddenStates", fbSparseFeaturesDelay->_hiddenStates());
    WriteDoubleBuffer2D(namePrefix + "_hiddenBiases", fbSparseFeaturesDelay->_hiddenBiases());
    WriteDoubleBuffer2D(namePrefix + "_hiddenSummationTemp", fbSparseFeaturesDelay->_hiddenSummationTemp());

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesDelay->_visibleLayerDescs()->Length(); i++) {
        const schemas::VisibleDelayLayerDesc* fbVisibleDelayLayerDesc = fbSparseFeaturesDelay->_visibleLayerDescs()->Get(i);

        cl_int2 _size = cl_int2{ fbVisibleDelayLayerDesc->_size().x(), fbVisibleDelayLayerDesc->_size().y() };
        int32_t _radius = fbVisibleDelayLayerDesc->_radius();
        uint8_t _ignoreMiddle = fbVisibleDelayLayerDesc->_ignoreMiddle();
        float _weightAlpha = fbVisibleDelayLayerDesc->_weightAlpha();
        float _lambda = fbVisibleDelayLayerDesc->_lambda();
        float _gamma = fbVisibleDelayLayerDesc->_gamma();
    }

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesDelay->_visibleLayers()->Length(); i++) {
        const schemas::VisibleDelayLayer* fbVisibleDelayLayer = fbSparseFeaturesDelay->_visibleLayers()->Get(i);
        std::string layerPrefix = namePrefix + "_visibleLayer" + std::to_string(i);

        WriteDoubleBuffer2D(layerPrefix + "_derivedInput", fbVisibleDelayLayer->_derivedInput());
        WriteDoubleBuffer3D(layerPrefix + "_weights", fbVisibleDelayLayer->_weights());

        cl_float2 _hiddenToVisible = cl_float2{ fbVisibleDelayLayer->_hiddenToVisible()->x(), fbVisibleDelayLayer->_hiddenToVisible()->y() };
        cl_float2 _visibleToHidden = cl_float2{ fbVisibleDelayLayer->_visibleToHidden()->x(), fbVisibleDelayLayer->_visibleToHidden()->y() };
        cl_int2 _reverseRadii = cl_int2{ fbVisibleDelayLayer->_reverseRadii()->x(), fbVisibleDelayLayer->_reverseRadii()->y() };
    }
}


void WriteSparseFeaturesSTDP(const std::string& namePrefix, schemas::SparseFeaturesSTDP* fbSparseFeaturesSTDP) {

    cl_int2 _hiddenSize = cl_int2{ fbSparseFeaturesSTDP->_hiddenSize()->x(), fbSparseFeaturesSTDP->_hiddenSize()->y() };

    int32_t _inhibitionRadius = fbSparseFeaturesSTDP->_inhibitionRadius();

    WriteDoubleBuffer2D(namePrefix + "_hiddenActivations", fbSparseFeaturesSTDP->_hiddenActivations());
    WriteDoubleBuffer2D(namePrefix + "_hiddenStates", fbSparseFeaturesSTDP->_hiddenStates());
    WriteDoubleBuffer2D(namePrefix + "_hiddenBiases", fbSparseFeaturesSTDP->_hiddenBiases());
    WriteDoubleBuffer2D(namePrefix + "_hiddenSummationTemp", fbSparseFeaturesSTDP->_hiddenSummationTemp());

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesSTDP->_visibleLayerDescs()->Length(); i++) {
        const schemas::VisibleSTDPLayerDesc* fbVisibleSTDPLayerDesc = fbSparseFeaturesSTDP->_visibleLayerDescs()->Get(i);

        cl_int2 _size = cl_int2{ fbVisibleSTDPLayerDesc->_size().x(), fbVisibleSTDPLayerDesc->_size().y() };
        int32_t _radius = fbVisibleSTDPLayerDesc->_radius();
        uint8_t _ignoreMiddle = fbVisibleSTDPLayerDesc->_ignoreMiddle();
        float _weightAlpha = fbVisibleSTDPLayerDesc->_weightAlpha();
    }

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesSTDP->_visibleLayers()->Length(); i++) {
        const schemas::VisibleSTDPLayer* fbVisibleSTDPLayer = fbSparseFeaturesSTDP->_visibleLayers()->Get(i);
        std::string layerPrefix = namePrefix + "_visibleLayer" + std::to_string(i);

        WriteDoubleBuffer2D(layerPrefix + "_derivedInput", fbVisibleSTDPLayer->_derivedInput());
        WriteDoubleBuffer3D(layerPrefix + "_weights", fbVisibleSTDPLayer->_weights());

        cl_float2 _hiddenToVisible = cl_float2{ fbVisibleSTDPLayer->_hiddenToVisible()->x(), fbVisibleSTDPLayer->_hiddenToVisible()->y() };
        cl_float2 _visibleToHidden = cl_float2{ fbVisibleSTDPLayer->_visibleToHidden()->x(), fbVisibleSTDPLayer->_visibleToHidden()->y() };
        cl_int2 _reverseRadii = cl_int2{ fbVisibleSTDPLayer->_reverseRadii()->x(), fbVisibleSTDPLayer->_reverseRadii()->y() };
    }
}


void WriteSparseFeaturesReLU(const std::string& namePrefix, schemas::SparseFeaturesReLU* fbSparseFeaturesReLU) {

    cl_int2 _hiddenSize = cl_int2{ fbSparseFeaturesReLU->_hiddenSize()->x(), fbSparseFeaturesReLU->_hiddenSize()->y() };

    int32_t _numSamples = fbSparseFeaturesReLU->_numSamples();
    float _gamma = fbSparseFeaturesReLU->_gamma();
    float _activeRatio = fbSparseFeaturesReLU->_activeRatio();
    float __biasAlpha = fbSparseFeaturesReLU->_biasAlpha();

    WriteDoubleBuffer2D(namePrefix + "_hiddenStates", fbSparseFeaturesReLU->_hiddenStates());
    WriteDoubleBuffer2D(namePrefix + "_hiddenBiases", fbSparseFeaturesReLU->_hiddenBiases());
    WriteDoubleBuffer2D(namePrefix + "_hiddenSummationTemp", fbSparseFeaturesReLU->_hiddenSummationTemp());

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesReLU->_visibleLayerDescs()->Length(); i++) {
        const schemas::VisibleReLULayerDesc* fbVisibleReLULayerDesc = fbSparseFeaturesReLU->_visibleLayerDescs()->Get(i);

        cl_int2 _size = cl_int2{ fbVisibleReLULayerDesc->_size().x(), fbVisibleReLULayerDesc->_size().y() };
        int32_t _radiusHidden = fbVisibleReLULayerDesc->_radiusHidden();
        int32_t _radiusVisible = fbVisibleReLULayerDesc->_radiusVisible();
        uint8_t _ignoreMiddle = fbVisibleReLULayerDesc->_ignoreMiddle();
        float _weightAlphaHidden = fbVisibleReLULayerDesc->_weightAlphaHidden();
        float _weightAlphaVisible = fbVisibleReLULayerDesc->_weightAlphaVisible();
        float _lambda = fbVisibleReLULayerDesc->_lambda();
        bool _predict = fbVisibleReLULayerDesc->_predict();
    }

    for (flatbuffers::uoffset_t i = 0; i < fbSparseFeaturesReLU->_visibleLayers()->Length(); i++) {
        const schemas::VisibleReLULayerDesc* fbVisibleReLULayerDesc = fbSparseFeaturesReLU->_visibleLayerDescs()->Get(i);
        const schemas::VisibleReLULayer* fbVisibleReLULayer = fbSparseFeaturesReLU->_visibleLayers()->Get(i);
        std::string layerPrefix = namePrefix + "_visibleLayer" + std::to_string(i);

        WriteDoubleBuffer2D(layerPrefix + "_derivedInput", fbVisibleReLULayer->_derivedInput());
        WriteDoubleBuffer3D(layerPrefix + "_samples", fbVisibleReLULayer->_samples());
        WriteDoubleBuffer3D(layerPrefix + "_weightsHidden", fbVisibleReLULayer->_weightsHidden());

        if (fbVisibleReLULayerDesc->_predict()) {
            WriteDoubleBuffer2D(layerPrefix + "_predictions", fbVisibleReLULayer->_predictions());
            WriteDoubleBuffer3D(layerPrefix + "_weightsVisible", fbVisibleReLULayer->_weightsVisible());
        }

        cl_float2 _hiddenToVisible = cl_float2{ fbVisibleReLULayer->_hiddenToVisible()->x(), fbVisibleReLULayer->_hiddenToVisible()->y() };
        cl_float2 _visibleToHidden = cl_float2{ fbVisibleReLULayer->_visibleToHidden()->x(), fbVisibleReLULayer->_visibleToHidden()->y() };
        cl_int2 _reverseRadiiHidden = cl_int2{ fbVisibleReLULayer->_reverseRadiiHidden()->x(), fbVisibleReLULayer->_reverseRadiiHidden()->y() };
        cl_int2 _reverseRadiiVisible = cl_int2{ fbVisibleReLULayer->_reverseRadiiVisible()->x(), fbVisibleReLULayer->_reverseRadiiVisible()->y() };
    }
}


void WritePredictorLayer(const std::string& namePrefix, const schemas::PredictorLayer* fbPredictorLayer) {

    cl_int2 _hiddenSize = cl_int2{ fbPredictorLayer->_hiddenSize()->x(), fbPredictorLayer->_hiddenSize()->y() };

    WriteDoubleBuffer2D(namePrefix + "_hiddenSummationTemp", fbPredictorLayer->_hiddenSummationTemp());
    WriteDoubleBuffer2D(namePrefix + "_hiddenStates", fbPredictorLayer->_hiddenStates());

    std::shared_ptr<SparseFeatures> _inhibitSparseFeatures;
    if (fbPredictorLayer->_inhibitSparseFeatures()) {
        switch (fbPredictorLayer->_inhibitSparseFeatures()->_sf_type()) {
        default:
            break;

        case schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesChunk: {
            schemas::SparseFeaturesChunk* fbSparseFeaturesChunk =
                (schemas::SparseFeaturesChunk*)(fbPredictorLayer->_inhibitSparseFeatures()->_sf());

            WriteSparseFeaturesChunk(namePrefix + "_chunk", fbSparseFeaturesChunk);
            break;
        }

        case schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesDelay: {
            schemas::SparseFeaturesDelay* fbSparseFeaturesDelay =
                (schemas::SparseFeaturesDelay*)(fbPredictorLayer->_inhibitSparseFeatures()->_sf());

            WriteSparseFeaturesDelay(namePrefix + "_delay", fbSparseFeaturesDelay);
            break;
        }

        case schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesSTDP: {
            schemas::SparseFeaturesSTDP* fbSparseFeaturesSTDP =
                (schemas::SparseFeaturesSTDP*)(fbPredictorLayer->_inhibitSparseFeatures()->_sf());

            WriteSparseFeaturesSTDP(namePrefix + "_STDP", fbSparseFeaturesSTDP);
            break;
        }

        case schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesReLU: {
            schemas::SparseFeaturesReLU* fbSparseFeaturesReLU =
                (schemas::SparseFeaturesReLU*)(fbPredictorLayer->_inhibitSparseFeatures()->_sf());

            WriteSparseFeaturesReLU(namePrefix + "_ReLU", fbSparseFeaturesReLU);
            break;
        }
        }
    }

    for (flatbuffers::uoffset_t i = 0; i < fbPredictorLayer->_visibleLayerDescs()->Length(); i++) {
        const schemas::VisiblePredictorLayerDesc* fbVisiblePredictorLayerDesc = fbPredictorLayer->_visibleLayerDescs()->Get(i);

        cl_int2 _size = cl_int2{ fbVisiblePredictorLayerDesc->_size().x(), fbVisiblePredictorLayerDesc->_size().y() };
        int32_t  _radius = fbVisiblePredictorLayerDesc->_radius();
        float _alpha = fbVisiblePredictorLayerDesc->_alpha();
    }

    for (flatbuffers::uoffset_t i = 0; i < fbPredictorLayer->_visibleLayers()->Length(); i++) {
        const schemas::VisiblePredictorLayer* fbVisiblePredictorLayer = fbPredictorLayer->_visibleLayers()->Get(i);
        std::string layerPrefix = namePrefix + "_visiblePredictorLayer" + std::to_string(i);

        cl_float2 _hiddenToVisible = cl_float2{ fbVisiblePredictorLayer->_hiddenToVisible()->x(), fbVisiblePredictorLayer->_hiddenToVisible()->y() };
        cl_float2 _visibleToHidden = cl_float2{ fbVisiblePredictorLayer->_visibleToHidden()->x(), fbVisiblePredictorLayer->_visibleToHidden()->y() };
        cl_int2 _reverseRadii = cl_int2{ fbVisiblePredictorLayer->_reverseRadii()->x(), fbVisiblePredictorLayer->_reverseRadii()->y() };

        WriteDoubleBuffer2D(layerPrefix + "_derivedInput", fbVisiblePredictorLayer->_derivedInput());
        WriteDoubleBuffer3D(layerPrefix + "_weights", fbVisiblePredictorLayer->_weights());
    }
}


void WriteValueField2D(const std::string& namePrefix, const schemas::ValueField2D* valueField) {
    const schemas::Vec2i* size = valueField->_size();

    sf::Color pixel;
    sf::Image img;
    img.create(size->x(), size->y());

    uint32_t numElements = size->x() * size->y();
    float pixelValue, minValue, maxValue;

#if defined(_DEBUG)
    minValue = FLT_MAX;
    maxValue = -FLT_MAX;
    const flatbuffers::Vector<float> *data = valueField->_data();
    for (flatbuffers::uoffset_t d = 0; d < data->Length(); d++) {
        if (data->Get(d) < minValue)    minValue = data->Get(d);
        if (data->Get(d) > maxValue)    maxValue = data->Get(d);
    }
#else
    minValue = *min_element(valueField->_data()->begin(), valueField->_data()->end());
    maxValue = *max_element(valueField->_data()->begin(), valueField->_data()->end());
#endif

    float offset = minValue;
    float scale = 1.f / (maxValue - minValue);

    for (uint32_t i = 0; i < numElements; i++) {
        pixelValue = ((float)valueField->_data()->Get(i) + offset) * scale;

        pixel.r = (sf::Uint8)(pixelValue * 255.f);
        img.setPixel(i % size->x(), i / size->x(), pixel);
    }

    img.saveToFile(namePrefix + ".jpg");
}


int main(int argc, char *argv[]) {

    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " namePrefix ohrCount" << std::endl;
        std::cout << "       A hierachy/agent is regenerated from a namePrefix.oar file." << std::endl;
        std::cout << "       The network state is reloaded from a namePrefix.ohr file(s)." << std::endl;
        std::cout << "       If ohrCount is specified, and greater than one, a count is " << std::endl;
        std::cout << "       appended to the namePrefix when loading ohr files." << std::endl;
        return 1;
    }

    std::string prefix(argv[1]);
    int ohrCount = 1;

    if (argc == 3)
        ohrCount = atoi(argv[2]);


    // --------------------------- Create the Hierarchy ---------------------------

    std::shared_ptr<Resources> res = std::make_shared<Resources>();

    res->create(ComputeSystem::_gpu);

    Architect arch;
    arch.initialize(1234, res);

    std::cout << "Constructing hierarchy" << std::endl;
    arch.load(prefix + ".oar");

    // Generate the hierarchy
    std::shared_ptr<Hierarchy> hierarchy = arch.generateHierarchy();


    // --------------------------- Dump out the hierarchies -------------------------

    for (int iter = 0; iter < ohrCount; iter++) {
        std::string namePrefix = "";

        if (ohrCount > 1)
            namePrefix += std::to_string(iter);

        std::cout << "Parsing hierarchy" << std::endl;
        std::string fileName(prefix + namePrefix + ".ohr");

        FILE* file = fopen(fileName.c_str(), "rb");
        fseek(file, 0L, SEEK_END);
        size_t length = ftell(file);
        fseek(file, 0L, SEEK_SET);

        std::vector<uint8_t> data(length);
        fread(data.data(), sizeof(uint8_t), length, file);
        fclose(file);

        flatbuffers::Verifier verifier = flatbuffers::Verifier(data.data(), length);

        bool verified =
            schemas::VerifyHierarchyBuffer(verifier) |
            schemas::HierarchyBufferHasIdentifier(data.data());

        if (verified) {
            const std::shared_ptr<ComputeSystem> &cs = res->getComputeSystem();
            const schemas::Hierarchy* fbHierarchy = schemas::GetHierarchy(data.data());

            // Hierarchy::load
            {
                const schemas::Predictor* fbPredictor = fbHierarchy->_p();

                //Predictor::load
                {
                    const schemas::FeatureHierarchy* fbFeatureHierarchy = fbPredictor->_h();

                    // FeatureHierarchy::load
                    {
                        for (flatbuffers::uoffset_t i = 0; i < fbFeatureHierarchy->_layerDescs()->Length(); i++) {
                            const schemas::FeatureHierarchyLayerDesc* fbFeatureHierarchyLayerDesc = fbFeatureHierarchy->_layerDescs()->Get(i);

                            // FeatureHierarchyLayerDesc::load
                            {
                                const schemas::SparseFeaturesDesc* fbSparseFeaturesDesc = fbFeatureHierarchyLayerDesc->_sfDesc();

                                int32_t _poolSteps = fbFeatureHierarchyLayerDesc->_poolSteps();
                                const char* _name = fbSparseFeaturesDesc->_name()->c_str();

                                SparseFeatures::InputType _inputType;
                                switch (fbSparseFeaturesDesc->_inputType()) {
                                default:
                                case schemas::InputType::InputType__feedForward:            _inputType = SparseFeatures::InputType::_feedForward; break;
                                case schemas::InputType::InputType__feedForwardRecurrent:   _inputType = SparseFeatures::InputType::_feedForwardRecurrent; break;
                                }
                            }
                        }

                        for (flatbuffers::uoffset_t i = 0; i < fbFeatureHierarchy->_layers()->Length(); i++) {
                            const schemas::FeatureHierarchyLayer* fbFeatureHierarchyLayer = fbFeatureHierarchy->_layers()->Get(i);

                            // FeatureHierarchyLayer::load
                            {
                                schemas::SparseFeatures* fbSparseFeatures = (schemas::SparseFeatures*)(fbFeatureHierarchyLayer->_sf());

                                // SparseFeatures::load
                                {
                                    switch (fbFeatureHierarchyLayer->_sf_type()) {
                                    default:
                                        break;

                                    case schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesChunk: {
                                        schemas::SparseFeaturesChunk* fbSparseFeaturesChunk =
                                            (schemas::SparseFeaturesChunk*)(fbSparseFeatures->_sf());

                                        WriteSparseFeaturesChunk(namePrefix + "_chunk", fbSparseFeaturesChunk);
                                        break;
                                    }

                                    case schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesDelay: {
                                        schemas::SparseFeaturesDelay* fbSparseFeaturesDelay =
                                            (schemas::SparseFeaturesDelay*)(fbSparseFeatures->_sf());

                                        WriteSparseFeaturesDelay(namePrefix + "_delay", fbSparseFeaturesDelay);
                                        break;
                                    }

                                    case schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesSTDP: {
                                        schemas::SparseFeaturesSTDP* fbSparseFeaturesSTDP =
                                            (schemas::SparseFeaturesSTDP*)(fbSparseFeatures->_sf());

                                        WriteSparseFeaturesSTDP(namePrefix + "_STDP", fbSparseFeaturesSTDP);
                                        break;
                                    }

                                    case schemas::SparseFeaturesType::SparseFeaturesType_SparseFeaturesReLU: {
                                        schemas::SparseFeaturesReLU* fbSparseFeaturesReLU =
                                            (schemas::SparseFeaturesReLU*)(fbSparseFeatures->_sf());

                                        WriteSparseFeaturesReLU(namePrefix + "_ReLU", fbSparseFeaturesReLU);
                                        break;
                                    }
                                    }

                                    int32_t _clock = fbFeatureHierarchyLayer->_clock();
                                    bool _tpReset = fbFeatureHierarchyLayer->_tpReset();
                                    bool _tpNextReset = fbFeatureHierarchyLayer->_tpNextReset();

                                    //WriteDoubleBuffer2D(namePrefix + "_tpBuffer", fbFeatureHierarchyLayer->_tpBuffer());
                                    WriteImage2D(namePrefix + "_predErrors", fbFeatureHierarchyLayer->_predErrors());
                                }
                            }
                        }
                    }
                }
/*
                for (flatbuffers::uoffset_t i = 0; i < fbPredictor->_pLayerDescs()->Length(); i++) {
                    const schemas::PredLayerDesc* fbPredLayerDesc = fbPredictor->_pLayerDescs()->Get(i);

                    int32_t  _radius = fbPredLayerDesc->_radius();
                    float _alpha = fbPredLayerDesc->_alpha();
                    float _beta = fbPredLayerDesc->_beta();
                }

                for (flatbuffers::uoffset_t i = 0; i < fbPredictor->_pLayers()->Length(); i++) {
                    const schemas::PredictorLayer* fbPredictorLayer = fbPredictor->_pLayers()->Get(i);
                    std::string layerPrefix = namePrefix + "_hierarchyPredictorLayer" + std::to_string(i);

                    WritePredictorLayer(layerPrefix, fbPredictorLayer);
                }
*/
            }

            //for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_inputImages()->Length(); i++) {
            //    WriteImage2D(namePrefix + "_inputImages" + std::to_string(i), fbHierarchy->_inputImages()->Get(i));
            //}

            //for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_corruptedInputImages()->Length(); i++) {
            //    WriteImage2D(namePrefix + "_inputImagesCorrupted" + std::to_string(i), fbHierarchy->_corruptedInputImages()->Get(i));
            //}
/*
            for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_readoutLayers()->Length(); i++) {
                const schemas::PredictorLayer* fbPredictorLayer = fbHierarchy->_readoutLayers()->Get(i);
                std::string layerPrefix = namePrefix + "_predictorLayer" + std::to_string(i);

                WritePredictorLayer(layerPrefix, fbPredictorLayer);
            }
*/
            for (flatbuffers::uoffset_t i = 0; i < fbHierarchy->_predictions()->Length(); i++) {
                WriteValueField2D(namePrefix + "_predictions" + std::to_string(i), fbHierarchy->_predictions()->Get(i));
            }
        }
    }

    return 0;
}