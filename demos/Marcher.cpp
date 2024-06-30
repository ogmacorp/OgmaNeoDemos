// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "constructs/Vec2f.h"
#include "constructs/Vec3f.h"
#include "constructs/Matrix4x4f.h"
#include "constructs/Quaternion.h"
#include <aogmaneo/ImageEncoder.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

#include <omp.h>
#include <memory.h>

#include <assert.h>

#include "toml.hpp"

float sigmoid(float x) {
    return std::tanh(x * 0.5f) * 0.5f + 0.5f;
}

struct View {
    Matrix4x4f pose;
    sf::Image img;
};

struct MiniEncoder {
    int numInputs;
    int numHiddenColumns;
    int numHiddenCellsPerColumn;

    std::vector<float> weights;
    std::vector<float> rates;

    std::vector<int> hiddenCIs;

    float lr = 0.03f;

    void initRandom(int numInputs, int numHiddenColumns, int numHiddenCellsPerColumn, std::mt19937 &rng) {
        this->numInputs = numInputs;
        this->numHiddenColumns = numHiddenColumns;
        this->numHiddenCellsPerColumn = numHiddenCellsPerColumn;

        weights.resize(numHiddenColumns * numHiddenCellsPerColumn * numInputs);

        std::uniform_real_distribution<float> weightDist(-1.0f, 1.0f);

        for (int i = 0; i < weights.size(); i++)
            weights[i] = weightDist(rng);

        rates = std::vector<float>(numHiddenColumns * numHiddenCellsPerColumn, 0.5f);

        hiddenCIs = std::vector<int>(numHiddenColumns, 0);
    }

    void step(const std::vector<float> &inputs, bool learnEnabled) {
        for (int h = 0; h < numHiddenColumns; h++) {
            int maxIndex = -1;
            float maxActivation = -999999.0f;

            for (int hc = 0; hc < numHiddenCellsPerColumn; hc++) {
                int hci = h * numHiddenCellsPerColumn + hc;

                float sum = 0.0f;

                for (int i = 0; i < numInputs; i++) {
                    float inValue = inputs[i];

                    float delta = inValue - weights[i + numInputs * hci];

                    sum -= delta * delta;
                }
                
                if (sum > maxActivation || maxIndex == -1) {
                    maxActivation = sum;
                    maxIndex = hc;
                }
            }

            hiddenCIs[h] = maxIndex;

            if (learnEnabled) {
                for (int dhc = -1; dhc <= 1; dhc++) {
                    int hc = maxIndex + dhc;

                    if (hc < 0 || hc >= numHiddenCellsPerColumn)
                        continue;

                    int hci = h * numHiddenCellsPerColumn + hc;

                    for (int i = 0; i < numInputs; i++) {
                        float inValue = inputs[i];

                        int wi = i + numInputs * hci;

                        float delta = inValue - weights[wi];

                        weights[wi] += rates[hci] * delta;
                    }

                    rates[hci] *= 1.0f - lr;
                }
            }
        }
    }
};

struct MiniDecoder {
    int numInputColumns;
    int numInputCellsPerColumn;
    int numOutputs;

    std::vector<float> weights;

    std::vector<float> outputs;

    float lr = 1.0f;

    void initRandom(int numInputColumns, int numInputCellsPerColumn, int numOutputs, std::mt19937 &rng) {
        this->numInputColumns = numInputColumns;
        this->numInputCellsPerColumn = numInputCellsPerColumn;
        this->numOutputs = numOutputs;

        weights.resize(numOutputs * numInputColumns * numInputCellsPerColumn);

        std::uniform_real_distribution<float> weightDist(0.0f, 0.01f);

        for (int i = 0; i < weights.size(); i++)
            weights[i] = weightDist(rng);

        outputs = std::vector<float>(numOutputs, 0.0f);
    }

    void activate(const std::vector<int> &inputCIs) {
        for (int o = 0; o < numOutputs; o++) {
            float sum = 0.0f;

            for (int i = 0; i < numInputColumns; i++)
                sum += weights[inputCIs[i] + numInputCellsPerColumn * (i + numInputColumns * o)];    
            
            sum /= numInputColumns;

            outputs[o] = sum;
        }
    }

    void learnErrors(const std::vector<int> &inputCIs, const std::vector<float> &errors) {
        for (int o = 0; o < numOutputs; o++) {
            float delta = lr * errors[o];

            for (int i = 0; i < numInputColumns; i++)
                weights[inputCIs[i] + numInputCellsPerColumn * (i + numInputColumns * o)] += delta; 
        }
    }
};

struct Term {
    float T;
    float prob;
    Vec3f subColor;
    float subDensity;
    std::vector<int> hiddenCIs;
    float densityErrorAccum;
};

struct Config {
    float step = 0.1f;
    float maxDist = 5.0f;
    float bounds = 4.0f;

    std::vector<float> inputs = { 0.0f, 0.0f, 0.0f };
};

//void encodePoint(const Vec3f &point, Config &cfg) {
//    assert(cfg.inputCIs.size() == cfg.numModules * 3);
//
//    Vec3f p = point;
//    float scale = cfg.startScale;
//
//    for (int m = 0; m < cfg.numModules; m++) {
//        Vec3f coords(std::fmod(p.x / scale, 1.0f), std::fmod(p.y / scale, 1.0f), std::fmod(p.z / scale, 1.0f));
//
//        int bx = static_cast<int>((coords.x * 0.5f + 0.5f) * (cfg.moduleSize - 1) + 0.5f);
//        int by = static_cast<int>((coords.y * 0.5f + 0.5f) * (cfg.moduleSize - 1) + 0.5f);
//        int bz = static_cast<int>((coords.z * 0.5f + 0.5f) * (cfg.moduleSize - 1) + 0.5f);
//
//        cfg.inputCIs[m * 3 + 0] = bx;
//        cfg.inputCIs[m * 3 + 1] = by;
//        cfg.inputCIs[m * 3 + 2] = bz;
//
//        float rescale = 1.0f / (cfg.moduleSize - 1);
//
//        Vec3f unbinnedCoords = Vec3f(bx, by, bz) * rescale * 2.0f - Vec3f(1.0f, 1.0f, 1.0f);
//
//        p -= unbinnedCoords * scale;
//
//        scale /= cfg.moduleScale;
//    }
//}

void sample(MiniEncoder &enc, MiniDecoder &dec, const Vec3f &point, float &density, Vec3f &color, Config &cfg) {
    // Encode point
    cfg.inputs[0] = point.x;
    cfg.inputs[1] = point.y;
    cfg.inputs[2] = point.z;

    // Encoder portion
    enc.step(cfg.inputs, false);

    // Decoder portion
    dec.activate(enc.hiddenCIs);

    density = std::max(0.0f, dec.outputs[0]);
    color.x = sigmoid(dec.outputs[1]);
    color.y = sigmoid(dec.outputs[2]);
    color.z = sigmoid(dec.outputs[3]);
}

void encUpdate(MiniEncoder &enc, const Vec3f &point, Config &cfg) {
    // Encode point
    cfg.inputs[0] = point.x;
    cfg.inputs[1] = point.y;
    cfg.inputs[2] = point.z;

    // Encoder portion
    enc.step(cfg.inputs, true);
}

void trace(MiniEncoder &enc, MiniDecoder &dec, const Vec3f &origin, const Vec3f &dir, Vec3f &color, Config &cfg) {
    color = Vec3f(0.0f, 0.0f, 0.0f);
    float accum = 0.0f;
    float t = 0.0f;

    while (t < cfg.maxDist) {
        t += cfg.step;

        Vec3f point = origin + dir * t;

        float subDensity;
        Vec3f subColor;

        if (point.x > cfg.bounds || point.x < -cfg.bounds ||
            point.y > cfg.bounds || point.y < -cfg.bounds ||
            point.z > cfg.bounds || point.z < -cfg.bounds)
        {
            // Out of bounds of volume
            continue;
        }

        sample(enc, dec, point, subDensity, subColor, cfg);

        float T = std::exp(-accum);
        float prob = std::exp(-subDensity);

        accum += subDensity;

        color += T * (1.0f - prob) * subColor;
    }
}

void update(MiniEncoder &enc, MiniDecoder &dec, const Vec3f &origin, const Vec3f &dir, const Vec3f &targetColor, Config &cfg) {
    std::vector<Term> terms;
    terms.reserve(static_cast<int>(std::ceil(cfg.maxDist / cfg.step)));

    Vec3f color = Vec3f(0.0f, 0.0f, 0.0f);
    float accum = 0.0f;
    float t = 0.0f;

    // Train enc
    while (t < cfg.maxDist) {
        t += cfg.step;

        Vec3f point = origin + dir * t;

        encUpdate(enc, point, cfg);
    }

    t = 0.0f;

    while (t < cfg.maxDist) {
        t += cfg.step;

        Vec3f point = origin + dir * t;

        float subDensity;
        Vec3f subColor;

        sample(enc, dec, point, subDensity, subColor, cfg);

        float T = std::exp(-accum);
        float prob = std::exp(-subDensity);

        accum += subDensity;

        color += T * (1.0f - prob) * subColor;

        Term term;

        term.T = T;
        term.prob = prob;
        term.subColor = subColor;
        term.subDensity = subDensity;
        term.hiddenCIs = enc.hiddenCIs;
        term.densityErrorAccum = 0.0f;

        terms.push_back(term);
    }

    Vec3f colorError = targetColor - color;

    for (int ti = 0; ti < terms.size(); ti++) {
        // Update past some more
        Vec3f subDensityErrors1 = colorError * terms[ti].subColor * (1.0f - terms[ti].prob) * -terms[ti].T;
        float d = subDensityErrors1.x + subDensityErrors1.y + subDensityErrors1.z;

        for (int ti2 = ti - 1; ti2 >= 0; ti2--)
            terms[ti2].densityErrorAccum += d;

        Vec3f subDensityErrors2 = colorError * terms[ti].subColor * terms[ti].T * terms[ti].prob;

        terms[ti].densityErrorAccum += subDensityErrors2.x + subDensityErrors2.y + subDensityErrors2.z;
    }

    // Second pass after density accum
    for (int ti = 0; ti < terms.size(); ti++) {
        Vec3f subColorError = colorError * terms[ti].T * (1.0f - terms[ti].prob) * terms[ti].subColor * (Vec3f(1.0f, 1.0f, 1.0f) - terms[ti].subColor);

        dec.learnErrors(terms[ti].hiddenCIs, { terms[ti].densityErrorAccum, subColorError.x, subColorError.y, subColorError.z }); 
    }
}

int main(int argc, char *argv[]) {
    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    sf::Clock clock;

    std::cout << "Loading views..." << std::endl;

    // Load views
    toml::table tbl;

    try {
        tbl = toml::parse_file("resources/transforms.toml");
    }
    catch (const toml::parse_error &err) {
       std::cerr << "Parsing expview.toml failed:" << std::endl << err << std::endl;

       return 1;
    }

    std::vector<View> views;

    toml::array* frames = tbl["frames"].as_array();

    if (frames) {
        views.resize(frames->size());

        for (int f = 0; f < frames->size(); f++) {
            toml::table* frame = (*frames)[f].as_table();

            if (frame) {
                std::optional<std::string> opt_file_path = (*frame)["file_path"].value<std::string>();

                if (opt_file_path) {
                    std::string file_path = *opt_file_path;

                    View &v = views[f];
                    v.img.loadFromFile("resources/" + file_path + ".png");
                    
                    toml::array* rows = (*frame)["transform_matrix"].as_array();

                    if (rows) {
                        for (int i = 0; i < rows->size(); i++) {
                            toml::array* columns = (*rows)[i].as_array();

                            if (columns) {
                                for (int j = 0; j < columns->size(); j++) {
                                    v.pose.set(j, i, *(*columns)[j].value<float>());
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Load test views
    //views.clear();

    //sf::Image testImg;
    //testImg.loadFromFile("resources/testImg.png");
    //views.resize(3);
    //views[0].pose = Matrix4x4f::translateMatrix(Vec3f(-3.0f, 0.0f, 0.0f));
    //views[0].img = testImg;
    //views[1].pose = Matrix4x4f::translateMatrix(Vec3f(0.0f, 3.0f, 0.0f)) * Matrix4x4f::rotateMatrixZ(M_PI * 0.5f);;
    //views[1].img = testImg;
    //views[2].pose = Matrix4x4f::translateMatrix(Vec3f(0.0f, 0.0f, -3.0f)) * Matrix4x4f::rotateMatrixY(M_PI * 0.5f);;
    //views[2].img = testImg;

    sf::Vector2u imgSize = views[0].img.getSize();

    float aspect = static_cast<float>(imgSize.x) / static_cast<float>(imgSize.y);

    float cameraAngleX = *tbl["camera_angle_x"].value<float>();
    float cameraAngleY = cameraAngleX / aspect;

    std::cout << "Views loaded." << std::endl;

    std::uniform_int_distribution<int> viewDist(0, views.size() - 1);
    std::uniform_int_distribution<int> widthDist(0, imgSize.x - 1);
    std::uniform_int_distribution<int> heightDist(0, imgSize.y - 1);

    float aperature_dist = 1.0f / std::sin(cameraAngleX);

    Config cfg;

    MiniEncoder enc;
    enc.initRandom(3, 3, 256, rng);

    MiniDecoder dec;
    dec.initRandom(enc.numHiddenColumns, enc.numHiddenCellsPerColumn, 4, rng);

    for (int it = 0; it < 1000000; it++) {
        int v = viewDist(rng);

        int px = widthDist(rng);
        int py = heightDist(rng);

        Vec2f normalized = (Vec2f(px, py) / Vec2f(imgSize.x, imgSize.x) * 2.0f) - Vec2f(1.0f, 1.0f);

        Vec3f origin = views[v].pose * Vec3f(0.0f, 0.0f, 0.0f);
        Vec3f dir = ((views[v].pose * (Vec3f(aperature_dist, normalized.y, normalized.x)).normalized()) - origin);

        sf::Color targetColor = views[v].img.getPixel(px, imgSize.y - 1 - py);

        if (targetColor.a < 127)
            targetColor = sf::Color::Black;

        update(enc, dec, origin, dir, Vec3f(targetColor.r / 255.0f, targetColor.g / 255.0f, targetColor.b / 255.0f), cfg);

        if (it % 1000 == 0) {
            std::cout << it << std::endl;
        }
    }

    std::cout << "Ready." << std::endl;

    // --------------------------- Create the window(s) ---------------------------

    bool quit = false;

    Vec3f camPosition(2.0f, 2.0f, 2.0f);
    Vec3f camVelocity(0.0, 0.0f, 0.0f);
    float angleX = 0.0f;
    float angleY = 0.0f;
    float sensitivity = 0.2f;
    float acceleration = 512.0f;
    float runMultiplier = 4.0f;
    float deceleration = 32.0f;

    const float renderScale = 0.1f;
    sf::Vector2u renderSize(imgSize.x * renderScale, imgSize.y * renderScale);

    sf::Image rendered;
    rendered.create(renderSize.x, renderSize.y);

    float dt = 1.0f / 60.0f;

    sf::RenderWindow window;

    float showScale = 8.0f;

    window.create(sf::VideoMode(rendered.getSize().x * showScale, rendered.getSize().y * showScale), "Marcher", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(20);
    aperature_dist *= 0.5f; // Increase fov for presentation

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;
        }

        sf::Vector2i mousePosition = sf::Mouse::getPosition(window) - sf::Vector2i(128, 128);
        sf::Mouse::setPosition(sf::Vector2i(128, 128), window);

        const float baseSensitivity = 0.01f;
        float finalSensitivity = baseSensitivity * sensitivity;

        angleX -= mousePosition.x * finalSensitivity;
        angleY += mousePosition.y * finalSensitivity;

        angleX = std::fmod(angleX, M_PI * 2.0f);

        if (angleY < -M_PI * 0.5f)
            angleY = -M_PI * 0.5f;
        else if (angleY > M_PI * 0.5f)
            angleY = M_PI * 0.5f;

        Matrix4x4f camT = Matrix4x4f::rotateMatrixY(-angleX) * Matrix4x4f::rotateMatrixZ(angleY);

        float accel = sf::Keyboard::isKeyPressed(sf::Keyboard::LShift) ? acceleration * runMultiplier : acceleration;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
            camVelocity += camT * Vec3f(accel * dt, 0.0f, 0.0f);
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
            camVelocity += camT * Vec3f(-accel * dt, 0.0f, 0.0f);

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
            camVelocity += camT * Vec3f(0.0f, 0.0f, accel * dt);
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
            camVelocity += camT * Vec3f(0.0f, 0.0f, -accel * dt);

        camVelocity += -deceleration * camVelocity * dt;

        camPosition += camVelocity * dt;

        std::cout << camPosition << std::endl;

        Matrix4x4f pose = Matrix4x4f::translateMatrix(camPosition) * camT;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num0))
            pose = views[viewDist(rng)].pose;

        window.clear();

        // Render frame
        for (int x = 0; x < renderSize.x; x++) {
            for (int y = 0; y < renderSize.y; y++) {
                Vec2f normalized = (Vec2f(x, y) / Vec2f(renderSize.x, renderSize.x) * 2.0f) - Vec2f(1.0f, 1.0f);

                Vec3f origin = pose * Vec3f(0.0f, 0.0f, 0.0f);
                Vec3f dir = ((pose * (Vec3f(aperature_dist, normalized.y, normalized.x)).normalized()) - origin);

                // Trace
                Vec3f color;

                trace(enc, dec, origin, dir, color, cfg);

                color.x = std::min(1.0f, std::max(0.0f, color.x));
                color.y = std::min(1.0f, std::max(0.0f, color.y));
                color.z = std::min(1.0f, std::max(0.0f, color.z));
                        
                rendered.setPixel(x, renderSize.y - 1 - y, sf::Color(color.x * 255.0f, color.y * 255.0f, color.z * 255.0f));
            }
        }

        sf::Texture tex;
        tex.loadFromImage(rendered);

        sf::Sprite s;
        s.setTexture(tex);
        s.setScale(showScale, showScale);
        window.draw(s);

        window.display();
    } while (!quit);

    return 0;
}


