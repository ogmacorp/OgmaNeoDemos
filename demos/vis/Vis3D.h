// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/ImageEncoder.h>

#include <vector>
#include <string>
#include <memory>
#include <tuple>

#include <raylib.h>

#define RAYGUI_SUPPORT_ICONS

#include "raygui/raygui.h"

class Vis3D {
public:
    struct ImgEncDesc {
        int hIndex; // Index of input in hierarchy where this encoder plugs in

        aon::ImageEncoder* enc;

        std::vector<aon::ByteBuffer> imgs;

        ImgEncDesc()
        :
        hIndex(0)
        {}
    };

private:
    int winWidth;
    int winHeight;

    Camera camera;

    // Storing generated model
    std::vector<std::tuple<Vector3, Vector3, Color>> columns;
    std::vector<std::tuple<Vector3, Color>> cells;
    std::vector<std::tuple<Vector3, Vector3, Color>> lines;

    // Images being encoded (optional)
    bool hasImgs;
    std::vector<Texture2D> imgEncTextures;
    std::vector<Model> imgEncPlanes;

    // Selection
    int selectLayer;
    int selectInput;
    int selectX;
    int selectY;
    int selectZ;

    int selectLayerPrev;
    int selectInputPrev;
    int selectXPrev;
    int selectYPrev;
    int selectZPrev;

    bool showTextures;
    bool refreshTextures;

    // FF
    int ffVli;
    int ffVliRange;
    int ffZ;
    int ffZRange;

    Texture2D ffTexture;
    int ffWidth;
    int ffHeight;

    // FB

public:
    Vis3D(
        int winWidth,
        int winHeight,
        const std::string &title
    );

    ~Vis3D();

    void update(
        const aon::Hierarchy &h
    ) {
        update(h, {});
    }

    void update(
        const aon::Hierarchy &h,
        const std::vector<ImgEncDesc> &imgEncDescs
    );

    void render();
};
