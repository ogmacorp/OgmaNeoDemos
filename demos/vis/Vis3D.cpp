// ----------------------------------------------------------------------------
//  PyAOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of PyAOgmaNeo is licensed to you under the terms described
//  in the PYAOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Vis3D.h"

#define RAYGUI_IMPLEMENTATION

#include "raygui/raygui.h"

#undef RAYGUI_IMPLEMENTATION

#include <unordered_map>
#include <iostream>

const Color hcellActiveColor = (Color){ 255, 64, 64, 255 };
const Color cellPredictedColor = (Color){ 64, 255, 64, 255 };
const Color cellOffColor = (Color){ 192, 192, 192, 255 };
const Color cellSelectColor = (Color){ 64, 64, 255, 255 };

const float cellRadius = 0.25f;
const float columnRadius = 0.3f;
const float layerDelta = 6.0f;
const float weightScaling = 1.0f;
const float textureScaling = 8.0f;

bool operator==(const aon::Int3 &left, const aon::Int3 &right) {
    return left.x == right.x && left.y == right.y && left.z == right.z;
}

Vis3D::Vis3D(
    int winWidth,
    int winHeight,
    const std::string &title
) {
    this->winWidth = winWidth;
    this->winHeight = winHeight;

    SetConfigFlags(FLAG_MSAA_4X_HINT);

    InitWindow(winWidth, winHeight, title.c_str());

    camera.position = (Vector3){ 30.0f, 30.0f, 30.0f };
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };
    camera.up = (Vector3){ 0.0f, 1.0f, 0.0f };
    camera.fovy = 70.0f;
    camera.type = CAMERA_PERSPECTIVE;

    SetCameraMode(camera, CAMERA_FREE);

    SetCameraAltControl(KEY_LEFT_SHIFT);

    SetTargetFPS(0);

    selectLayer = -1;
    selectInput = -1;
    selectX = -1;
    selectY = -1;
    selectZ = -1;

    selectLayerPrev = selectLayer;
    selectInputPrev = selectInput;
    selectXPrev = selectX;
    selectYPrev = selectY;
    selectZPrev = selectZ;

    showTextures = false;
    refreshTextures = false;
    hasImgs = false;

    ffVli = 0;
    ffVliRange = 0;
    ffZ = 0;
    ffZRange = 0;
}

Vis3D::~Vis3D() {
    if (showTextures)
        UnloadTexture(ffTexture);

    if (hasImgs) {
        for (int i = 0; i < imgEncPlanes.size(); i++)
            UnloadModel(imgEncPlanes[i]);
    }

    CloseWindow();
}

void Vis3D::update(
    const aon::Hierarchy &h,
    const std::vector<ImgEncDesc> &imgEncDescs
) {
    bottomMost = 0.0f;

    int oldColumnSize = columns.size();
    int oldCellsSize = cells.size();
    int oldLinesSize = lines.size();

    columns.clear();
    cells.clear();
    lines.clear();

    if (oldColumnSize > 0)
        columns.reserve(oldColumnSize);

    if (oldCellsSize > 0)
        cells.reserve(oldCellsSize);

    if (oldLinesSize > 0)
        lines.reserve(oldLinesSize);

    bool select = IsMouseButtonPressed(MOUSE_RIGHT_BUTTON);

    Ray ray = { 0 };

    float minDistance = -1.0f;
    Vector3 minPosition = (Vector3){ 0.0f, 0.0f, 0.0f };

    if (select) {
        ray = GetMouseRay(GetMousePosition(), camera);

        selectLayer = -1;
        selectInput = -1;
        selectX = -1;
        selectY = -1;
        selectZ = -1;
    }

    // Generate necessary geometry

    // Calculate full size
    float hierarchyHeight = 0.0f;

    for (int l = 0; l < h.getNumLayers(); l++)
        hierarchyHeight += (l < h.getNumLayers() - 1 ? layerDelta : 0) + h.getSCLayer(l).getHiddenSize().z;

    // Find total input layer width
    float inputWidthTotal = 0.0f;
    float maxInputHeight = 0.0f;

    for (int i = 0; i < h.getInputSizes().size(); i++) {
        inputWidthTotal += (i < h.getInputSizes().size() - 1 ? layerDelta : 0) + h.getInputSizes()[i].x;

        maxInputHeight = std::max<float>(maxInputHeight, h.getInputSizes()[i].z);
    }

    float zOffset = -hierarchyHeight * 0.5f;

    // Render input layers
    float xOffset = -inputWidthTotal * 0.5f;

    for (int i = 0; i < h.getInputSizes().size(); i++) {
        const aon::CircleBuffer<aon::IntBuffer> &hist = h.getHistories(0)[i];

        aon::IntBuffer csdr = hist[0];
        aon::IntBuffer pcsdr = h.getPredictionCIs(i);
        
        Vector3 offset = (Vector3){ -h.getInputSizes()[i].x * 0.5f + h.getInputSizes()[i].x * 0.5f + xOffset, -h.getInputSizes()[i].y * 0.5f, -h.getInputSizes()[i].z * 0.5f + zOffset - layerDelta - maxInputHeight * 0.5f};

        // Update bottom-most
        bottomMost = aon::min<float>(bottomMost, offset.z);

        // Construct columns
        for (int cx = 0; cx < h.getInputSizes()[i].x; cx++)
            for (int cy = 0; cy < h.getInputSizes()[i].y; cy++) {
                int columnIndex = aon::address2(aon::Int2(cx, cy), aon::Int2(h.getInputSizes()[i].x, h.getInputSizes()[i].y));

                int c = csdr[columnIndex];
                
                columns.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ cx + offset.x + 0.5f, offset.z + h.getInputSizes()[i].z * 0.5f - columnRadius, cy + offset.y + 0.5f }, (Vector3){ columnRadius * 2.0f, h.getInputSizes()[i].z + columnRadius * 2.0f, columnRadius * 2.0f }, (Color){0, 0, 0, 64}));
                
                Vector3 lowerBound = (Vector3){ std::get<0>(columns.back()).x - std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y - std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z - std::get<1>(columns.back()).z * 0.5f };
                Vector3 upperBound = (Vector3){ std::get<0>(columns.back()).x + std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y + std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z + std::get<1>(columns.back()).z * 0.5f };
                
                bool columnCollision = select ? CheckCollisionRayBox(ray, (BoundingBox){ lowerBound, upperBound }) : false;
                
                for (int cz = 0; cz < h.getInputSizes()[i].z; cz++) {
                    Vector3 position = (Vector3){ cx + offset.x + 0.5f, cz + offset.z, cy + offset.y + 0.5f };

                    bool cellCollision = columnCollision ? CheckCollisionRaySphere(ray, position, cellRadius) : false;

                    if (cellCollision) {
                        // If already found one, compare distance
                        if (selectX != -1 && minDistance > 0.0f) {
                            float dx = position.x - minPosition.x;
                            float dy = position.y - minPosition.y;
                            float dz = position.z - minPosition.z;

                            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                            if (dist < minDistance) {
                                minDistance = dist;

                                selectLayer = -1;
                                selectInput = i;
                                selectX = cx;
                                selectY = cy;
                                selectZ = cz;
                            }
                        }
                        else {
                            selectLayer = -1;
                            selectInput = i;
                            selectX = cx;
                            selectY = cy;
                            selectZ = cz;
                        }
                    }

                    bool isSelected = cellCollision || (selectLayer == -1 && selectInput == i && selectX == cx && selectY == cy && selectZ == cz);

                    Color color = cellOffColor;

                    if (cz == c)
                        color = (Color){ std::max(color.r, hcellActiveColor.r), std::max(color.g, hcellActiveColor.g), std::max(color.b, hcellActiveColor.b), std::max(color.a, hcellActiveColor.a) };

                    if (cz == pcsdr[columnIndex])
                        color = (Color){ std::max(color.r, cellPredictedColor.r), std::max(color.g, cellPredictedColor.g), std::max(color.b, cellPredictedColor.b), std::max(color.a, cellPredictedColor.a) };

                    cells.push_back(std::tuple<Vector3, Color>(position, (isSelected ? cellSelectColor : color)));
                }
            }

        // Line to next layer
        lines.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ xOffset + h.getInputSizes()[i].x * 0.5f, zOffset - layerDelta - (maxInputHeight - h.getInputSizes()[i].z) * 0.5f, 0.0f }, (Vector3){ 0.0f, zOffset, 0.0f }, (Color){ 0, 0, 0, 128 }));

        xOffset += layerDelta + h.getInputSizes()[i].x;
    }

    for (int l = 0; l < h.getNumLayers(); l++) {
        aon::IntBuffer hcsdr = h.getSCLayer(l).getHiddenCIs();
        aon::IntBuffer pcsdr;
        
        if (l < h.getNumLayers() - 1)
            pcsdr = h.getPLayers(l + 1)[h.getTicksPerUpdate(l + 1) - 1 - h.getTicks(l + 1)]->getHiddenCIs();

        Vector3 offset = (Vector3){ -h.getSCLayer(l).getHiddenSize().x * 0.5f, -h.getSCLayer(l).getHiddenSize().y * 0.5f, zOffset };

        // Construct columns
        for (int cx = 0; cx < h.getSCLayer(l).getHiddenSize().x; cx++)
            for (int cy = 0; cy < h.getSCLayer(l).getHiddenSize().y; cy++) {
                int columnIndex = aon::address2(aon::Int2(cx, cy), aon::Int2(h.getSCLayer(l).getHiddenSize().x, h.getSCLayer(l).getHiddenSize().y));

                int hc = hcsdr[columnIndex];

                columns.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ cx + offset.x + 0.5f, offset.z + h.getSCLayer(l).getHiddenSize().z * 0.5f - columnRadius, cy + offset.y + 0.5f }, (Vector3){ columnRadius * 2.0f, h.getSCLayer(l).getHiddenSize().z + columnRadius * 2.0f, columnRadius * 2.0f }, (Color){0, 0, 0, 64}));
                
                Vector3 lowerBound = (Vector3){ std::get<0>(columns.back()).x - std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y - std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z - std::get<1>(columns.back()).z * 0.5f };
                Vector3 upperBound = (Vector3){ std::get<0>(columns.back()).x + std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y + std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z + std::get<1>(columns.back()).z * 0.5f };
                
                bool columnCollision = select ? CheckCollisionRayBox(ray, (BoundingBox){ lowerBound, upperBound }) : false;
                
                for (int cz = 0; cz < h.getSCLayer(l).getHiddenSize().z; cz++) {
                    Vector3 position = (Vector3){ cx + offset.x + 0.5f, cz + offset.z, cy + offset.y + 0.5f };

                    bool cellCollision = columnCollision ? CheckCollisionRaySphere(ray, position, cellRadius) : false;

                    if (cellCollision) {
                        // If already found one, compare distance
                        if (selectX != -1 && minDistance > 0.0f) {
                            float dx = position.x - minPosition.x;
                            float dy = position.y - minPosition.y;
                            float dz = position.z - minPosition.z;

                            float dist = std::sqrt(dx * dx + dy * dy + dz * dz);

                            if (dist < minDistance) {
                                minDistance = dist;
                                
                                selectLayer = l;
                                selectInput = 0;
                                selectX = cx;
                                selectY = cy;
                                selectZ = cz;
                            }
                        }
                        else {
                            selectLayer = l;
                            selectInput = 0;
                            selectX = cx;
                            selectY = cy;
                            selectZ = cz;
                        }
                    }

                    bool isSelected = cellCollision || (selectLayer == l && selectInput == 0 && selectX == cx && selectY == cy && selectZ == cz);

                    Color color = cellOffColor;

                    if (cz == hc)
                        color = (Color){ std::max(color.r, hcellActiveColor.r), std::max(color.g, hcellActiveColor.g), std::max(color.b, hcellActiveColor.b), std::max(color.a, hcellActiveColor.a) };

                    if (l < h.getNumLayers() - 1 && cz == pcsdr[columnIndex])
                        color = (Color){ std::max(color.r, cellPredictedColor.r), std::max(color.g, cellPredictedColor.g), std::max(color.b, cellPredictedColor.b), std::max(color.a, cellPredictedColor.a) };

                    cells.push_back(std::tuple<Vector3, Color>(position, (isSelected ? cellSelectColor : color)));
                }
            }

        if (l < h.getNumLayers() - 1)
            lines.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ 0.0f, zOffset + h.getSCLayer(l).getHiddenSize().z, 0.0f }, (Vector3){ 0.0f, zOffset + h.getSCLayer(l).getHiddenSize().z + layerDelta, 0.0f }, (Color){ 0, 0, 0, 128 }));

        zOffset += layerDelta + h.getSCLayer(l).getHiddenSize().z;
    }

    // Display active cell receptive fields
    bool changed = selectLayer != selectLayerPrev || selectInput != selectInputPrev || selectX != selectXPrev || selectY != selectYPrev || selectZ != selectZPrev;

    if (selectX != -1 && changed)
        refreshTextures = true;
    else if (selectX == -1)
        showTextures = false;

    if (refreshTextures) {
        // FF
        if (selectLayer >= 0) {
            ffVliRange = h.getSCLayer(selectLayer).getNumVisibleLayers();

            // Clamp
            ffVli = aon::min(ffVli, ffVliRange - 1);

            const aon::SparseCoder::VisibleLayer &hvl = h.getSCLayer(selectLayer).getVisibleLayer(ffVli);
            const aon::SparseCoder::VisibleLayerDesc &hvld = h.getSCLayer(selectLayer).getVisibleLayerDesc(ffVli);

            aon::Int3 hiddenSize = h.getSCLayer(selectLayer).getHiddenSize();
            int hiddenIndex = aon::address3(aon::Int3(selectX, selectY, selectZ), hiddenSize);

            ffZRange = hvld.size.z;

            // Clamp
            ffZ = aon::min(ffZ, ffZRange - 1);

            int diam = hvld.radius * 2 + 1;

            // Projection
            aon::Float2 hToV = aon::Float2(static_cast<float>(hvld.size.x) / static_cast<float>(hiddenSize.x),
                static_cast<float>(hvld.size.y) / static_cast<float>(hiddenSize.y));

            aon::Int2 visibleCenter = project(aon::Int2(selectX, selectY), hToV);

            // Lower corner
            aon::Int2 fieldLowerBound(visibleCenter.x - hvld.radius, visibleCenter.y - hvld.radius);

            // Bounds of receptive field, clamped to input size
            aon::Int2 iterLowerBound(aon::max(0, fieldLowerBound.x), aon::max(0, fieldLowerBound.y));
            aon::Int2 iterUpperBound(aon::min(hvld.size.x - 1, visibleCenter.x + hvld.radius), aon::min(hvld.size.y - 1, visibleCenter.y + hvld.radius));

            int width = diam;
            int height = diam;

            aon::Array<Color> colors(width * height, (Color){ 0, 0, 0, 255 });

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = aon::address2(aon::Int2(ix, iy), aon::Int2(hvld.size.x, hvld.size.y));

                    aon::Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    unsigned char hc = (int)(aon::sigmoid(hvl.weights[ffZ + hvld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))]) * weightScaling * 255.0f);

                    colors[offset.y + offset.x * diam] = (Color){ hc, 0, 0, 255 };
                }

            // Load image
            Image image = { 0 };
            image.data = NULL;
            image.width = width;
            image.height = height;
            image.mipmaps = 1;
            image.format = UNCOMPRESSED_R8G8B8A8;

            int k = 0;

            image.data = (unsigned char*)RL_MALLOC(image.width * image.height * 4 * sizeof(unsigned char));

            for (int i = 0; i < image.width * image.height * 4; i += 4) {
                ((unsigned char*)image.data)[i] = colors[k].r;
                ((unsigned char*)image.data)[i + 1] = colors[k].g;
                ((unsigned char*)image.data)[i + 2] = colors[k].b;
                ((unsigned char*)image.data)[i + 3] = colors[k].a;
                k++;
            }

            // Load texture
            if (showTextures) // Already have, just update
                UpdateTexture(ffTexture, image.data);
            else
                ffTexture = LoadTextureFromImage(image);

            ffWidth = width;
            ffHeight = height;

            // Unload image
            UnloadImage(image);

            showTextures = true;
        }
        else { // Image encoders
            // If there is an image encoder on this input
            aon::ImageEncoder* enc = nullptr;

            for (int i = 0; i < imgEncDescs.size(); i++) {
                if (imgEncDescs[i].hIndex == selectInput) {
                    enc = imgEncDescs[i].enc;
                    break;
                }
            }

            if (enc != nullptr) {
                ffVliRange = enc->getNumVisibleLayers();

                // Clamp
                ffVli = aon::min(ffVli, ffVliRange - 1);

                const aon::ImageEncoder::VisibleLayer &vl = enc->getVisibleLayer(ffVli);
                const aon::ImageEncoder::VisibleLayerDesc &vld = enc->getVisibleLayerDesc(ffVli);

                aon::Int3 hiddenSize = enc->getHiddenSize();
                int hiddenIndex = aon::address3(aon::Int3(selectX, selectY, selectZ), hiddenSize);

                ffZRange = vld.size.z;

                // Clamp
                ffZ = aon::min(ffZ, ffZRange - 1);

                int diam = vld.radius * 2 + 1;

                // Projection
                aon::Float2 hToV = aon::Float2(static_cast<float>(vld.size.x) / static_cast<float>(hiddenSize.x),
                    static_cast<float>(vld.size.y) / static_cast<float>(hiddenSize.y));

                aon::Int2 visibleCenter = project(aon::Int2(selectX, selectY), hToV);

                // Lower corner
                aon::Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

                // Bounds of receptive field, clamped to input size
                aon::Int2 iterLowerBound(aon::max(0, fieldLowerBound.x), aon::max(0, fieldLowerBound.y));
                aon::Int2 iterUpperBound(aon::min(vld.size.x - 1, visibleCenter.x + vld.radius), aon::min(vld.size.y - 1, visibleCenter.y + vld.radius));

                int width = diam;
                int height = diam;

                aon::Array<Color> colors(width * height, (Color){ 0, 0, 0, 255 });

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = aon::address2(aon::Int2(ix, iy), aon::Int2(vld.size.x,  vld.size.y));

                        aon::Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        if (vld.size.z == 2) {
                            unsigned char r = vl.protos[0 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                            unsigned char g = vl.protos[1 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];

                            colors[offset.y + offset.x * diam] = (Color){ r, g, 0, 255 };
                        }
                        else if (vld.size.z == 3) {
                            unsigned char r = vl.protos[0 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                            unsigned char g = vl.protos[1 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                            unsigned char b = vl.protos[2 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];

                            colors[offset.y + offset.x * diam] = (Color){ r, g, b, 255 };
                        }
                        else {
                            unsigned char c = vl.protos[ffZ + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];

                            colors[offset.y + offset.x * diam] = (Color){ c, c, c, 255 };
                        }
                    }

                // Load image
                Image image = { 0 };
                image.data = NULL;
                image.width = width;
                image.height = height;
                image.mipmaps = 1;
                image.format = UNCOMPRESSED_R8G8B8A8;

                int k = 0;

                image.data = (unsigned char*)RL_MALLOC(image.width * image.height * 4 * sizeof(unsigned char));

                for (int i = 0; i < image.width * image.height * 4; i += 4) {
                    ((unsigned char*)image.data)[i] = colors[k].r;
                    ((unsigned char*)image.data)[i + 1] = colors[k].g;
                    ((unsigned char*)image.data)[i + 2] = colors[k].b;
                    ((unsigned char*)image.data)[i + 3] = colors[k].a;
                    k++;
                }

                // Load texture
                if (showTextures) // Already have, just update
                    UpdateTexture(ffTexture, image.data);
                else
                    ffTexture = LoadTextureFromImage(image);

                ffWidth = width;
                ffHeight = height;

                // Unload image
                UnloadImage(image);

                showTextures = true;
            }
        }
    }

    int totalNumImgs = 0;

    for (int ei = 0; ei < imgEncDescs.size(); ei++)
        totalNumImgs += imgEncDescs[ei].imgs.size();

    imgEncTextures.resize(totalNumImgs);
    imgEncPlanes.resize(totalNumImgs);
    int imgIndex = 0;

    for (int ei = 0; ei < imgEncDescs.size(); ei++) {
        int numImgs = imgEncDescs[ei].imgs.size();

        for (int ii = 0; ii < numImgs; ii++) {
            aon::Int3 imgSize = imgEncDescs[ei].enc->getVisibleLayerDesc(ii).size;

            int imgWidth = imgSize.x;
            int imgHeight = imgSize.y;
            int imgDepth = imgSize.z;

            aon::Array<Color> colors(imgWidth * imgHeight, (Color){ 0, 0, 0, 255 });

            if (imgDepth == 1) {
                for (int x = 0; x < imgWidth; x++)
                    for (int y = 0; y < imgHeight; y++) {
                        unsigned char c = imgEncDescs[ei].imgs[ii][y + imgHeight * x];

                        colors[x + imgWidth * y] = (Color){ c, c, c, 255 };
                    }
            }
            else if (imgDepth == 2) {
                for (int x = 0; x < imgWidth; x++)
                    for (int y = 0; y < imgHeight; y++) {
                        unsigned char r = imgEncDescs[ei].imgs[ii][0 + 2 * (y + imgHeight * x)];
                        unsigned char g = imgEncDescs[ei].imgs[ii][1 + 2 * (y + imgHeight * x)];

                        colors[x + imgWidth * y] = (Color){ r, g, 0, 255 };
                    }
            }
            else if (imgDepth == 3) {
                for (int x = 0; x < imgWidth; x++)
                    for (int y = 0; y < imgHeight; y++) {
                        unsigned char r = imgEncDescs[ei].imgs[ii][0 + 3 * (y + imgHeight * x)];
                        unsigned char g = imgEncDescs[ei].imgs[ii][1 + 3 * (y + imgHeight * x)];
                        unsigned char b = imgEncDescs[ei].imgs[ii][2 + 3 * (y + imgHeight * x)];

                        colors[x + imgWidth * y] = (Color){ r, g, b, 255 };
                    }
            }

            // Load im enc texture
            Image image = { 0 };
            image.data = NULL;
            image.width = imgWidth;
            image.height = imgHeight;
            image.mipmaps = 1;
            image.format = UNCOMPRESSED_R8G8B8A8;

            int k = 0;

            image.data = (unsigned char*)RL_MALLOC(image.width * image.height * 4 * sizeof(unsigned char));

            for (int i = 0; i < image.width * image.height * 4; i += 4) {
                ((unsigned char*)image.data)[i] = colors[k].r;
                ((unsigned char*)image.data)[i + 1] = colors[k].g;
                ((unsigned char*)image.data)[i + 2] = colors[k].b;
                ((unsigned char*)image.data)[i + 3] = colors[k].a;
                k++;
            }

            if (hasImgs)
                UpdateTexture(imgEncTextures[imgIndex], image.data);
            else {
                imgEncTextures[imgIndex] = LoadTextureFromImage(image);

                Mesh m = GenMeshPlane(30.0f, 30.0f, 1, 1);

                imgEncPlanes[imgIndex] = LoadModelFromMesh(m);

                //UnloadMesh(m);

                imgEncPlanes[imgIndex].materials[0].maps[MAP_DIFFUSE].texture = imgEncTextures[imgIndex];
            }

            UnloadImage(image);

            imgIndex++;
        }
    }

    if (totalNumImgs != 0)
        hasImgs = true;

    selectLayerPrev = selectLayer;
    selectInputPrev = selectInput;
    selectXPrev = selectX;
    selectYPrev = selectY;
    selectZPrev = selectZ;
}

void Vis3D::render() {
    UpdateCamera(&camera);

    BeginDrawing();

        ClearBackground(RAYWHITE);

        BeginMode3D(camera);

            for (int i = 0; i < cells.size(); i++)
                DrawSphereEx(std::get<0>(cells[i]), cellRadius, 6, 6, std::get<1>(cells[i]));

            for (int i = 0; i < columns.size(); i++)
                DrawCubeWiresV(std::get<0>(columns[i]), std::get<1>(columns[i]), std::get<2>(columns[i]));

            for (int i = 0; i < lines.size(); i++)
                DrawLine3D(std::get<0>(lines[i]), std::get<1>(lines[i]), std::get<2>(lines[i]));

            if (hasImgs) {
                for (int ii = 0; ii < imgEncTextures.size(); ii++)
                    DrawModel(imgEncPlanes[ii], (Vector3){ static_cast<float>(ii - imgEncTextures.size() / 2) * 60.0f, bottomMost - 10.0f, 0.0f }, 1.0f, (Color){ 255, 255, 255, 255 });
            }

        EndMode3D();

        DrawRectangle( 10, 10, 290, 60, Fade(SKYBLUE, 0.5f));
        DrawRectangleLines( 10, 10, 290, 60, BLUE);

        DrawText("Middle mouse button + move mouse -> pan", 20, 20, 8, DARKGRAY);
        DrawText("Shift + middle mouse button + move mouse -> rotate", 20, 30, 8, DARKGRAY);
        DrawText("Scroll wheel -> zoom", 20, 40, 8, DARKGRAY);
        DrawText("Right click on cell -> select", 20, 50, 8, DARKGRAY);

        GuiEnable();

        if (showTextures) {
            DrawTextureEx(ffTexture, (Vector2){ 10, winHeight - 80 - ffHeight * textureScaling}, 0.0f, textureScaling, (Color){ 255, 255, 255, 255 });

            int oldffVli = ffVli;
            int oldffZ = ffZ;

            ffVli = GuiSlider((Rectangle){ 20, static_cast<float>(winHeight) - 30, 200, 20 }, "Vli", TextFormat("%i", ffVli), ffVli, 0, ffVliRange);
            ffZ = GuiSlider((Rectangle){ 20, static_cast<float>(winHeight) - 60, 200, 20 }, "Z", TextFormat("%i", ffZ), ffZ, 0, ffZRange);

            refreshTextures = ffVli != oldffVli || ffZ != oldffZ;
        }

    EndDrawing();
}
