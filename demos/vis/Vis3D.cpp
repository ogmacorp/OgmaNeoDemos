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

#include <unordered_map>
#include <iostream>

const Color hcellActiveColor = (Color){ 255, 64, 64, 255 };
const Color cellPredictedColor = (Color){ 64, 255, 64, 255 };
const Color cellOffColor = (Color){ 192, 192, 192, 16 };
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
    camera.projection = CAMERA_PERSPECTIVE;

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
    const aon::Array<aon::Int_Buffer_View> &inputCIs,
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

        //selectLayer = -1;
        //selectInput = -1;
        //selectX = -1;
        //selectY = -1;
        //selectZ = -1;
    }

    // Generate necessary geometry

    // Calculate full size
    float hierarchyHeight = 0.0f;

    for (int l = 0; l < h.get_num_layers(); l++)
        hierarchyHeight += (l < h.get_num_layers() - 1 ? layerDelta : 0) + h.get_encoder(l).get_hidden_size().z;

    // Find total input layer width
    float inputWidthTotal = 0.0f;
    float maxInputHeight = 0.0f;

    for (int i = 0; i < h.get_num_io(); i++) {
        inputWidthTotal += (i < h.get_num_io() - 1 ? layerDelta : 0) + h.get_io_size(i).x;

        maxInputHeight = std::max<float>(maxInputHeight, h.get_io_size(i).z);
    }

    float zOffset = -hierarchyHeight * 0.5f;

    // Render input layers
    float xOffset = -inputWidthTotal * 0.5f;

    for (int i = 0; i < h.get_num_io(); i++) {
        aon::Int_Buffer_View csdr = inputCIs[i];
        aon::Int_Buffer pcsdr = h.get_prediction_cis(i);
        
        Vector3 offset = (Vector3){ -h.get_io_size(i).x * 0.5f + h.get_io_size(i).x * 0.5f + xOffset, -h.get_io_size(i).y * 0.5f, -h.get_io_size(i).z * 0.5f + zOffset - layerDelta - maxInputHeight * 0.5f};

        // Update bottom-most
        bottomMost = aon::min<float>(bottomMost, offset.z);

        // Construct columns
        for (int cx = 0; cx < h.get_io_size(i).x; cx++)
            for (int cy = 0; cy < h.get_io_size(i).y; cy++) {
                int columnIndex = aon::address2(aon::Int2(cx, cy), aon::Int2(h.get_io_size(i).x, h.get_io_size(i).y));

                int c = csdr[columnIndex];
                
                columns.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ cx + offset.x + 0.5f, offset.z + h.get_io_size(i).z * 0.5f - columnRadius, cy + offset.y + 0.5f }, (Vector3){ columnRadius * 2.0f, h.get_io_size(i).z + columnRadius * 2.0f, columnRadius * 2.0f }, (Color){255, 255, 255, 16}));
                
                Vector3 lowerBound = (Vector3){ std::get<0>(columns.back()).x - std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y - std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z - std::get<1>(columns.back()).z * 0.5f };
                Vector3 upperBound = (Vector3){ std::get<0>(columns.back()).x + std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y + std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z + std::get<1>(columns.back()).z * 0.5f };
                
                bool columnCollision = select ? GetRayCollisionBox(ray, (BoundingBox){ lowerBound, upperBound }).hit : false;
                
                for (int cz = 0; cz < h.get_io_size(i).z; cz++) {
                    Vector3 position = (Vector3){ cx + offset.x + 0.5f, cz + offset.z, cy + offset.y + 0.5f };

                    bool cellCollision = columnCollision ? GetRayCollisionSphere(ray, position, cellRadius).hit : false;

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
        lines.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ xOffset + h.get_io_size(i).x * 0.5f, zOffset - layerDelta - (maxInputHeight - h.get_io_size(i).z) * 0.5f, 0.0f }, (Vector3){ 0.0f, zOffset, 0.0f }, (Color){ 255, 255, 255, 64 }));

        xOffset += layerDelta + h.get_io_size(i).x;
    }

    for (int l = 0; l < h.get_num_layers(); l++) {
        aon::Int_Buffer hcsdr = h.get_encoder(l).get_hidden_cis();
        aon::Int_Buffer pcsdr;
        
        if (l < h.get_num_layers() - 1) {
            int numInputs = h.get_histories(0).size() * h.get_histories(0)[0].size();
            pcsdr = h.get_encoder(l + 1).get_visible_layer(numInputs + h.get_ticks_per_update(l + 1) - 1 - h.get_ticks(l + 1)).recon_cis;
            //pcsdr = h.get_decoder(l + 1, h.get_ticks_per_update(l + 1) - 1 - h.get_ticks(l + 1)).get_hidden_cis();
            //pcsdr = h.get_decoder(l + 1, 0).get_hidden_cis();
        }

        Vector3 offset = (Vector3){ -h.get_encoder(l).get_hidden_size().x * 0.5f, -h.get_encoder(l).get_hidden_size().y * 0.5f, zOffset };

        // Construct columns
        for (int cx = 0; cx < h.get_encoder(l).get_hidden_size().x; cx++)
            for (int cy = 0; cy < h.get_encoder(l).get_hidden_size().y; cy++) {
                int columnIndex = aon::address2(aon::Int2(cx, cy), aon::Int2(h.get_encoder(l).get_hidden_size().x, h.get_encoder(l).get_hidden_size().y));

                int hc = hcsdr[columnIndex];

                columns.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ cx + offset.x + 0.5f, offset.z + h.get_encoder(l).get_hidden_size().z * 0.5f - columnRadius, cy + offset.y + 0.5f }, (Vector3){ columnRadius * 2.0f, h.get_encoder(l).get_hidden_size().z + columnRadius * 2.0f, columnRadius * 2.0f }, (Color){255, 255, 255, 16}));
                
                Vector3 lowerBound = (Vector3){ std::get<0>(columns.back()).x - std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y - std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z - std::get<1>(columns.back()).z * 0.5f };
                Vector3 upperBound = (Vector3){ std::get<0>(columns.back()).x + std::get<1>(columns.back()).x * 0.5f, std::get<0>(columns.back()).y + std::get<1>(columns.back()).y * 0.5f, std::get<0>(columns.back()).z + std::get<1>(columns.back()).z * 0.5f };
                
                bool columnCollision = select ? GetRayCollisionBox(ray, (BoundingBox){ lowerBound, upperBound }).hit : false;
                
                for (int cz = 0; cz < h.get_encoder(l).get_hidden_size().z; cz++) {
                    Vector3 position = (Vector3){ cx + offset.x + 0.5f, cz + offset.z, cy + offset.y + 0.5f };

                    bool cellCollision = columnCollision ? GetRayCollisionSphere(ray, position, cellRadius).hit : false;

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

                    if (l < h.get_num_layers() - 1 && cz == pcsdr[columnIndex])
                        color = (Color){ std::max(color.r, cellPredictedColor.r), std::max(color.g, cellPredictedColor.g), std::max(color.b, cellPredictedColor.b), std::max(color.a, cellPredictedColor.a) };

                    cells.push_back(std::tuple<Vector3, Color>(position, (isSelected ? cellSelectColor : color)));
                }
            }

        if (l < h.get_num_layers() - 1)
            lines.push_back(std::tuple<Vector3, Vector3, Color>((Vector3){ 0.0f, zOffset + h.get_encoder(l).get_hidden_size().z, 0.0f }, (Vector3){ 0.0f, zOffset + h.get_encoder(l).get_hidden_size().z + layerDelta, 0.0f }, (Color){ 255, 255, 255, 64 }));

        zOffset += layerDelta + h.get_encoder(l).get_hidden_size().z;
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
            ffVliRange = h.get_encoder(selectLayer).get_num_visible_layers();

            // Clamp
            ffVli = aon::min(ffVli, ffVliRange - 1);

            const aon::Encoder::Visible_Layer &hvl = h.get_encoder(selectLayer).get_visible_layer(ffVli);
            const aon::Encoder::Visible_Layer_Desc &hvld = h.get_encoder(selectLayer).get_visible_layer_desc(ffVli);

            aon::Int3 hiddenSize = h.get_encoder(selectLayer).get_hidden_size();
            int hiddenColumn = aon::address2(aon::Int2(selectX, selectY), aon::Int2(hiddenSize.x, hiddenSize.y));
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
            ffWeights.resize(colors.size(), 0.0f);

            for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                    int visibleColumnIndex = aon::address2(aon::Int2(ix, iy), aon::Int2(hvld.size.x, hvld.size.y));

                    aon::Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                    int wi = ffZ + hvld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex));

                    float w = hvl.weights[wi] / 255.0f;

                    ffWeights[offset.y + offset.x * diam] = w; 

                    unsigned char wc = hvl.weights[wi];

                    //int wi = offset.y + diam * (offset.x + diam * hiddenIndex);

                    //int index = hvl.weight_indices[wi];
                    //float w = hvl.weights[wi] / 255.0f;

                    //ffWeights[offset.y + offset.x * diam] = (ffZ == index ? w : 0); 

                    //unsigned char wc = (ffZ == index ? hvl.weights[wi] : 0);

                    //int wi = selectZ + hiddenSize.z * (offset.y + diam * (offset.x + diam * hiddenColumn));

                    //float w = hvl.weights[wi];

                    //ffWeights[offset.y + offset.x * diam] = w; 

                    //unsigned char wc = hvl.weights[wi] * 255.0f;

                    colors[offset.y + offset.x * diam] = (Color){ wc, 0, 0, 255 };
                }

            // Load image
            Image image = { 0 };
            image.data = NULL;
            image.width = width;
            image.height = height;
            image.mipmaps = 1;
            image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

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
            if (showTextures && !(selectLayer != selectLayerPrev || selectInput != selectInputPrev)) // Already have, just update
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
            aon::Image_Encoder* enc = nullptr;

            for (int i = 0; i < imgEncDescs.size(); i++) {
                if (imgEncDescs[i].hIndex == selectInput) {
                    enc = imgEncDescs[i].enc;
                    break;
                }
            }

            if (enc != nullptr) {
                ffVliRange = enc->get_num_visible_layers();

                // Clamp
                ffVli = aon::min(ffVli, ffVliRange - 1);

                const aon::Image_Encoder::Visible_Layer &vl = enc->get_visible_layer(ffVli);
                const aon::Image_Encoder::Visible_Layer_Desc &vld = enc->get_visible_layer_desc(ffVli);

                aon::Int3 hiddenSize = enc->get_hidden_size();
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
                            unsigned char r = vl.weights0[0 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                            unsigned char g = vl.weights0[1 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];

                            colors[offset.y + offset.x * diam] = (Color){ r, g, 0, 255 };
                        }
                        else if (vld.size.z == 3) {
                            unsigned char r = vl.weights0[0 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                            unsigned char g = vl.weights0[1 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];
                            unsigned char b = vl.weights0[2 + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];

                            colors[offset.y + offset.x * diam] = (Color){ r, g, b, 255 };
                        }
                        else {
                            unsigned char c = vl.weights0[ffZ + vld.size.z * (offset.y + diam * (offset.x + diam * hiddenIndex))];

                            colors[offset.y + offset.x * diam] = (Color){ c, c, c, 255 };
                        }
                    }

                // Load image
                Image image = { 0 };
                image.data = NULL;
                image.width = width;
                image.height = height;
                image.mipmaps = 1;
                image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

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
                if (showTextures && !(selectLayer != selectLayerPrev || selectInput != selectInputPrev)) // Already have, just update
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
            aon::Int3 imgSize = imgEncDescs[ei].enc->get_visible_layer_desc(ii).size;

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
            image.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

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

                imgEncPlanes[imgIndex].materials[0].maps[MATERIAL_MAP_DIFFUSE].texture = imgEncTextures[imgIndex];
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
    UpdateCamera(&camera, CAMERA_THIRD_PERSON);

    BeginDrawing();

        ClearBackground(Color{ 33, 33, 33, 255 });

        BeginMode3D(camera);

            for (int i = 0; i < cells.size(); i++)
                DrawSphereEx(std::get<0>(cells[i]), cellRadius, 4, 4, std::get<1>(cells[i]));

            for (int i = 0; i < columns.size(); i++)
                DrawCubeWiresV(std::get<0>(columns[i]), std::get<1>(columns[i]), std::get<2>(columns[i]));

            for (int i = 0; i < lines.size(); i++)
                DrawLine3D(std::get<0>(lines[i]), std::get<1>(lines[i]), std::get<2>(lines[i]));

            if (hasImgs) {
                for (int ii = 0; ii < imgEncTextures.size(); ii++)
                    DrawModel(imgEncPlanes[ii], (Vector3){ static_cast<float>(ii - imgEncTextures.size() / 2) * 60.0f, bottomMost - 10.0f, 0.0f }, 1.0f, (Color){ 255, 255, 255, 255 });
            }

        EndMode3D();

        DrawRectangle( 10, 10, 290, 60, Color{ 128, 128, 128, 32 });
        DrawRectangleLines( 10, 10, 290, 60, Color{ 255, 255, 255, 32 });

        DrawText("Middle mouse button + move mouse -> pan", 20, 20, 8, DARKGRAY);
        DrawText("Shift + middle mouse button + move mouse -> rotate", 20, 30, 8, DARKGRAY);
        DrawText("Scroll wheel -> zoom", 20, 40, 8, DARKGRAY);
        DrawText("Right click on cell -> select", 20, 50, 8, DARKGRAY);

        GuiEnable();

        if (showTextures) {
            Vector2 mousePos = GetMousePosition();

            Rectangle texRec;
            texRec.width = ffTexture.width * textureScaling;
            texRec.height = ffTexture.height * textureScaling;
            texRec.x = 10;
            texRec.y = winHeight - 80 - ffHeight * textureScaling;

            if (CheckCollisionPointRec(mousePos, texRec)) {
                int wx = (mousePos.x - texRec.x) / textureScaling;
                int wy = (mousePos.y - texRec.y) / textureScaling;

                float weight = ffWeights[wx + wy * ffTexture.width];

                DrawText(std::to_string(weight).c_str(), 10, texRec.y - 40, 24, (Color){ 255, 255, 255, 255 });
            }

            DrawTextureEx(ffTexture, (Vector2){ texRec.x, texRec.y }, 0.0f, textureScaling, (Color){ 255, 255, 255, 255 });

            int oldffVli = ffVli;
            int oldffZ = ffZ;

            ffVli = GuiSlider((Rectangle){ 20, static_cast<float>(winHeight) - 30, 200, 20 }, "Vli", TextFormat("%i", ffVli), ffVli, 0, ffVliRange);
            ffZ = GuiSlider((Rectangle){ 20, static_cast<float>(winHeight) - 60, 200, 20 }, "Z", TextFormat("%i", ffZ), ffZ, 0, ffZRange);

            refreshTextures = ffVli != oldffVli || ffZ != oldffZ;
        }

    EndDrawing();
}
