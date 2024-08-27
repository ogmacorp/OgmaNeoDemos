// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/helpers.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

#include <pmmintrin.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <omp.h>
#include <memory.h>

using namespace aon;

#define RGBA32F_SIZE 16

typedef unsigned char u8;
typedef int i32;
typedef long i64;
typedef float f32;
typedef double f64;

static const __m128 c256 = _mm_set1_ps(256);

void resize_nearest_4f32(f32 src[], f32 dst[], i32 src_width, i32 src_height, i32 dst_width, i32 dst_height) {
    f32 ratio_x = (f32)src_width / (f32)dst_width;
    f32 ratio_y = (f32)src_height / (f32)dst_height;

    i32 dst_size = dst_width * dst_height;

    for (i32 dst_x = 0; dst_x < dst_width; dst_x++) {
        for (i32 dst_y = 0; dst_y < dst_height; dst_y++) {
            i32 src_x = (i32)((dst_x + 0.5f) * ratio_x);
            i32 src_y = (i32)((dst_y + 0.5f) * ratio_y);

            memcpy(&dst[4 * (dst_y + dst_height * dst_x)], &src[4 * (src_y + src_height * src_x)], RGBA32F_SIZE);
        }
    }
}

void scale_bilinear_4f32(f32 src[], f32 dst[], i32 src_width, i32 src_height, i32 dst_width, i32 dst_height) {
    f32 ratio_x = (f32)(src_width - 1) / (f32)dst_width;
    f32 ratio_y = (f32)(src_height - 1) / (f32)dst_height;

    for (i32 dst_x = 0; dst_x < dst_width; dst_x++) {
        for (i32 dst_y = 0; dst_y < dst_height; dst_y++) {
            f32 src_x_f = (dst_x + 0.5f) * ratio_x;
            f32 src_y_f = (dst_y + 0.5f) * ratio_y;
            i32 src_x = (i32)src_x_f;
            i32 src_y = (i32)src_y_f;
            f32 interp_x = src_x_f - src_x;
            f32 interp_y = src_y_f - src_y;

            i32 dst_start = 4 * (dst_y + dst_height * dst_x);

            i32 src_start00 = 4 * (src_y + src_height * src_x);
            i32 src_start01 = src_start00 + 4;
            i32 src_start10 = src_start00 + src_height * 4;
            i32 src_start11 = src_start10 + 4;

            __m128 ix = _mm_set1_ps(interp_x);
            __m128 ix1 = _mm_set1_ps(1.0f - interp_x);
            __m128 iy = _mm_set1_ps(interp_y);
            __m128 iy1 = _mm_set1_ps(1.0f - interp_y);

            __m128 p00, p01, p10, p11;
            p00 = _mm_load_ps(src + src_start00);
            p01 = _mm_load_ps(src + src_start01);
            p10 = _mm_load_ps(src + src_start10);
            p11 = _mm_load_ps(src + src_start11);

            p00 = _mm_add_ps(_mm_mul_ps(p00, ix1), _mm_mul_ps(p10, ix));
            p01 = _mm_add_ps(_mm_mul_ps(p01, ix1), _mm_mul_ps(p11, ix));

            p00 = _mm_add_ps(_mm_mul_ps(p00, iy1), _mm_mul_ps(p01, iy));

            _mm_store_ps(dst + dst_start, p00);
        }
    }
}

void scale_bilinear_4f32_nosimd(f32 src[], f32 dst[], i32 src_width, i32 src_height, i32 dst_width, i32 dst_height) {
    f32 ratio_x = (f32)(src_width - 1) / (f32)dst_width;
    f32 ratio_y = (f32)(src_height - 1) / (f32)dst_height;

    for (i32 dst_x = 0; dst_x < dst_width; dst_x++) {
        for (i32 dst_y = 0; dst_y < dst_height; dst_y++) {
            f32 src_x_f = (dst_x + 0.5f) * ratio_x;
            f32 src_y_f = (dst_y + 0.5f) * ratio_y;
            i32 src_x = (i32)src_x_f;
            i32 src_y = (i32)src_y_f;
            f32 interp_x = src_x_f - src_x;
            f32 interp_y = src_y_f - src_y;

            i32 dst_start = 4 * (dst_y + dst_height * dst_x);

            i32 src_start00 = 4 * (src_y + src_height * src_x);
            i32 src_start01 = src_start00 + 4;
            i32 src_start10 = src_start00 + src_height * 4;
            i32 src_start11 = src_start10 + 4;

            f32 interp_x1 = 1.0f - interp_x;
            f32 interp_y1 = 1.0f - interp_y;

            f32 pr0 = interp_x1 * src[src_start00    ] + interp_x * src[src_start10    ];
            f32 pr1 = interp_x1 * src[src_start01    ] + interp_x * src[src_start11    ];

            f32 pg0 = interp_x1 * src[src_start00 + 1] + interp_x * src[src_start10 + 1];
            f32 pg1 = interp_x1 * src[src_start01 + 1] + interp_x * src[src_start11 + 1];

            f32 pb0 = interp_x1 * src[src_start00 + 2] + interp_x * src[src_start10 + 2];
            f32 pb1 = interp_x1 * src[src_start01 + 2] + interp_x * src[src_start11 + 2];

            f32 pa0 = interp_x1 * src[src_start00 + 3] + interp_x * src[src_start10 + 3];
            f32 pa1 = interp_x1 * src[src_start01 + 3] + interp_x * src[src_start11 + 3];

            dst[dst_start    ] = interp_y1 * pr0 + interp_y * pr1;
            dst[dst_start + 1] = interp_y1 * pg0 + interp_y * pg1;
            dst[dst_start + 2] = interp_y1 * pb0 + interp_y * pb1;
            dst[dst_start + 3] = interp_y1 * pa0 + interp_y * pa1;
        }
    }
}

void scale_bilinear_3u8(const unsigned char src[], unsigned char dst[], int src_width, int src_height, int dst_width, int dst_height) {
    float ratio_x = (float)(src_width - 1) / (float)dst_width;
    float ratio_y = (float)(src_height - 1) / (float)dst_height;
    int dst_width3 = dst_width * 3;

    int src_width3s[8];
    src_width3s[0] = src_width * 3;
    
    for (int i = 1; i < 6; i++)
        src_width3s[i] = src_width3s[0] + i;

    for (int dst_y = 0; dst_y < dst_height; dst_y++) {
        float src_y_f = (dst_y + 0.5f) * ratio_y;
        int src_y = (int)src_y_f;
        float interp_y = src_y_f - src_y;
        float interp_y1 = 1.0f - interp_y;

        int dst_offset3 = dst_width3 * dst_y;
        int src_offset3 = src_width3s[0] * src_y;

        for (int dst_x = 0; dst_x < dst_width; dst_x++) {
            float src_x_f = (dst_x + 0.5f) * ratio_x;
            int src_x = (int)src_x_f;
            float interp_x = src_x_f - src_x;

            int dst_start = dst_x * 3 + dst_offset3;

            int src_start00 = src_x * 3 + src_offset3;

            float interp_x1 = 1.0f - interp_x;

            float pr0 = interp_y1 * src[src_start00    ] + interp_y * src[src_start00 + src_width3s[0]];
            float pr1 = interp_y1 * src[src_start00 + 3] + interp_y * src[src_start00 + src_width3s[3]];

            float pg0 = interp_y1 * src[src_start00 + 1] + interp_y * src[src_start00 + src_width3s[1]];
            float pg1 = interp_y1 * src[src_start00 + 4] + interp_y * src[src_start00 + src_width3s[4]];

            float pb0 = interp_y1 * src[src_start00 + 2] + interp_y * src[src_start00 + src_width3s[2]];
            float pb1 = interp_y1 * src[src_start00 + 5] + interp_y * src[src_start00 + src_width3s[5]];

            dst[dst_start    ] = (unsigned char)(interp_x1 * pr0 + interp_x * pr1);
            dst[dst_start + 1] = (unsigned char)(interp_x1 * pg0 + interp_x * pg1);
            dst[dst_start + 2] = (unsigned char)(interp_x1 * pb0 + interp_x * pb1);
        }
    }
}

int main(int argc, char *argv[]) {
    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    // --------------------------- Create the window(s) ---------------------------

    sf::Image src;
    src.loadFromFile("resources/test_color.png");
    sf::Image dst;
    dst.create(32, 32);

    Byte_Buffer src_data(src.getSize().x * src.getSize().y * 3);
    Byte_Buffer dst_data(dst.getSize().x * dst.getSize().y * 3);

    for (int x = 0; x < src.getSize().x; x++)
        for (int y = 0; y < src.getSize().y; y++) {
            sf::Color c = src.getPixel(x, y);

            src_data[0 + 3 * (y + x * src.getSize().y)] = c.r;
            src_data[1 + 3 * (y + x * src.getSize().y)] = c.g;
            src_data[2 + 3 * (y + x * src.getSize().y)] = c.b;
        }

    scale_bilinear_3u8(&src_data[0], &dst_data[0], src.getSize().x, src.getSize().y, dst.getSize().x, dst.getSize().y);

    for (int x = 0; x < dst.getSize().x; x++)
        for (int y = 0; y < dst.getSize().y; y++) {
            sf::Color c;
            c.r = dst_data[0 + 3 * (y + x * dst.getSize().y)];
            c.g = dst_data[1 + 3 * (y + x * dst.getSize().y)];
            c.b = dst_data[2 + 3 * (y + x * dst.getSize().y)];

            dst.setPixel(x, y, c);
        }

    dst.saveToFile("result1.png");

    //unsigned int windowWidth = 1000;
    //unsigned int windowHeight = 500;

    //sf::RenderWindow window;

    //window.create(sf::VideoMode(windowWidth, windowHeight), "Wavy Test", sf::Style::Default);

    //window.setVerticalSyncEnabled(false);
    ////window.setFramerateLimit(60);

    //vis::Plot plot;
    ////plot.backgroundColor = sf::Color(64, 64, 64, 255);
    //plot.plotXAxisTicks = true;
    //plot.curves.resize(2);
    //plot.curves[0].shadow = 0.0f; // Input
    //plot.curves[1].shadow = 0.0f; // Prediction

    //float minCurve = -1.25f;
    //float maxCurve = 1.25f;

    //sf::RenderTexture plotRT;
    //plotRT.create(windowWidth, windowHeight, false);
    //plotRT.setActive();
    //plotRT.clear(sf::Color::White);

    //sf::Texture lineGradient;
    //lineGradient.loadFromFile("resources/lineGradient.png");

    //sf::Font tickFont;
    //tickFont.loadFromFile("resources/Hack-Regular.ttf");

    //// Generate curve
    //for (int i = 0; i < 200; i++) {
    //    // Plot target data
    //    float x = i * 0.02f;

    //    vis::Point p;
    //    p.position.x = x;
    //    p.position.y = aon::powf(x, 3.0f);
    //    p.color = sf::Color::Red;

    //    plot.curves[0].points.push_back(p);

    //    p.position.y = std::pow(x, 3.0f) + 0.1f;
    //    p.color = sf::Color::Blue;

    //    plot.curves[1].points.push_back(p);
    //}

    //bool quit = false;

    //do {
    //    sf::Event event;

    //    while (window.pollEvent(event)) {
    //        switch (event.type) {
    //        case sf::Event::Closed:
    //            quit = true;
    //            break;
    //        }
    //    }

    //    if (window.hasFocus()) {
    //        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
    //            quit = true;
    //    }
    //    window.clear();

    //    plot.draw(
    //        plotRT, lineGradient, tickFont, 0.5f,
    //        sf::Vector2f(plot.curves[0].points.front().position.x, plot.curves[0].points.back().position.x),
    //        sf::Vector2f(minCurve, maxCurve), sf::Vector2f(48.0f, 48.0f),
    //        sf::Vector2f(plot.curves[0].points.back().position.x / 10.0f, (maxCurve - minCurve) / 10.0f),
    //        2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4
    //    );

    //    plotRT.display();

    //    sf::Sprite plotSprite;
    //    plotSprite.setTexture(plotRT.getTexture());

    //    window.draw(plotSprite);

    //    window.display();
    //} while (!quit);

    return 0;
}

