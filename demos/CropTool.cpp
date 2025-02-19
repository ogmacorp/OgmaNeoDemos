// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>

#include <cmath>

using namespace cv;

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <vector>
 
#include <aogmaneo/helpers.h>

using namespace aon;

float gumble(float x, float mean, float scale) {
    return 0.0f;
    //return mean - scale * logf(-logf(max(limit_small, x)));
}

template<int Height = 5, int BarWidth = 1, int Padding = 1, int Offset = 0, class Seq>
void draw_vbars(Seq&& s, const bool DrawMinMax = true)
{
    static_assert(0 < Height and 0 < BarWidth and 0 <= Padding and 0 <= Offset);
 
    auto cout_n = [](auto&& v, int n = 1)
    {
        while (n-- > 0)
            std::cout << v;
    };
 
    const auto [min, max] = std::minmax_element(std::cbegin(s), std::cend(s));
 
    std::vector<std::div_t> qr;
    for (typedef decltype(*std::cbegin(s)) V; V e : s)
        qr.push_back(std::div(std::lerp(V(0), 8 * Height,
                                        (e - *min) / (*max - *min)), 8));
 
    for (auto h{Height}; h-- > 0; cout_n('\n'))
    {
        cout_n(' ', Offset);
 
        for (auto dv : qr)
        {
            const auto q{dv.quot}, r{dv.rem};
            unsigned char d[]{0xe2, 0x96, 0x88, 0}; // Full Block: '█'
            q < h ? d[0] = ' ', d[1] = 0 : q == h ? d[2] -= (7 - r) : 0;
            cout_n(d, BarWidth), cout_n(' ', Padding);
        }
 
        if (DrawMinMax && Height > 1)
            Height - 1 == h ? std::cout << "┬ " << *max:
                          h ? std::cout << "│ "
                            : std::cout << "┴ " << *min;
    }
}
 
int main()
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
 
    std::extreme_value_distribution<> d{0.0f, 1.0f};
 
    const int norm = 10'000;
    const float cutoff = 0.000'3f;
 
    std::map<int, int> hist{};
    for (int n = 0; n != norm; ++n)
        ++hist[std::round(gumble(randf(), 0.0f, 1.0f))];
 
    std::vector<float> bars;
    std::vector<int> indices;
    for (const auto& [n, p] : hist)
        if (const float x = p * (1.0f / norm); x > cutoff)
        {
            bars.push_back(x);
            indices.push_back(n);
        }
 
    draw_vbars<8,4>(bars);
 
    for (int n : indices)
        std::cout << ' ' << std::setw(2) << n << "  ";
    std::cout << '\n';
}

//int main(int argc, char* argv[]) {
//    if (argc != 2) {
//        std::cout << "Requires a single argument (file name)." << std::endl;
//
//        return 0;
//    }
//
//    std::string file_name = argv[1];
//
//    // Open the video file
//    VideoCapture capture(file_name);
//    Mat frame;
//
//    if (!capture.isOpened()) {
//        std::cout << "Could not open capture: " << file_name << std::endl;
//
//        return 0;
//    }
//
//    const int video_width = static_cast<int>(capture.get(CAP_PROP_FRAME_WIDTH));
//    const int video_height = static_cast<int>(capture.get(CAP_PROP_FRAME_HEIGHT));
//    const int video_length = static_cast<int>(capture.get(CAP_PROP_FRAME_COUNT));
//
//    // Initialize a random number generator
//    std::mt19937 rng(time(nullptr));
//
//    const unsigned int window_width = 1024;
//    const unsigned int window_height = 1024;
//
//    sf::RenderWindow window;
//
//    window.create(sf::VideoMode(window_width, window_height), "Crop Tool", sf::Style::Default);
//
//    // Uncap framerate
//    window.setFramerateLimit(60);
//
//    sf::Font font;
//    font.loadFromFile("resources/Hack-Regular.ttf");
//
//    bool quit = false;
//    
//    int fi = 0;
//
//    float lower_x = 0.0f;
//    float lower_y = 0.0f;
//    float upper_x = 0.0f;
//    float upper_y = 0.0f;
//
//    do {
//        // ----------------------------- Input -----------------------------
//
//        sf::Event window_event;
//
//        while (window.pollEvent(window_event)) {
//            switch (window_event.type) {
//            case sf::Event::Closed:
//                quit = true;
//                break;
//            }
//        }
//
//        if (window.hasFocus()) {
//            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Q))
//                quit = true;
//
//            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
//                sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
//
//                lower_x = mouse_pos.x / static_cast<float>(window_width);
//                lower_y = mouse_pos.y / static_cast<float>(window_height);
//
//                std::cout << "(" << lower_x << ", " << upper_x << ", " << lower_y << ", " << upper_y << ")" << std::endl;
//            }
//
//            if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
//                sf::Vector2i mouse_pos = sf::Mouse::getPosition(window);
//
//                upper_x = mouse_pos.x / static_cast<float>(window_width);
//                upper_y = mouse_pos.y / static_cast<float>(window_width);
//
//                std::cout << "(" << lower_x << ", " << upper_x << ", " << lower_y << ", " << upper_y << ")" << std::endl;
//            }
//
//            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
//                for (int i = 0; i < 10; i++) {
//                    capture >> frame;
//
//                    fi++;
//
//                    if (fi >= video_length)
//                        goto done;
//                }
//            }
//        }
//
//        window.clear();
//        
//        capture >> frame;
//
//        sf::Image img;
//
//        img.create(video_width, video_height);
//
//        int video_channels = frame.channels();
//
//        for (int y = 0; y < video_height; y++)
//            for (int x = 0; x < video_width; x++) {
//                sf::Uint8 b = frame.data[0 + video_channels * (x + video_width * y)];
//                sf::Uint8 g = frame.data[1 + video_channels * (x + video_width * y)];
//                sf::Uint8 r = frame.data[2 + video_channels * (x + video_width * y)];
//
//                img.setPixel(x, y, sf::Color(r, g, b));
//            }
//
//        sf::Sprite s;
//
//        sf::Texture tex;
//        tex.loadFromImage(img);
//
//        s.setTexture(tex);
//
//        s.setScale(window_width / static_cast<float>(video_width), window_height / static_cast<float>(video_height));
//
//        window.draw(s);
//
//        sf::RectangleShape rs;
//
//        rs.setFillColor(sf::Color::Transparent);
//        rs.setOutlineColor(sf::Color::Green);
//        rs.setOutlineThickness(2.0f);
//
//        rs.setPosition(sf::Vector2f(lower_x * window_width, lower_y * window_height));
//        rs.setSize(sf::Vector2f((upper_x - lower_x) * window_width, (upper_y - lower_y) * window_height));
//
//        window.draw(rs);
//
//        window.display();
//
//        fi++;
//
//        if (fi >= video_length)
//            goto done;
//    } while (!quit);
//
//done:
//    std::cout << "Done." << std::endl;
//
//    return 0;
//}
//
