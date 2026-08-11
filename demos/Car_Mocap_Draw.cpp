// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <time.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <cmath>

const float pi = 3.141592f;

float min_angle_delta(float delta) {
    return std::fmod(delta + pi, 2.0f * pi) - pi;
}

float length(const sf::Vector2f &v) {
    return std::sqrt(v.x * v.x + v.y * v.y);
}

float length(const sf::Vector3f &v) {
    return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

sf::Vector2f normalize(const sf::Vector2f &v) {
    return v / std::max(0.00001f, length(v));
}

int main() {
    // Initialize a random number generator
    std::mt19937 rng(time(nullptr));

    sf::Vector2u window_size(1280, 720);

    sf::RenderWindow window(sf::VideoMode({window_size.x, window_size.y}), "Car Mocap Draw", sf::Style::Default);

    window.setFramerateLimit(120);

    // load csv
    std::ifstream from_file("out_aionly.csv");

    std::vector<sf::Vector3f> points;

    int line_count = 0;

    while (from_file.good() && !from_file.eof()) {
        std::string line;
        std::getline(from_file, line);

        if (line_count > 0) { // skip header
            std::string word;
            int last_index = 0;
            int component = 0;

            sf::Vector3f point;
            
            for (int i = 0; i < line.length(); i++) {
                if (line[i] == ',' || i == line.length() - 1) {
                    word = line.substr(last_index, i);
                    last_index = i + 1;

                    float value = std::stof(word);

                    switch(component) {
                    case 0: // frame count
                        break;
                    case 1: // marker index
                        break;
                    case 2:
                        point.x = value;
                        break;

                    case 3:
                        point.y = value;
                        break;

                    case 4:
                        point.z = value;
                        break;
                    }

                    component++;
                }
            }

            points.push_back(point);
        }

        line_count++;
    }

    // remove outliers/noise
    {
        float average_speed = 0.0f;

        for (int f = 1; f < points.size(); f++) {
            const sf::Vector3f &p = points[f];
            const sf::Vector3f &pp = points[f - 1];

            float speed = length(p - pp);

            average_speed += speed;
        }

        average_speed /= points.size();

        std::cout << "Average speed: " << average_speed * (120.0f / 1000.0f) << " m/s" << std::endl;

        const float too_fast_multiplier = 3.0f;

        for (int f = 1; f < points.size();) {
            const sf::Vector3f &p = points[f];
            const sf::Vector3f &pp = points[f - 1];

            float speed = length(p - pp);

            // if too fast, drop point
            if (speed > too_fast_multiplier * average_speed)
                points.erase(points.begin() + f);
            else
                f++;
        }
    }

    // find bounds
    const float max_range = 999999.0f;

    sf::Vector3f lower(max_range, max_range, max_range);
    sf::Vector3f upper = -lower;

    for (int f = 0; f < points.size(); f++) {
        const sf::Vector3f &p = points[f];

        if (p.x < lower.x)
            lower.x = p.x;
        if (p.y < lower.y)
            lower.y = p.y;
        if (p.z < lower.z)
            lower.z = p.z;

        if (p.x > upper.x)
            upper.x = p.x;
        if (p.y > upper.y)
            upper.y = p.y;
        if (p.z > upper.z)
            upper.z = p.z;
    }

    sf::Vector3f center = (upper - lower) * 0.5f + lower;

    const float buffer = 100.0f;
    float max_dim2 = std::max(upper.x - lower.x, upper.y - lower.y) + buffer;
    float rescale = std::min(window.getSize().x, window.getSize().y) / max_dim2;

    bool quit = false;

    sf::View v = window.getDefaultView();
    v.setCenter(sf::Vector2f(0.0f, 0.0f));
    window.setView(v);

    for (int f = 0; f < points.size() && !quit; f++) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                quit = true;
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Q))
                quit = true;
        }

        window.clear();
        
        sf::Vector3f pos_centered = points[f] - center;

        sf::Vector2f pos = sf::Vector2f(pos_centered.x * rescale, -pos_centered.y * rescale);

        sf::CircleShape cs;
        cs.setRadius(8.0f);
        cs.setOrigin(sf::Vector2f(8.0f, 8.0f));
        cs.setPosition(pos);

        window.draw(cs);

        window.display();
    }

    sf::Texture lineGradientTexture("resources/lineGradient.png");

    sf::RenderTexture rt(sf::Vector2u(3000, 1600));

    rt.clear(sf::Color::Transparent);

    v = rt.getDefaultView();
    v.setCenter(sf::Vector2f(0.0f, 0.0f));
    rt.setView(v);

    rescale = std::min(rt.getSize().x, rt.getSize().y) / max_dim2;

    sf::VertexArray va(sf::PrimitiveType::Triangles);

    va.resize((points.size() - 1) * 6);

    sf::RenderStates rs;
    rs.texture = &lineGradientTexture;

    int index = 0;

    float average_speed_delayed = 0.0f;
    float average_speed = 0.0f;

    const float filter_rate = 0.5f;

    sf::Vector3f filtered_pos = points[0];

    sf::Vector2f pos_prev;
    sf::Color color_prev;

    {
        sf::Vector3f pos_centered = points[0] - center;

        pos_prev = sf::Vector2f(pos_centered.x * rescale, -pos_centered.y * rescale);

        color_prev = sf::Color(127, 127, 0, 127);
    }

    const float line_thickness = 4.0f;

    for (int f = 1; f < points.size(); f++) {
        float speed = length(points[f] - points[f - 1]);

        filtered_pos += filter_rate * (points[f] - filtered_pos);

        average_speed += 0.1f * (speed - average_speed);
        average_speed_delayed += 0.1f * (average_speed - average_speed_delayed);

        float accel = average_speed - average_speed_delayed;

        float squash = std::tanh(accel * 0.25f) * 0.5f + 0.5f;

        sf::Vector3f pos_centered = filtered_pos - center;

        sf::Vector2f pos = sf::Vector2f(pos_centered.x * rescale, -pos_centered.y * rescale);

        sf::Color color((1.0f - squash) * 255.0f, squash * 255.0f, 0, 127);

        // set vertices
        {
            sf::Vector2f dir = pos - pos_prev;

            sf::Vector2f perpendicular = normalize(sf::Vector2f(-dir.y, dir.x));

            va[index].position = pos_prev - perpendicular * line_thickness;
            va[index].texCoords = sf::Vector2f(0.0f, 0.0f);
            va[index].color = color_prev;

            index++;

            va[index].position = pos - perpendicular * line_thickness;
            va[index].texCoords = sf::Vector2f(0.0f, 0.0f);
            va[index].color = color;

            index++;

            va[index].position = pos + perpendicular * line_thickness;
            va[index].texCoords = sf::Vector2f(0.0f, lineGradientTexture.getSize().y);
            va[index].color = color;

            index++;

            va[index].position = pos_prev - perpendicular * line_thickness;
            va[index].texCoords = sf::Vector2f(0.0f, 0.0f);
            va[index].color = color_prev;

            index++;

            va[index].position = pos + perpendicular * line_thickness;
            va[index].texCoords = sf::Vector2f(0.0f, lineGradientTexture.getSize().y);
            va[index].color = color;

            index++;

            va[index].position = pos_prev + perpendicular * line_thickness;
            va[index].texCoords = sf::Vector2f(0.0f, lineGradientTexture.getSize().y);
            va[index].color = color_prev;

            index++;
        }

        pos_prev = pos;
        color_prev = color;
    }

    rt.draw(va, rs);

    rt.display();

    bool result = rt.getTexture().copyToImage().saveToFile("car_tracking_result.png");

    return 0;
}


