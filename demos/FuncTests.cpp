#include <SFML/Graphics.hpp>

#include <aogmaneo/hierarchy.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <thread>
#include <mutex>
#include <cmath>

void draw_circle_part(sf::Image &img, int xc, int yc, int x, int y){
    img.setPixel(sf::Vector2u(xc + x, yc + y), sf::Color::White);
    img.setPixel(sf::Vector2u(xc - x, yc + y), sf::Color::White);
    img.setPixel(sf::Vector2u(xc + x, yc - y), sf::Color::White);
    img.setPixel(sf::Vector2u(xc - x, yc - y), sf::Color::White);
    img.setPixel(sf::Vector2u(xc + y, yc + x), sf::Color::White);
    img.setPixel(sf::Vector2u(xc - y, yc + x), sf::Color::White);
    img.setPixel(sf::Vector2u(xc + y, yc - x), sf::Color::White);
    img.setPixel(sf::Vector2u(xc - y, yc - x), sf::Color::White);
}

void draw_circle(sf::Image &img, int xc, int yc, int r){
    int x = 0, y = r;

    int d = 3 - 2 * r;

    draw_circle_part(img, xc, yc, x, y);

    while (y >= x){
        if (d > 0) {
            y--; 
            d = d + 4 * (x - y) + 10;
        }
        else
            d = d + 4 * x + 6;

        x++;
        
        draw_circle_part(img, xc, yc, x, y);
    }
}


int main() {
    int radius = 3;

    sf::Image img(sf::Vector2u(radius * 2 + 1, radius * 2 + 1), sf::Color::Black);

    draw_circle(img, radius, radius, radius);

    bool saved = img.saveToFile("bresenham_circle.png");

    return 0;
}
