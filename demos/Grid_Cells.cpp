// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <Box2D/Box2D.h>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "constructs/Matrix3x3f.h"

#include <time.h>
#include <iostream>
#include <random>

#include <aogmaneo/Hierarchy.h>

using namespace aon;

class GridCells {
public:
    struct Layer {
        
    };
};

int main() {
    std::mt19937 rng(time(nullptr));

    setNumThreads(8);

    sf::RenderWindow window;
    window.setFramerateLimit(60);
    window.setVerticalSyncEnabled(true);

    window.create(sf::VideoMode(800, 600), "Grid Cells", sf::Style::Default);

    sf::Font font;
    font.loadFromFile("resources/Hack-Regular.ttf");

    // ---------------------------- Game Loop -----------------------------

    bool quit = false;

    bool trainMode = true;
    bool showMode = false;

    bool tPressedPrev = false;
    bool sPressedPrev = false;

    do {
        // ----------------------------- Input -----------------------------

        // Receive events
        sf::Event windowEvent;

        while (window.pollEvent(windowEvent))
        {
            switch (windowEvent.type)
            {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            bool tPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::T);

            if (tPressed && !tPressedPrev)
                trainMode = !trainMode;
        
            tPressedPrev = tPressed; 

            bool sPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::S);

            if (sPressed && !sPressedPrev)
                showMode = !showMode;
        
            sPressedPrev = sPressed; 
        }

        if (!showMode) {
        }

        // Show on main window
        window.clear();

        window.display();
    } while (!quit);

    return 0;
}
