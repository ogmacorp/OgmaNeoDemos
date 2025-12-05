#ifndef _GUI_CONTROL_SFML_HPP
#define _GUI_CONTROL_SFML_HPP

#include <SFML/Graphics.hpp>

struct guiControl
{
    sf::View view;
    sf::Clock clock;
    sf::RectangleShape virtRect;
    sf::Vector2f position;
    sf::Vector2i winDimensions;

    float zoomScaling; // mouse wheel
    float moveScaling; // horizontal motion of the key
    float zoomX;
    float zoomY;

    guiControl(sf::Vector2i winDimensions_, float zoom = 0.01f, float move = 10.f)
    {
        winDimensions = winDimensions_;
        zoomScaling   = zoom;
        moveScaling   = move;

        // View background virtual Object
        virtRect = sf::RectangleShape(sf::Vector2f(winDimensions.x, winDimensions.x));
        virtRect.setFillColor(sf::Color::Transparent);

        // Reseting the View
        view.setCenter(sf::Vector2f(0.0f, 0.0f));
        view.setSize(sf::Vector2f(winDimensions.x, winDimensions.y));
        zoomX = 1.0f;
        zoomY = 1.0f;
        //view.setViewport(sf::FloatRect(sf::Vector2f(0, 0), sf::Vector2f(zoomX, zoomY)));
        position = sf::Vector2f(winDimensions.x/2.0f, winDimensions.y/2.0f);
    };

    void update(sf::RenderWindow &render)
    {
        if (virtRect.getPosition().x + 10 > winDimensions.x / 2.0f)
            position.x = virtRect.getPosition().x + 10;
        else
            position.x = winDimensions.x / 2.0f;
        view.setCenter(position);
        render.setView(view);
        render.draw(virtRect);
    };

    void eventHandling (sf::RenderWindow &render, bool &quit)
    {
        while (const std::optional event = render.pollEvent())
        {
            float zoomFactor = 0.0;
            if (event->is<sf::Event::Closed>())
                quit = true;

            if (const auto* mouseWheelScrolled = event->getIf<sf::Event::MouseWheelScrolled>()) {
                // positive if forward, negative if backward
                if (mouseWheelScrolled->delta > 0)
                    zoomFactor = +zoomScaling * clock.getElapsedTime().asSeconds(); // bigger view, object appears smaller
                else
                    zoomFactor = -zoomScaling * clock.getElapsedTime().asSeconds(); // smaller view, object appears bigger
                zoomX += zoomFactor;
                if (zoomX < 0.01) zoomX = 0.01;
                zoomY += zoomFactor;
                if (zoomY < 0.01) zoomY = 0.01;
                //view.setViewport(sf::FloatRect(sf::Vector2f(0, 0), sf::Vector2f(zoomX, zoomY)));
            }

            if (const auto* mouseButtonPressed = event->getIf<sf::Event::MouseButtonPressed>()) {
                if (mouseButtonPressed->button == sf::Mouse::Button::Right)
                    virtRect.move(sf::Vector2f( moveScaling* clock.getElapsedTime().asSeconds(), 0));
                else if (mouseButtonPressed->button == sf::Mouse::Button::Left)
                    virtRect.move(sf::Vector2f(-moveScaling* clock.getElapsedTime().asSeconds(), 0));
                break;
            }
        }
        //update(render);
    };
};

#endif
