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
	  view.reset(sf::FloatRect(0, 0, winDimensions.x, winDimensions.y));
	  zoomX = 1.0f;
	  zoomY = 1.0f;
	  view.setViewport(sf::FloatRect(0, 0, zoomX, zoomY));
	  position = sf::Vector2f(winDimensions.x/2, winDimensions.y/2);
	};

	void update(sf::RenderWindow &render)
	{
		if (virtRect.getPosition().x + 10 > winDimensions.x / 2)
      position.x = virtRect.getPosition().x + 10;
    else
      position.x = winDimensions.x / 2;
    view.setCenter(position);
    render.setView(view);
    render.draw(virtRect);
  };

  void eventHandling (sf::RenderWindow &render, bool &quit)
  {
		sf::Event event;

    while (render.pollEvent(event))
    {
      float zoomFactor = 0.0;
      switch (event.type)
      {
      case sf::Event::Closed:
        quit = true;
        break;

      case sf::Event::MouseWheelMoved:
        // positive if forward, negative if backward
        if (event.mouseWheel.delta > 0)
          zoomFactor = +zoomScaling * clock.getElapsedTime().asSeconds(); // bigger view, object appears smaller
        else
          zoomFactor = -zoomScaling * clock.getElapsedTime().asSeconds(); // smaller view, object appears bigger
        zoomX += zoomFactor;
        if (zoomX < 0.01) zoomX = 0.01;
        zoomY += zoomFactor;
        if (zoomY < 0.01) zoomY = 0.01;
        view.setViewport(sf::FloatRect(0, 0, zoomX, zoomY));
        break;

      case sf::Event::MouseButtonPressed:
        if (event.mouseButton.button == sf::Mouse::Right)
          virtRect.move( moveScaling* clock.getElapsedTime().asSeconds(), 0);
        else if (event.mouseButton.button == sf::Mouse::Left)
          virtRect.move(-moveScaling* clock.getElapsedTime().asSeconds(), 0);
        break;
      }
    }
    update(render);
  };
};

#endif