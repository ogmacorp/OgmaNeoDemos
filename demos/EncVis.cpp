#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "constructs/Matrix3x3f.h"

#include <aogmaneo/encoder.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

using namespace aon;

int main() {
    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Enc Vis", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    aon::set_num_threads(4);

    int resolution = 128;

    // Encoder
    Array<Encoder::Visible_Layer_Desc> vlds(1);
    vlds[0].size = Int3(1, 2, resolution);

    Encoder e;
    e.init_random(Int3(1, 3, 32), vlds);
    Encoder::Params params;
    
    Int_Buffer inputs(2);

    Array<Int_Buffer_View> input_cis(1);
    input_cis[0] = inputs;

    bool quit = false;

    sf::View view = window.getDefaultView();

    const float dt = 0.017f;
    const float zoomRate = 0.4f;
    const float viewInterpolateRate = 20.0f;

    view.setCenter(64.0f, 64.0f);
    view.zoom(0.1f);

    sf::View newView = view;

    window.setView(view);

    sf::Image img;
    img.loadFromFile("resources/density_image5.png");
    sf::Texture tex;
    tex.loadFromImage(img);

    std::vector<float> densities(img.getSize().x * img.getSize().y);
    float total_density = 0.0f;

    for (int x = 0; x < img.getSize().x; x++)
        for (int y = 0; y < img.getSize().y; y++) {
            sf::Color c = img.getPixel(x, y);

            float gray = (c.r / 255.0f + c.g / 255.0f + c.b / 255.0f) * 0.333f;

            densities[y + x * img.getSize().y] = gray;
            total_density += gray;
        }

    // generate color pallette
    std::vector<sf::Color> pallette(e.get_hidden_cis().size());

    for (int i = 0; i < pallette.size(); i++)
        pallette[i] = sf::Color(dist01(rng) * 255.0f, dist01(rng) * 255.0f, dist01(rng) * 255.0f, 255);

    sf::Vector2f testPos(0.0f, 0.0f);

    std::cout << "Ready." << std::endl;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            if (window.hasFocus()) {
                switch (event.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                case sf::Event::MouseWheelMoved:
                    int dWheel = event.mouseWheel.delta;

                    newView = view;

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta0 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    newView.setSize(view.getSize() + newView.getSize() * (-zoomRate * dWheel));

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta1 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    window.setView(view);

                    newView.setCenter(view.getCenter() + mouseZoomDelta0 - mouseZoomDelta1);

                    break;
                }
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                sf::Vector2f mousePos = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                testPos = mousePos;
            }
        }

        int iters = 1;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
            iters = 100;

        for (int it = 0; it < iters; it++) {
            // sample 
            int sample_index = 0;
            float cusp = dist01(rng) * total_density;
            float sum_so_far = 0.0f;

            for (int i = 0; i < densities.size(); i++) {
                sum_so_far += densities[i];

                if (sum_so_far >= cusp) {
                    sample_index = i;
                    break;
                }
            }

            int sample_x = sample_index / img.getSize().y;
            int sample_y = sample_index % img.getSize().y;

            int x_ci = (static_cast<float>(sample_x) / img.getSize().x) * resolution;
            int y_ci = (static_cast<float>(sample_y) / img.getSize().y) * resolution;

            inputs[0] = x_ci;
            inputs[1] = y_ci;

            e.step(input_cis, true, params);
        }

        view.setCenter(view.getCenter() + (newView.getCenter() - view.getCenter()) * viewInterpolateRate * dt);
        view.setSize(view.getSize() + (newView.getSize() - view.getSize()) * viewInterpolateRate * dt);

        window.setView(view);

        window.clear(sf::Color::Black);

        // Render image

        sf::Sprite s;
        s.setTexture(tex);
        window.draw(s);

        // Draw SOMs as lines
        Int3 hidden_size = e.get_hidden_size();
        int num_columns = e.get_hidden_cis().size();
        int num_cells_per_column = e.get_hidden_size().z;

        for (int c = 0; c < num_columns; c++) {
            Int2 columnPos(c / hidden_size.y, c % hidden_size.y);

            sf::VertexArray va(sf::LineStrip, num_cells_per_column);
            sf::Color color = pallette[c];

            sf::CircleShape cs;
            cs.setFillColor(color);
            cs.setRadius(0.3f);
            cs.setOrigin(cs.getRadius(), cs.getRadius());

            for (int cc = 0; cc < num_cells_per_column; cc++) {
                int hidden_cell_index = cc + c * hidden_size.z;

                Encoder::Visible_Layer &vl = e.get_visible_layer(0);
                const Encoder::Visible_Layer_Desc &vld = e.get_visible_layer_desc(0);

                int diam = vld.radius * 2 + 1;
     
                // Projection
                Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
                    static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

                Int2 visibleCenter = project(columnPos, hToV);

                // Lower corner
                Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

                // Bounds of receptive field, clamped to input size
                Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
                Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

                sf::Vector2f coord;
                int coordIndex = 0;

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int wi = cc + hidden_size.z * (offset.y + diam * (offset.x + diam * c));
                        //int wi = offset.y + diam * (offset.x + diam * hidden_cell_index);

                        float p = vl.means[wi];

                        if (coordIndex == 0)
                            coord.x = p * vld.size.z;
                        else if (coordIndex == 1)
                            coord.y = p * vld.size.z;

                        coordIndex++;
                    }

                va[cc].position = coord;
                va[cc].color = color;

                cs.setPosition(coord);
                window.draw(cs);
            }

            window.draw(va);
        }

        // draw test position
        sf::CircleShape cs;
        cs.setPosition(testPos);
        cs.setFillColor(sf::Color::Green);
        cs.setRadius(0.5f);
        cs.setOrigin(0.5f, 0.5f);

        window.draw(cs);

        int x_ci = std::min(0.9999f, std::max(0.0f, testPos.x / img.getSize().x)) * resolution;
        int y_ci = std::min(0.9999f, std::max(0.0f, testPos.y / img.getSize().y)) * resolution;

        inputs[0] = x_ci;
        inputs[1] = y_ci;

        e.step(input_cis, false, params);

        // draw columns
        const float spacing = 1.5f;
        const float rect_size = 1.3f;

        for (int c = 0; c < num_columns; c++) {
            Int2 columnPos(c / hidden_size.y, c % hidden_size.y);

            int hidden_ci = e.get_hidden_cis()[c];

            for (int cc = 0; cc < num_cells_per_column; cc++) {
                int hidden_cell_index = address3(Int3(columnPos.x, columnPos.y, cc), hidden_size);

                sf::Color color;

                if (cc == hidden_ci)
                    color = sf::Color::White;
                else
                    color = sf::Color(32, 32, 32);

                sf::RectangleShape rs;
                rs.setSize(sf::Vector2f(rect_size, rect_size));
                rs.setFillColor(color);
                rs.setPosition(c * spacing, cc * spacing);

                window.draw(rs);
            }
        }
        
        window.display();
    } while (!quit);

    return 0;
}
