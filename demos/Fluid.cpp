#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/image_encoder.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

using namespace aon;

class CustomStreamReader : public aon::Stream_Reader {
public:
    std::ifstream ins;

    void read(
        void* data,
        int len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        int len
    ) override {
        outs.write(static_cast<const char*>(data), len);
    }
};

int main() {
    unsigned int windowWidth = 512;
    unsigned int windowHeight = 512;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Fluid", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    aon::set_num_threads(8);

    int sim_width = 32;
    int sim_height = 32;

    float scale = std::min(static_cast<float>(windowWidth) / sim_width, static_cast<float>(windowHeight) / sim_height);

    Image_Encoder img_enc;
    Hierarchy h;

    {
        CustomStreamReader reader;

        reader.ins.open("resources/fluidsim.oenc", std::ios::binary);

        int magic;
        reader.read(&magic, sizeof(int));

        img_enc.read(reader);
    }

    {
        CustomStreamReader reader;

        reader.ins.open("resources/fluidsim.ohr", std::ios::binary);

        int magic;
        reader.read(&magic, sizeof(int));

        h.read(reader);
    }

    Array<Byte_Buffer_View> imgs(1);
    Byte_Buffer img(sim_width * sim_height * 3, 0);
    imgs[0] = img;

    Array<Int_Buffer_View> input_cis(1);

    bool quit = false;
    bool sim_mode = false;
    bool s_pressed_prev = false;

    sf::View view = window.getDefaultView();

    sf::RenderTexture rt;
    rt.create(sim_width, sim_height);
    rt.clear();

    std::cout << "Ready." << std::endl;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            if (window.hasFocus()) {
                switch (event.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                }
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
                rt.clear();

            if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);

                // draw terrain
                sf::CircleShape cs;
                cs.setRadius(2.0f);
                cs.setOrigin(cs.getRadius(), cs.getRadius());
                cs.setPosition(mousePos.x * sim_width / static_cast<float>(window.getSize().x), mousePos.y * sim_height / static_cast<float>(window.getSize().y));

                rt.draw(cs);
            }

            if (sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
                sf::Vector2i mousePos = sf::Mouse::getPosition(window);

                // draw terrain
                sf::CircleShape cs;
                cs.setFillColor(sf::Color::Blue);
                cs.setRadius(2.0f);
                cs.setOrigin(cs.getRadius(), cs.getRadius());
                cs.setPosition(mousePos.x * sim_width / static_cast<float>(window.getSize().x), mousePos.y * sim_height / static_cast<float>(window.getSize().y));

                rt.draw(cs);
            }

            bool s_pressed = sf::Keyboard::isKeyPressed(sf::Keyboard::S);

            if (s_pressed && !s_pressed_prev) {
                sim_mode = !sim_mode;

                if (sim_mode) {
                    h.clear_state();

                    rt.display();

                    // copy img
                    sf::Image rt_img = rt.getTexture().copyToImage();

                    for (int x = 0; x < sim_width; x++)
                        for (int y = 0; y < sim_height; y++) {
                            sf::Color c = rt_img.getPixel(x, y);

                            // transpose needed
                            img[0 + 3 * (x + sim_width * y)] = c.r;
                            img[1 + 3 * (x + sim_width * y)] = c.g;
                            img[2 + 3 * (x + sim_width * y)] = c.b;
                        }

                    img_enc.step(imgs, false);

                    input_cis[0] = img_enc.get_hidden_cis();

                    h.step(input_cis, false);
                }
            }

            s_pressed_prev = s_pressed;
        }

        sf::Sprite s;
        s.setOrigin(sim_width * 0.5f, sim_height * 0.5f);
        s.setPosition(windowWidth * 0.5f, windowHeight * 0.5f);
        s.setScale(scale, scale);

        sf::Texture tex;

        if (sim_mode) {
            img_enc.reconstruct(h.get_prediction_cis(0));

            img = img_enc.get_reconstruction(0);

            input_cis[0] = h.get_prediction_cis(0);

            h.step(input_cis, false);

            sf::Image result;
            result.create(sim_width, sim_height);

            for (int x = 0; x < sim_width; x++)
                for (int y = 0; y < sim_height; y++) {
                    sf::Color c;

                    // transpose needed
                    c.r = img[0 + 3 * (x + sim_width * y)];
                    c.g = img[1 + 3 * (x + sim_width * y)];
                    c.b = img[2 + 3 * (x + sim_width * y)];
                    c.a = 255;

                    result.setPixel(x, y, c);
                }

            tex.loadFromImage(result);
            s.setTexture(tex);
        }
        else {
            rt.display();

            s.setTexture(rt.getTexture());
        }

        window.draw(s);
        
        window.display();
    } while (!quit);

    return 0;
}

