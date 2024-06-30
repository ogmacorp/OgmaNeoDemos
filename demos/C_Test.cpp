#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "maze/maze_generator.h"

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/image_encoder.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

using namespace aon;

inline float sigmoid(
    float x
) {
    if (x < 0.0f) {
        float z = std::exp(x);

        return z / (1.0f + z);
    }
    
    return 1.0f / (1.0f + std::exp(-x));
}

int main() {
    unsigned int windowWidth = 640;
    unsigned int windowHeight = 640;

    int maze_width = 16;
    int maze_height = 16;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "C Test", sf::Style::Default);

    window.setVerticalSyncEnabled(true);
    //window.setFramerateLimit(60);

    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    Maze_Generator mg;

    mg.generate_maze(maze_width, maze_height, rng);

    sf::Image img;
    img.create(maze_width, maze_height);

    sf::Texture tex;

    set_num_threads(8);

    // Create hierarchy
    Array<Hierarchy::Layer_Desc> lds(2);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(4, 4, 32);
    }

    const int max_free_space = 4;

    // Two IODescs, for sensors and for actions
    // types none and prediction (no prediction and predictions used as actions)
    Array<Hierarchy::IO_Desc> ioDescs(3);
    ioDescs[0] = Hierarchy::IO_Desc(Int3(1, 1, max_free_space), IO_Type::prediction, 0, 2); // input
    ioDescs[1] = Hierarchy::IO_Desc(Int3(1, 1, 3), IO_Type::action, 0, 2); // action
    ioDescs[2] = Hierarchy::IO_Desc(Int3(2, 2, 16), IO_Type::action, 2, 2); // C

    Hierarchy h;
    h.init_random(ioDescs, lds);

    Int_Buffer input(1, 0);
    Int_Buffer action(1, 0);
    Int_Buffer C(4, 0);

    Array<const Int_Buffer*> inputs(3);
    inputs[0] = &input;
    inputs[1] = &action;
    inputs[2] = &C;

    std::uniform_int_distribution<int> n_dist(0, mg.num_free_cells - 1);
    int free_cell_idx = n_dist(rng);

    int start_cell = mg.free_cells[free_cell_idx];

    int free_cell_idx2 = n_dist(rng) - 1;

    int end_cell = mg.free_cells[(free_cell_idx + 1 + free_cell_idx2) % mg.num_free_cells];

    Int2 pos(start_cell / maze_height + maze_offset, start_cell % maze_height + maze_offset);
    int dir = 0;

    Int2 end_pos(end_cell / maze_height + maze_offset, end_cell % maze_height + maze_offset);

    const std::array<Int2, 4> dir_to_delta = { Int2(1, 0), Int2(0, 1), Int2(-1, 0), Int2(0, -1) };

    bool quit = false;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;
        }

        int iters = 1;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
            iters = 1000;

        for (int it = 0; it < iters; it++) {
            // Observe
            int dist = max_free_space - 1;

            Int2 scan_pos = pos;

            for (int i = 0; i < max_free_space; i++) {
                scan_pos = Int2(scan_pos.x + dir_to_delta[dir].x, scan_pos.y + dir_to_delta[dir].y);

                if (mg.get(scan_pos.x, scan_pos.y) == WALL_CELL) {
                    dist = i;
                    break;
                }
            }

            input[0] = dist;

            action = h.get_prediction_cis(1);

            C = h.get_prediction_cis(2);

            bool found = (pos.x == end_pos.x && pos.y == end_pos.y);
            float reward = found * 10.0f;

            h.step(inputs, true, reward, 0.0f);

            int action = h.get_prediction_cis(1)[0];

            Int2 new_pos = pos;

            switch (action) {
            case 0:
                new_pos.x += dir_to_delta[dir].x;
                new_pos.y += dir_to_delta[dir].y;
                break;
            case 1:
                dir = (dir + 1) % 3;
                break;
            case 2:
                dir = (dir + 3) % 3;
                break;
            }

            if (mg.get(new_pos.x, new_pos.y) == EMPTY_CELL) {
                pos = new_pos;
            }
        }

        window.clear();

        for (int x = 0; x < maze_width; x++)
            for (int y = 0; y < maze_height; y++)
                img.setPixel(x, y, mg.get(x, y) ? sf::Color::White : sf::Color::Black);

        tex.loadFromImage(img);

        sf::Sprite s;

        float scale = std::min(windowWidth / (float)maze_width, windowHeight / (float)maze_height);

        s.setTexture(tex);
        s.setScale(scale, scale);

        window.draw(s);

        window.display();
    } while (!quit);

    return 0;
}
