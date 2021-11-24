#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "generative/Layer.h"
#define USE_MNIST_LOADER
#include "mnist.h"

#include <omp.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

int main() {
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::normal_distribution<float> nDist(0.0f, 1.0f);

    omp_set_num_threads(8);

    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    int inputWidth = 64;
    int inputHeight = 64;

    std::vector<Layer> layers(4);

    int prevWidth = inputWidth;
    int prevHeight = inputHeight;
    int prevDepth = 6;
    int outputDepth = 2;
    int columnSize = 16;

    for (int l = 0; l < layers.size(); l++) {
        outputDepth = 2 + l;

        layers[l].initRandom(prevWidth, prevHeight, prevDepth, outputDepth, columnSize, l == 0 ? 1 : 3, 3);

        prevWidth = layers[l].getOutputWidth();
        prevHeight = layers[l].getOutputHeight();
        prevDepth = outputDepth * columnSize;
    }

    int numOutputColumns = prevWidth * prevHeight * outputDepth;

    std::cout << "Output size: (" << prevWidth << ", " << prevHeight << ", " << prevDepth << ")" << std::endl;

    //mnist_data* data;
    //unsigned int cnt;
    //int ret = mnist_load("resources/MNIST/train-images-idx3-ubyte", "resources/MNIST/train-labels-idx1-ubyte", &data, &cnt);

    //if (ret) {
    //    printf("An error occured: %d\n", ret);
    //    return -1;
    //} else {
    //    printf("image count: %d\n", cnt);
    //}

    // Train
    std::uniform_int_distribution<int> dataDist(0, 999);

    std::vector<float> inputs(inputWidth * inputHeight * 6, 0.0f);

    sf::RenderTexture resizeRT;
    resizeRT.create(inputWidth, inputHeight);

    const float lr = 0.1f;
    for (int it = 0; it < 1000; it++) {
        int index = it;//dataDist(rng);

        int numDigits = 0;

        int value = index;

        while (value != 0) {
            value /= 10;
            numDigits++;
        }

        if (numDigits == 0)
            numDigits = 1;

        int numZeros = 5 - numDigits;

        std::string zeros;

        for (int i = 0; i < numZeros; i++)
            zeros += '0';

        sf::Texture tex;
        tex.loadFromFile("resources/faces0/" + zeros + std::to_string(index) + ".png");

        resizeRT.clear();

        sf::Sprite s;
        s.setTexture(tex);
        s.setScale(static_cast<float>(inputWidth) / tex.getSize().x, static_cast<float>(inputHeight) / tex.getSize().y);

        resizeRT.draw(s);

        resizeRT.display();

        sf::Image img = resizeRT.getTexture().copyToImage();

        for (int x = 0; x < inputWidth; x++)
            for (int y = 0; y < inputHeight; y++) {
                sf::Color c = img.getPixel(x, y);

                inputs[0 + 6 * (y + x * inputHeight)] = c.r / 255.0f;
                inputs[1 + 6 * (y + x * inputHeight)] = c.g / 255.0f;
                inputs[2 + 6 * (y + x * inputHeight)] = c.b / 255.0f;

                inputs[3 + 6 * (y + x * inputHeight)] = 1.0f - c.r / 255.0f;
                inputs[4 + 6 * (y + x * inputHeight)] = 1.0f - c.g / 255.0f;
                inputs[5 + 6 * (y + x * inputHeight)] = 1.0f - c.b / 255.0f;
            }

        for (int l = 0; l < layers.size(); l++) {
            if (l == 0)
                layers[l].step(inputs, lr);
            else
                layers[l].step(layers[l - 1].outputs, lr);
        }

        if (it % 10 == 0) {
            std::cout << "Iteration " << it << std::endl;
        }
    }

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Gen Demo", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    std::vector<float> reconOutputs = layers.back().outputs;

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
            iters = 100;

        window.clear();

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::R)) {
            std::uniform_int_distribution<int> columnDist(0, columnSize - 1);

            for (int i = 0; i < numOutputColumns; i++) {
                int r = columnDist(rng);

                for (int j = 0; j < columnSize; j++) {
                    reconOutputs[j + i * columnSize] = (j == r ? 1.0f : 0.0f);
                }
            }
        }

        // Reconstruct a digit
        std::vector<float> reconTemp = reconOutputs;

        for (int l = layers.size() - 1; l >= 0; l--) {
            layers[l].reconstruct(reconTemp);            

            reconTemp = layers[l].reconstruction;
        }
        
        sf::Image img;
        img.create(inputWidth, inputHeight);

        for (int x = 0; x < inputWidth; x++)
            for (int y = 0; y < inputHeight; y++) {
                sf::Uint8 r = std::min(1.0f, std::max(0.0f, reconTemp[0 + 6 * (y + x * inputHeight)])) * 255.0f;
                sf::Uint8 g = std::min(1.0f, std::max(0.0f, reconTemp[1 + 6 * (y + x * inputHeight)])) * 255.0f;
                sf::Uint8 b = std::min(1.0f, std::max(0.0f, reconTemp[2 + 6 * (y + x * inputHeight)])) * 255.0f;

                img.setPixel(x, y, sf::Color(r, g, b, 255));
            }

        sf::Texture tex;
        tex.loadFromImage(img);

        sf::Sprite s;
        s.setTexture(tex);
        s.setScale(4.0f, 4.0f);

        // Render editor
        sf::Vector2f startPos(10.0f, inputHeight * 4.0f + 10.0f);

        sf::CircleShape cs;
        cs.setRadius(4.0f);
        cs.setOrigin(4.0f, 4.0f);
        cs.setFillColor(sf::Color::White);

        sf::Vector2i mousePos = sf::Mouse::getPosition(window);

        for (int i = 0; i < numOutputColumns; i++) {
            for (int j = 0; j < columnSize; j++) {
                cs.setPosition(sf::Vector2f(i * 10.0f, j * 10.0f) + startPos);        

                if (sf::Mouse::isButtonPressed(sf::Mouse::Left)) {
                    sf::Vector2f delta(cs.getPosition() - sf::Vector2f(mousePos.x, mousePos.y));

                    float dist = std::sqrt(delta.x * delta.x + delta.y * delta.y);

                    if (dist < 5.0f) {
                        // Reset column
                        for (int oj = 0; oj < columnSize; oj++)
                            reconOutputs[oj + i * columnSize] = 0.0f;

                        reconOutputs[j + i * columnSize] = 1.0f;
                    }
                }

                if (reconOutputs[j + i * columnSize] > 0.0f) {
                    cs.setFillColor(sf::Color::Red);

                    window.draw(cs);

                    cs.setFillColor(sf::Color::White);
                }
                else
                    window.draw(cs);
            }
        }

        window.draw(s);

        window.display();
    } while (!quit);

    return 0;
}

