#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <valarray>
#include <random>
#include <omp.h>

#include <aogmaneo/helpers.h>

#include <assert.h>

typedef std::valarray<int> Vec;

Vec gen_rand(int vec_size, std::mt19937 &rng) {
    Vec v(vec_size);

    std::uniform_int_distribution<int> bdist(0, 1);

    for (int i = 0; i < v.size(); i++)
        v[i] = bdist(rng) * 2 - 1;

    return v;
}

Vec bind(const Vec &v1, const Vec &v2) {
    assert(v1.size() == v2.size());

    Vec v3(v1.size());

    for (int i = 0; i < v3.size(); i++)
        v3[i] = v1[i] * v2[i];

    return v3;
}

Vec embedding2d(float x, float y, const std::valarray<float> &L) {
    int vec_size = L.size() / 3;

    // mat mul
    Vec p(vec_size);

    for (int i = 0; i < vec_size; i++)
        p[i] = (std::cos(L[i * 3] * x + L[i * 3 + 1] * y + L[i * 3 + 2]) > 0.0f) * 2 - 1; // real part

    return p;
}

Vec embedding3d(float x, float y, float z, const std::valarray<float> &L) {
    int vec_size = L.size() / 4;

    // mat mul
    Vec p(vec_size);

    for (int i = 0; i < vec_size; i++)
        p[i] = (std::cos(L[i * 4] * x + L[i * 4 + 1] * y + L[i * 4 + 2] * z + L[i * 4 + 3]) > 0.0f) * 2 - 1; // real part

    return p;
}

Vec permute(const Vec &v, int amount = 1) {
    Vec vp(v.size());

    for (int i = 0; i < v.size(); i++)
        vp[i] = v[((i - amount) + v.size()) % v.size()];

    return vp;
}

Vec threshold(const Vec &v) {
    Vec vt(v.size());

    for (int i = 0; i < v.size(); i++)
        vt[i] = (v[i] > 0) * 2 - 1;

    return vt;
}

int similarity(const Vec &v1, const Vec &v2) {
    assert(v1.size() == v2.size());

    int sum = 0;

    for (int i = 0; i < v1.size(); i++)
        sum += v1[i] * v2[i];

    return sum;
}

void print(const Vec &v) {
    std::cout << "[ " << v[0] << "\n";

    for (int i = 1; i < v.size(); i++) {
        std::cout << "  " << v[i];

        if (i < v.size() - 1)
            std::cout << "\n";
    }

    std::cout << " ]" << std::endl;
}

int main() {
    std::mt19937 rng(time(nullptr));

    const int vec_size = 1024;

    omp_set_num_threads(8);

    std::vector<Vec> vecs(100);

    for (int i = 0; i < vecs.size(); i++)
        vecs[i] = gen_rand(vec_size, rng);

    Vec a = threshold(vecs[0] + vecs[1] + vecs[2]);

    Vec result = bind(a, bind(vecs[1], vecs[2]));

    // most similar
    int max_index = 0;
    int max_similarity = -999999;

    for (int i = 0; i < vecs.size(); i++) {
        int s = similarity(result, vecs[i]);

        if (s > max_similarity) {
            max_similarity = s;
            max_index = i;
        }
    }

    std::cout << "MAX: " << max_index << std::endl;

    return 0;

    unsigned int windowWidth = 1600;
    unsigned int windowHeight = 1200;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "VSA Test", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    std::valarray<float> C(vec_size * 4); // color
    std::valarray<float> L(vec_size * 3); // location

    std::normal_distribution<float> ndist(0.0f, 1.0f);

    int color_res = 6;

    for (int i = 0; i < C.size(); i++)
        C[i] = aon::rand_normalf() * color_res * 0.5f;

    for (int i = 0; i < L.size(); i++)
        L[i] = aon::rand_normalf() * 0.5f;

    float color_div = 1.0f / (color_res - 1);

    std::vector<Vec> color_lookup(color_res * color_res * color_res);

    for (int x = 0; x < color_res; x++)
        for (int y = 0; y < color_res; y++)
            for (int z = 0; z < color_res; z++) {
                float xf = x * color_div;
                float yf = y * color_div;
                float zf = z * color_div;

                color_lookup[z + color_res * (y + color_res * x)] = embedding3d(xf, yf, zf, C);
            }

    // bind to location
    //Vec test_vec = gen_rand(vec_size, rng);

    float target_x = 0.0f;
    float target_y = 0.0f;

    sf::Image test_img;
    test_img.loadFromFile("resources/test_color.png");

    sf::Image img;
    sf::Texture tex;

    bool quit = false;

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
        }

        int speed = 1.0f;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::A))
            target_x -= speed;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::D))
            target_x += speed;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::W))
            target_y -= speed;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::S))
            target_y += speed;

        window.clear(sf::Color::Black);

        // bind in a small radius
        Vec res(0, vec_size);

        //int width = 1;

        //for (float a = 0.0f; a < 2.0f * 3.1415f; a += 0.1f) {
        //    float adx = std::cos(a) * 10.0f;
        //    float ady = std::sin(a) * 10.0f;

        //    for (int dx = -width; dx <= width; dx++)
        //        for (int dy = -width; dy <= width; dy++) {
        //            float x = target_x + adx + dx;
        //            float y = target_y + ady + dy;

        //            res += bind(test_vec, embedding2d(x, y, L));
        //        }
        //}

        for (int x = 0; x < test_img.getSize().x; x++)
            for (int y = 0; y < test_img.getSize().y; y++) {
                sf::Color color = test_img.getPixel(x, y);
                
                float r = color.r / 255.0f;
                float g = color.g / 255.0f;
                float b = color.b / 255.0f;

                res += bind(embedding3d(r, g, b, C), embedding2d(target_x + x, target_y + y, L));
            }

        res = threshold(res); // final state vector, superposition of bound test_vec at different locations

        sf::Image img;
        img.create(64, 64);
        
        #pragma omp parallel for
        for (int x = 0; x < img.getSize().x; x++)
            for (int y = 0; y < img.getSize().y; y++) {
                Vec color_vec(vec_size);

                for (int i = 0; i < vec_size; i++)
                    color_vec[i] = res[i] * ((std::cos(L[i * 3] * x + L[i * 3 + 1] * y + L[i * 3 + 2]) > 0.0f) * 2 - 1);

                int ms = -999999;
                int mx = 0;
                int my = 0;
                int mz = 0;

                for (int cx = 0; cx < color_res; cx++)
                    for (int cy = 0; cy < color_res; cy++)
                        for (int cz = 0; cz < color_res; cz++) {
                            int s = similarity(color_vec, color_lookup[cz + color_res * (cy + color_res * cx)]);

                            if (s > ms) {
                                ms = s;
                                mx = cx;
                                my = cy;
                                mz = cz;
                            }
                        }

                float xf = mx * color_div;
                float yf = my * color_div;
                float zf = mz * color_div;

                sf::Color color;
                color.r = xf * 255.0f;
                color.g = yf * 255.0f;
                color.b = zf * 255.0f;

                img.setPixel(x, y, color);
            }

        tex.loadFromImage(img);

        sf::Sprite s;
        s.setTexture(tex);
        s.setScale(4.0f, 4.0f);

        window.draw(s);
        
        window.display();
    } while (!quit);

    return 0;
}
