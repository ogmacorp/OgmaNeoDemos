#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <valarray>
#include <random>

#include <assert.h>

typedef std::valarray<float> Vec;

const float epsilon = 0.0001f;

Vec gen_rand(int vec_size, std::mt19937 &rng) {
    Vec v(vec_size);

    std::normal_distribution<float> ndist(0.0f, 1.0f);

    float total2 = 0.0f;

    for (int i = 0; i < v.size(); i++) {
        v[i] = ndist(rng);

        total2 += v[i] * v[i];
    }

    float scale = 1.0f / std::max(epsilon, std::sqrt(total2));

    for (int i = 0; i < v.size(); i++)
        v[i] *= scale;

    return v;
}

Vec bind(const Vec &v1, const Vec &v2) {
    assert(v1.size() == v2.size());

    Vec v3(v1.size());

    for (int i = 0; i < v1.size(); i++) {
        float sum = 0.0f;

        for (int j = 0; j < v2.size(); j++)
            sum += v1[i] * v2[(i - j + v1.size()) % v1.size()];

        v3[i] = sum;
    }

    return v3;
}

Vec normalized(const Vec &v) {
    Vec n(v.size());

    float total2 = 0.0f;

    for (int i = 0; i < v.size(); i++)
        total2 += v[i] * v[i];

    float scale = 1.0f / std::max(epsilon, std::sqrt(total2));

    for (int i = 0; i < v.size(); i++)
        n[i] = v[i] * scale;

    return n;
}

Vec invert(const Vec &v) {
    Vec inv(v.size());

    inv[0] = v[0];

    for (int i = 1; i < v.size(); i++)
        inv[i] = v[v.size() - i];

    return inv;
}

float similarity(const Vec &v1, const Vec &v2) {
    assert(v1.size() == v2.size());

    float sum = 0.0f;

    for (int i = 0; i < v1.size(); i++)
        sum += v1[i] * v2[i];

    return sum;
}

float distance(const Vec &v1, const Vec &v2) {
    assert(v1.size() == v2.size());

    float sum = 0.0f;

    for (int i = 0; i < v1.size(); i++) {
        float delta = v1[i] - v2[i];

        sum += delta * delta;
    }

    return std::sqrt(sum);
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

class Predictor {
public:
    int vec_size;
    Vec trace;
    Vec pos;

    void init(int vec_size, std::mt19937 &rng) {
        this->vec_size = vec_size;    

        trace = gen_rand(vec_size, rng);
        pos = gen_rand(vec_size, rng);
    }

    // returns prediction
    Vec step(const Vec &item) {
        // update predictor head
        trace = normalized(bind(pos, item) + trace);
        pos = normalized(bind(pos, pos));
        
        return bind(invert(pos), trace);
    }
};

class Sequencer {
public:
    int vec_size;
    Vec trace;
    Vec pos;

    std::vector<Vec> items;
    std::vector<Vec> protos;

    void init(int vec_size, int num_labels, std::mt19937 &rng) {
        this->vec_size = vec_size;    

        trace = gen_rand(vec_size, rng);
        pos = gen_rand(vec_size, rng);

        items.resize(num_labels);
        protos.resize(num_labels);

        for (int i = 0; i < num_labels; i++) {
            items[i] = gen_rand(vec_size, rng);
            protos[i] = gen_rand(vec_size, rng);
        }
    }

    // returns prediction
    int step(int index, float alpha=0.01f) {
        // learn prototype
        protos[index] = normalized(protos[index] + alpha * trace);

        trace = normalized(bind(pos, items[index]) + trace);
        pos = normalized(bind(pos, pos));
        
        // find post similar prototype
        int max_index = 0;
        float max_sim = -999999.0f;

        for (int i = 0; i < protos.size(); i++) {
            float sim = similarity(protos[i], trace);

            if (sim > max_sim) {
                max_sim = sim;
                max_index = i;
            }
        }
        
        return max_index;
    }
};

int main() {
    const int vec_size = 64;

    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "SPA", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    std::mt19937 rng(time(nullptr));

    //std::vector<Vec> vecs(3);

    //for (int i = 0; i < vecs.size(); i++)
    //    vecs[i] = gen_rand(vec_size, rng);

    //Vec v = add(add(bind(vecs[0], vecs[1]), bind(vecs[2], vecs[3])), bind(vecs[4], vecs[5]));

    //Vec r = bind(v, invert(vecs[1]));

    //int max_index = 0;
    //float max_sim = -999999.0f;

    //for (int i = 0; i < vecs.size(); i++) {
    //    float sim = similarity(r, vecs[i]);

    //    if (sim > max_sim) {
    //        max_sim = sim;
    //        max_index = i;
    //    }
    //}

    //std::cout << "Index: " << max_index << std::endl;

    Sequencer seq;
    seq.init(vec_size, 4, rng);

    std::vector<int> sequence = { 0, 0, 0, 0, 1, 2, 3 };

    for (int t = 0; t < 10000; t++) {
        int index = sequence[t % sequence.size()];

        int pred_index = seq.step(index);

        std::cout << index << " " << pred_index << std::endl;
    }

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

        window.clear(sf::Color::Black);
        
        window.display();
    } while (!quit);

    return 0;
}
