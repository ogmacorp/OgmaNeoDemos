#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <valarray>
#include <random>
#include <omp.h>

#include <vec.h>

#include <assert.h>

using namespace v;

const int S = 256;
const int L = 16;
const int N = S * L;

typedef Vec<S, L> Vec1;
typedef Bundle<S, L> Bundle1;

//class Resonator_Mat {
//private:
//    static const int ss = S * (S + 1) / 2; // swap - 1 instead of + 1 for ignoring diagonal
//
//    int buffer[ss];
//
//public:
//    Resonator_Mat()
//    {}
//
//    Resonator_Mat(
//        const Array<Vec<S>> &codes,
//        int codes_start,
//        int num_codes
//    ) {
//        set_from(codes, codes_start, num_codes);
//    }
//
//    void set_from(
//        const Array<Vec<S>> &codes,
//        int codes_start,
//        int num_codes
//    ) {
//        for (int r = 0; r < S; r++) {
//            int start = r * (r + 1) / 2;
//
//            for (int c = 0; c <= r; c++) {
//                int index = c + start;
//
//                int sum = 0;
//
//                for (int c2 = 0; c2 < num_codes; c2++)
//                    sum += codes[codes_start + c2].get(r) * codes[codes_start + c2].get(c);
//
//                buffer[index] = sum;
//            }
//        }
//    }
//
//    Bundle<S> operator*(
//        const Vec<S> &v
//    ) {
//        Bundle<S> result = 0;
//
//        for (int r = 0; r < S; r++) {
//            int start = r * (r + 1) / 2;
//
//            // lower triangle, duplicated into upper triangle as well
//            for (int c = 0; c < r; c++) { // ignore diagonal for now
//                int index = c + start;
//                
//                result[c] += buffer[index] * v.get(r); // lower triangle (original)
//                result[r] += buffer[index] * v.get(c); // upper triangle (duplicate)
//            }
//
//            // now do diagonal (not duplicated)
//            result[r] += buffer[r + start] * v.get(r);
//        }
//
//        return result;
//    }
//};

Vec1 embedding1d(float x, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 2] * x + loc[i * 2 + 1];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

float unembedding1d(const Vec1 &v, const std::valarray<float> &loc) {
    // mat mul
    float x = 0.0f;

    for (int i = 0; i < S; i++) {
        float val = v[i] / static_cast<float>(L);

        val = (val - loc[i * 2 + 1]) / v::max(v::limit_small, abs(loc[i * 2])) * (loc[i * 2] > 0.0f ? 1.0f : -1.0f);

        x += val;
    }

    return x / S;
}

Vec1 embedding2d(float x, float y, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 3] * x + loc[i * 3 + 1] * y + loc[i * 3 + 2];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

Vec1 embedding3d(float x, float y, float z, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 4] * x + loc[i * 4 + 1] * y + loc[i * 4 + 2] * z + loc[i * 4 + 3];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

void print(const Vec1 &v) {
    std::cout << "[ ";

    for (int i = 0; i < v.segments(); i++)
        std::cout << static_cast<int>(v[i]) << " ";

    std::cout << " ]" << std::endl;
}

void print(const Bundle1 &b) {
    std::cout << "[ ";

    for (int i = 1; i < b.size(); i++)
        std::cout << b[i] << " ";

    std::cout << " ]" << std::endl;
}

int main() {
    // Create hierarchy
    //set_num_threads(8);
    //global_state = rand_get_state(time(nullptr));

    //Array<Vec1> vecs(100);
    //Array<Vec1> pos(vecs.size());

    //std::valarray<float> loc(2 * S);

    //for (int i = 0; i < loc.size(); i++)
    //    loc[i] = aon::rand_normalf();

    //for (int i = 0; i < vecs.size(); i++) {
    //    vecs[i] = Vec1::randomized();
    //    pos[i] = embedding1d(i * 1.0f, loc);
    //}

    //Bundle1 sup = 0;

    //for (int i = 0; i < vecs.size(); i++) {
    //    sup += vecs[i] * pos[i];
    //}
    //
    //Vec1 vs = sup.thin();

    //Vec1 a = vs / pos[6];

    //// most similar
    //int max_index = 0;
    //int max_similarity = -999999;

    //for (int i = 0; i < vecs.size(); i++) {
    //    int similarity = a.dot(vecs[i]);

    //    if (similarity > max_similarity) {
    //        max_similarity = similarity;
    //        max_index = i;
    //    }
    //}

    //std::cout << max_index << std::endl;

    //return 0;

    //Array<Vec1> vecs(3 * 16);

    //for (int i = 0; i < vecs.size(); i++)
    //    vecs[i] = Vec1::randomized();

    //Resonator_Mat rm_x(vecs, 0, 16);
    //Resonator_Mat rm_y(vecs, 16, 16);
    //Resonator_Mat rm_z(vecs, 32, 16);

    //Vec1 a = vecs[0 + 0] * vecs[1 + 16] * vecs[2 + 32];

    //Bundle<S> sx = 0;
    //Bundle<S> sy = 0;
    //Bundle<S> sz = 0;

    //for (int i = 0; i < 32; i++) {
    //    sx += vecs[i + 0];
    //    sy += vecs[i + 16];
    //    sz += vecs[i + 32];
    //}

    //Vec1 x = sx.thin();
    //Vec1 y = sy.thin();
    //Vec1 z = sz.thin();

    //for (int it = 0; it < 16; it++) {
    //    Vec1 xn = (rm_x * (a * y * z)).thin();
    //    Vec1 yn = (rm_y * (a * x * z)).thin();
    //    Vec1 zn = (rm_z * (a * x * y)).thin();

    //    x = xn;
    //    y = yn;
    //    z = zn;
    //}

    //{
    //    // most similar
    //    int max_index = 0;
    //    int max_similarity = -999999;

    //    for (int i = 0; i < 16; i++) {
    //        int similarity = x.dot(vecs[i + 0]);

    //        if (similarity > max_similarity) {
    //            max_similarity = similarity;
    //            max_index = i;
    //        }
    //    }

    //    std::cout << "MAX_X: " << max_index << std::endl;
    //}

    //{
    //    // most similar
    //    int max_index = 0;
    //    int max_similarity = -999999;

    //    for (int i = 0; i < 16; i++) {
    //        int similarity = y.dot(vecs[i + 16]);

    //        if (similarity > max_similarity) {
    //            max_similarity = similarity;
    //            max_index = i;
    //        }
    //    }

    //    std::cout << "MAX_Y: " << max_index << std::endl;
    //}

    //{
    //    // most similar
    //    int max_index = 0;
    //    int max_similarity = -999999;

    //    for (int i = 0; i < 16; i++) {
    //        int similarity = z.dot(vecs[i + 32]);

    //        if (similarity > max_similarity) {
    //            max_similarity = similarity;
    //            max_index = i;
    //        }
    //    }

    //    std::cout << "MAX_Z: " << max_index << std::endl;
    //}
    //
    //return 0;

    //Array<Hierarchy1::Layer_Desc> lds(1);

    //for (int i = 0; i < lds.size(); i++) {
    //    lds[i].hidden_size = Int4(1, 1, 4, 32);
    //    lds[i].ticks_per_update = 1;
    //    lds[i].temporal_horizon = 1;
    //}

    //int vals = 4;
    //int res = 4;

    //Array<Hierarchy1::IO_Desc> io_descs(1);
    //io_descs[0] = Hierarchy1::IO_Desc(Int4(1, 1, 1, res), IO_Type::prediction);

    //Hierarchy1 h;
    //h.init_random(io_descs, lds);

    //Array<Encoder1::Visible_Layer_Desc> evlds(1);
    //evlds[0].size = Int4(1, 1, vals, res);

    //Encoder1 e;
    //e.init_random(Int4(1, 1, 3, 16), 1.0f, evlds);

    //Encoder1::Params params;

    //Int_Buffer input_cis(vals);

    //Array<Int_Buffer_View> all_input_cis(1);
    //all_input_cis[0] = input_cis;
    //
    //Array<Int_Buffer> data(4);

    //for (int i = 0; i < data.size(); i++) {
    //    data[i].resize(vals);

    //    for (int j = 0; j < vals; j++)
    //        data[i][j] = aon::rand() % res;
    //}

    //for (int t = 0; t < 1; t++) {
    //    int index = t % data.size();

    //    input_cis = data[index];

    //    e.set_input_cis(0, input_cis);

    //    //h.step(all_input_cis, true);
    //    e.step(true, params);

    //    e.reconstruct(0, params);

    //    for (int i = 0; i < e.get_hidden_size().z; i++)
    //        std::cout << e.get_hidden_cis()[i] << " ";

    //    std::cout << std::endl;
    //}

    //// corrupt data
    //Array<Int_Buffer> data_corrupted = data;

    //for (int i = 0; i < data.size(); i++) {
    //    for (int j = 0; j < vals; j++)
    //        if (aon::randf() < 0.2f)
    //            data_corrupted[i][j] = aon::rand() % res;
    //}

    //for (int index = 0; index < data.size(); index++) {
    //    input_cis = data[index]; // data_corrupted

    //    e.set_input_cis(0, input_cis);

    //    //h.step(all_input_cis, true);
    //    e.step(false, params);

    //    e.reconstruct(0, params);

    //    for (int i = 0; i < e.get_visible_layer(0).recon_cis.size(); i++)
    //        std::cout << e.get_visible_layer(0).recon_cis[i] << " " << data[index][i] << " " << data_corrupted[index][i] << std::endl;

    //    std::cout << std::endl;
    //    std::cout << std::endl;
    //}

    //return 0;

    omp_set_num_threads(8);
    global_state = rand_get_state(time(nullptr));

    unsigned int windowWidth = 1600;
    unsigned int windowHeight = 1200;

    sf::RenderWindow window;

    window.create(sf::VideoMode(sf::Vector2u(windowWidth, windowHeight)), "VSA Test", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    std::mt19937 rng(time(nullptr));

    std::valarray<float> C(S * 2); // color
    std::valarray<float> Z(S * 2); // location

    std::normal_distribution<float> ndist(0.0f, 1.0f);

    int color_res = 6;

    for (int i = 0; i < C.size(); i++)
        C[i] = ndist(rng) * 1.0f;

    for (int i = 0; i < Z.size(); i++)
        Z[i] = ndist(rng) * 1.0f;

    float color_div = 1.0f / (color_res - 1);

    std::vector<Vec1> color_lookup(color_res * color_res * color_res);

    for (int x = 0; x < color_res; x++)
        for (int y = 0; y < color_res; y++)
            for (int z = 0; z < color_res; z++) {
                float xf = x * color_div;
                float yf = y * color_div;
                float zf = z * color_div;

                std::cout << xf << " " << unembedding1d(embedding1d(xf, C), C) << std::endl;

                color_lookup[z + color_res * (y + color_res * x)] = embedding1d(xf, C) * embedding1d(yf, C).permute(1) * embedding1d(zf, C).permute(2);
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
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                quit = true;
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::Escape))
                quit = true;
        }

        int speed = 1.0f;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::A))
            target_x -= speed;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::D))
            target_x += speed;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::W))
            target_y -= speed;
        else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Key::S))
            target_y += speed;

        window.clear(sf::Color::Black);

        // bind in a small radius
        Bundle<S, L> res = 0;

        //int width = 1;

        //for (float a = 0.0f; a < 2.0f * 3.1415f; a += 0.1f) {
        //    float adx = std::cos(a) * 10.0f;
        //    float ady = std::sin(a) * 10.0f;

        //    for (int dx = -width; dx <= width; dx++)
        //        for (int dy = -width; dy <= width; dy++) {
        //            float x = target_x + adx + dx;
        //            float y = target_y + ady + dy;

        //            res += bind(test_vec, embedding2d(x, y, Z));
        //        }
        //}

        float scale = 1.0f / 32.0f;

        for (int x = 0; x < test_img.getSize().x; x++)
            for (int y = 0; y < test_img.getSize().y; y++) {
                sf::Color color = test_img.getPixel(sf::Vector2u(x, y));
                
                float r = color.r / 255.0f;
                float g = color.g / 255.0f;
                float b = color.b / 255.0f;

                Vec1 c = embedding1d(r, C) * embedding1d(g, C).permute(1) * embedding1d(b, C).permute(2);

                Vec1 pos = embedding1d((target_x + x) * scale, Z) * embedding1d((target_y + y) * scale, Z).permute(1);

                res += c * pos;
            }

        //for (int i = 0; i < res.size(); i++)
        //    std::cout << res[i] << std::endl;

        Vec1 res_final = res.thin(); // final state vector, superposition of bound test_vec at different locations

        //print(res_final);

        sf::Image img(sf::Vector2u(64, 64));
        
        #pragma omp parallel for
        for (int x = 0; x < img.getSize().x; x++)
            for (int y = 0; y < img.getSize().y; y++) {
                Vec1 pos = embedding1d(x * scale, Z) * embedding1d(y * scale, Z).permute(1);

                Vec1 color_vec = res_final / pos;

                int ms = -999999;
                int mx = 0;
                int my = 0;
                int mz = 0;

                for (int cx = 0; cx < color_res; cx++)
                    for (int cy = 0; cy < color_res; cy++)
                        for (int cz = 0; cz < color_res; cz++) {
                            int s = color_vec.dot(color_lookup[cz + color_res * (cy + color_res * cx)]);

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
                img.setPixel(sf::Vector2u(x, y), color);
            }

        tex.loadFromImage(img);

        sf::Sprite s(tex);
        s.setScale(sf::Vector2f(4.0f, 4.0f));

        window.draw(s);
        
        window.display();
    } while (!quit);

    return 0;
}

