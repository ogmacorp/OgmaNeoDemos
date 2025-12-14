#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <valarray>
#include <random>
#include <omp.h>

#include <aogmaneo/hierarchy.h>

#include <assert.h>

using namespace aon;

const int S = 1024;
const int L = 32;
const int N = S * L;

typedef Hierarchy<S, L> Hierarchy1;
typedef Vec<S, L> Vec1;
typedef Bundle<S, L> Bundle1;

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
    set_num_threads(8);
    global_state = rand_get_state(time(nullptr));

    Array<Layer_Desc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int2(1, 1);
    }

    Array<Vec1> vecs(10);

    for (int i = 0; i < vecs.size(); i++)
        vecs[i] = Vec1::randomized();

    Array<IO_Desc> io_descs(1);
    io_descs[0] = IO_Desc(Int2(1, 1), IO_Type::prediction);

    Hierarchy1 h;
    h.init_random(io_descs, lds);

    Array<Vec1> input_vecs(1);

    Array<Array_View<Vec1>> all_input_vecs(1);
    all_input_vecs[0] = input_vecs;

    for (int e = 0; e < 10000; e++) {
        int count = 0;

        int length = 12;
        int quant = 4;

        for (int t = 0; t < length * 1000; t++) {
            int index = (t % length < quant ? (t % length + 1) : 0);

            input_vecs[0] = vecs[index];

            h.step(all_input_vecs, true);

            Vec1 pred = h.get_prediction_vecs(0)[0];

            // search for closest vector
            int max_index = 0;
            int max_similarity = -999999;

            for (int i = 0; i < vecs.size(); i++) {
                int similarity = pred.dot(vecs[i]);

                if (similarity > max_similarity) {
                    max_similarity = similarity;
                    max_index = i;
                }
            }

            //print(h.get_layer(4).get_hidden_vecs()[0]);

            std::cout << index << " -> " << max_index << std::endl;
            
            int t2 = t + 1;

            count += (max_index != (t2 % 9 < 4 ? (t2 % 9 + 1) : 0));
        }

        std::cout << count << std::endl;
    }

    return 0;
}
