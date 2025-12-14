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

const int S = 64;
const int L = 64;

typedef Vec<S, L> Vec1;
typedef Bundle<S, L> Bundle1;
typedef Hierarchy<S, L> Hierarchy1;

class CustomStreamReader : public aon::Stream_Reader {
public:
    std::ifstream ins;

    void read(
        void* data,
        long len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        long len
    ) override {
        outs.write(static_cast<const char*>(data), len);
    }
};

void print(const Vec1 &v) {
    std::cout << "[ ";

    for (int i = 0; i < v.segments(); i++)
        std::cout << static_cast<int>(v[i]) << " ";

    std::cout << " ]" << std::endl;
}

void print(const Bundle1 &b) {
    std::cout << "[ ";

    for (int i = 0; i < b.size(); i++)
        std::cout << b[i] << " ";

    std::cout << " ]" << std::endl;
}

int main() {
    // Create hierarchy
    set_num_threads(4);
    global_state = rand_get_state(12345);

    Array<Vec1> vecs(128);

    for (int i = 0; i < vecs.size(); i++)
        vecs[i] = Vec1::randomized();

    Array<Vec1> input_vecs(1);

    Array<Array_View<Vec1>> all_input_vecs(1);
    all_input_vecs[0] = input_vecs;

    Hierarchy1 h;

    CustomStreamReader reader;
    reader.ins.open("char.vohr", std::ios::in | std::ios::binary);
    h.read(reader);

    for (int t = 0; t < 5000; t++) {
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

        std::cout << static_cast<char>(max_index);

        if (randf() < 0.03f)
            max_index = aon::rand() % vecs.size();

        input_vecs[0] = vecs[max_index]; // cleaned up

        h.step(all_input_vecs, false);
    }

    std::cout << "-------------------- DONE ----------------------" << std::endl;

    return 0;
}

