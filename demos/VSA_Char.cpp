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
const int L = 16;

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

    Array<Layer_Desc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int2(1, 1);
        lds[i].num_memories = 4;
    }

    Array<IO_Desc> io_descs(1);
    io_descs[0] = IO_Desc(Int2(1, 1), IO_Type::prediction);

    Hierarchy1 h;
    h.init_random(io_descs, lds);

    Array<Vec1> input_vecs(1);

    Array<Array_View<Vec1>> all_input_vecs(1);
    all_input_vecs[0] = input_vecs;

    std::ifstream from_file("resources/ts_snippet.txt");

    if (!from_file.is_open()) {
        std::cout << "Could not open!" << std::endl;

        return 0;
    }

    {
        int t = 0;

        for (int e = 0; e < 30; e++) {
            int correct = 0;
            int count = 0;

            while (from_file.good() && !from_file.eof()) {
                Vec1 pred = h.get_prediction_vecs(0)[0];

                // search for closest vector
                int max_index = 0;
                int max_similarity = 0;

                for (int i = 0; i < vecs.size(); i++) {
                    int similarity = pred.dot(vecs[i]);

                    if (similarity > max_similarity) {
                        max_similarity = similarity;
                        max_index = i;
                    }
                }

                char c;

                from_file.read(&c, 1);

                int index = c;

                if (max_index == index)
                    correct++;

                count++;
                t++;

                input_vecs[0] = vecs[index];

                h.step(all_input_vecs, true);

                if (count % 5000 == 4999)
                    std::cout << "c: " << count << std::endl;

                if (count % 50000 == 49999) {
                    std::cout << "Saving..." << std::endl;

                    CustomStreamWriter writer;
                    writer.outs.open("char.vohr", std::ios::out | std::ios::binary);
                    h.write(writer);

                    std::cout << "Saved." << std::endl;
                }
            }

            from_file.clear();
            from_file.seekg(0);

            std::cout << (static_cast<float>(correct) / count) << std::endl;
        }

        std::cout << "Saving..." << std::endl;

        CustomStreamWriter writer;
        writer.outs.open("char.vohr", std::ios::out | std::ios::binary);
        h.write(writer);

        std::cout << "Saved." << std::endl;
    }

    for (int t = 0; t < 1000; t++) {
        Vec1 pred = h.get_prediction_vecs(0)[0];

        // search for closest vector
        int max_index = 0;
        int max_similarity = 0;

        for (int i = 0; i < vecs.size(); i++) {
            int similarity = pred.dot(vecs[i]);

            if (similarity > max_similarity) {
                max_similarity = similarity;
                max_index = i;
            }
        }

        std::cout << static_cast<char>(max_index);

        input_vecs[0] = vecs[max_index]; // cleaned up

        h.step(all_input_vecs, false);
    }

    std::cout << "-------------------- DONE ----------------------" << std::endl;

    return 0;
}

