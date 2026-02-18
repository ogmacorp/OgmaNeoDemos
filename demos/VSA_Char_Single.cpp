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
#include "vec.h"

#include <assert.h>

using namespace aon;

const int S = 256;
const int L = 8;

typedef v::Vec<S, L> Vec1;
typedef v::Bundle<S, L> Bundle1;

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

int main() {
    std::mt19937 rng(time(nullptr));

    // Create hierarchy
    set_num_threads(8);
    global_state = rand_get_state(12345);

    aon::Array<aon::Hierarchy::Layer_Desc> lds(2);

    for (int i = 0; i < lds.size(); i++)
        lds[i].hidden_size = aon::Int3(7, 7, 64);

    aon::Array<aon::Hierarchy::IO_Desc> iods(1);
    iods[0].size = Int3( 16, 16, 8);
    iods[0].up_radius = 5;
    iods[0].down_radius = 3;

    Hierarchy h;
    h.init_random(iods, lds);

    std::ifstream from_file("resources/ts_snippet.txt");

    if (!from_file.is_open()) {
        std::cout << "Could not open!" << std::endl;

        return 0;
    }

    int max_index = 0;

    std::vector<char> word_buffer(128, ' ');
    int last_char_index = word_buffer.size() - 1;

    Array<Vec1> pos(word_buffer.size());

    std::valarray<float> loc(S * 2); // location

    std::normal_distribution<float> ndist(0.0f, 1.0f);

    for (int i = 0; i < loc.size(); i++)
        loc[i] = ndist(rng);

    for (int i = 0; i < pos.size(); i++)
        pos[i] = embedding1d(i * 1.0f, loc);

    Array<Vec1> vecs(128);

    for (int i = 0; i < vecs.size(); i++)
        vecs[i] = Vec1::randomized();

    Array<S32_Array_View> input_cis(1);
    S32_Array csdr(S, 0);
    input_cis[0] = csdr;

    {
        int t = 0;

        for (int e = 0; e < 10; e++) {
            int correct = 0;
            int count = 0;

            while (from_file.good() && !from_file.eof()) {
                char c;

                from_file.read(&c, 1);

                int index = c;

                if (max_index == index)
                    correct++;

                count++;
                t++;

                // shift
                for (int i = 0; i < last_char_index; i++)
                    word_buffer[i] = word_buffer[i + 1];

                word_buffer[last_char_index] = c;

                if (c == ' ' || c == '\n') {
                    // build word vector from buffer
                    Bundle1 b = 0;
                    int word_size = 0;

                    for (int i = last_char_index; i >= 0; i--) {
                        b += vecs[word_buffer[i]] * pos[i];
                        word_size++;

                        if (i < last_char_index && (word_buffer[i] == ' ' || word_buffer[i] == '\n'))
                            break;
                    }

                    Vec1 v = b.thin();

                    // to csdr
                    for (int i = 0; i < S; i++)
                        csdr[i] = v[i];

                    input_cis[0] = csdr;

                    // step
                    h.step(input_cis, true);

                    {
                        int word_size2 = 0;
                        std::vector<char> word_buffer2(word_buffer.size(), ' ');

                        for (int i = last_char_index; i >= 0; i--) {
                            Vec1 cv = v / pos[i];

                            // search for closest vector
                            max_index = 0;
                            int max_similarity = 0;

                            for (int j = 0; j < vecs.size(); j++) {
                                int similarity = cv.dot(vecs[j]);

                                if (similarity > max_similarity) {
                                    max_similarity = similarity;
                                    max_index = j;
                                }
                            }

                            char ch = max_index;

                            if (i < last_char_index && (ch == ' ' || ch == '\n'))
                                break;

                            word_buffer2[i] = ch;

                            word_size2++;
                        }

                        std::string result = "";

                        // append word
                        for (int i = word_buffer2.size() - word_size2; i < word_buffer2.size(); i++) {
                            result += word_buffer2[i];
                        }

                        //std::cout << result;
                    }
                }

                if (count % 5000 == 4999)
                    std::cout << "c: " << count << std::endl;
            }

            from_file.clear();
            from_file.seekg(0);

            std::cout << (static_cast<float>(correct) / count) << std::endl;
        }
    }

    for (int t = 0; t < 1000; t++) {
        input_cis[0] = h.get_prediction_cis(0);

        h.step(input_cis, false);

        // predict
        Vec1 p;

        for (int i = 0; i < S; i++)
            p[i] = h.get_prediction_cis(0)[i];

        {
            int word_size2 = 0;
            std::vector<char> word_buffer2(word_buffer.size(), ' ');

            for (int i = last_char_index; i >= 0; i--) {
                Vec1 cv = p / pos[i];

                // search for closest vector
                int max_index2 = 0;
                int max_similarity = 0;

                for (int j = 0; j < vecs.size(); j++) {
                    int similarity = cv.dot(vecs[j]);

                    if (similarity > max_similarity) {
                        max_similarity = similarity;
                        max_index2 = j;
                    }
                }

                char ch = max_index2;

                if (i < last_char_index && (ch == ' ' || ch == '\n'))
                    break;

                word_buffer2[i] = ch;

                word_size2++;
            }

            std::string result = "";

            // append word
            for (int i = word_buffer2.size() - word_size2; i < word_buffer2.size(); i++) {
                result += word_buffer2[i];
            }

            std::cout << result;
        }
    }

    std::cout << "-------------------- DONE ----------------------" << std::endl;

    return 0;
}

