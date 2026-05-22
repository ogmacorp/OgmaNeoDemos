#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <valarray>
#include <random>
#include <omp.h>

#include <ultrasparse/model.h>

#include <assert.h>

using namespace us;

const int nx = 10000;
const int nh = 10000;
const int px = 8;
const int ph = 8;
const int vx = 20; 
const int vh = 20; 

void print(const SDR &v) {
    std::cout << "[ ";

    for (int i = 0; i < v.get_p(); i++)
        std::cout << static_cast<int>(v[i]) << " ";

    std::cout << " ]" << std::endl;
}

int main() {
    // Create hierarchy
    set_num_threads(4);
    global_state = rand_get_state(12345);

    Array<SDR> sdrs(128);

    for (int i = 0; i < sdrs.size(); i++)
        sdrs[i].init_random(nx, px);

    s32 num_layers = 1;
    S32_Array nhs(num_layers);
    S32_Array phs(num_layers);
    S32_Array vxs(num_layers);
    S32_Array vhs(num_layers);

    for (int i = 0; i < num_layers; i++) {
        nhs[i] = nh;
        phs[i] = ph;
        vxs[i] = vx;
        vhs[i] = vh;
    }

    Model m;
    m.init(nx, px, nhs, phs, vxs, vhs);

    SDR x_pred(nx, px);

    std::ifstream from_file("ts_snippet.txt");

    if (!from_file.is_open()) {
        std::cout << "Could not open!" << std::endl;

        return 0;
    }

    {
        int t = 0;

        for (int e = 0; e < 1000; e++) {
            int correct = 0;
            int count = 0;

            while (from_file.good() && !from_file.eof()) {
                // search for closest vector
                int max_index = 0;
                int max_similarity = 0;

                for (int i = 0; i < sdrs.size(); i++) {
                    int similarity = x_pred.overlap(sdrs[i]);

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

                //std::cout << (char)max_index;
                count++;
                t++;

                m.step(sdrs[index], true);

                print(m.get_h(0));

                x_pred = m.get_x_pred();

                if (count % 5000 == 4999)
                    std::cout << "c: " << count << std::endl;
            }

            from_file.clear();
            from_file.seekg(0);

            std::cout << (static_cast<float>(correct) / count) << std::endl;
        }
    }

    for (int t = 0; t < 1000; t++) {
        int max_index = 0;
        int max_similarity = 0;

        for (int i = 0; i < sdrs.size(); i++) {
            int similarity = x_pred.overlap(sdrs[i]);

            if (similarity > max_similarity) {
                max_similarity = similarity;
                max_index = i;
            }
        }

        std::cout << static_cast<char>(max_index);

        m.step(sdrs[max_index], false);

        x_pred = m.get_x_pred();
    }

    std::cout << "-------------------- DONE ----------------------" << std::endl;

    return 0;
}

