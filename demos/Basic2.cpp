#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

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

    int num_hidden = 256;
    std::vector<float> classifier(10 * num_hidden);
    std::vector<float> hidden_weights(num_hidden * 28 * 28);
    std::vector<float> hidden_state(num_hidden, 0.0f);
    std::vector<float> hidden_errors(num_hidden, 0.0f);

    for (int i = 0; i < classifier.size(); i++)
        classifier[i] = nDist(rng) * 0.01f;

    for (int i = 0; i < hidden_weights.size(); i++)
        hidden_weights[i] = nDist(rng) * 1.0f;

    std::vector<float> probs(10, 0.0f);
    mnist_data* data;
    unsigned int cnt;
    int ret = mnist_load("resources/train-images-idx3-ubyte", "resources/train-labels-idx1-ubyte", &data, &cnt);

    if (ret) {
        printf("An error occured: %d\n", ret);
        return -1;
    } else {
        printf("image count: %d\n", cnt);
    }

    // Train
    std::uniform_int_distribution<int> dataDist(0, cnt);

    std::vector<float> inputs(28 * 28, 0.0f);

    float accuracy = 0.0f;

    for (int it = 0; it < 1000000; it++) {
        int index = dataDist(rng);

        int label = data[index].label;

        for (int i = 0; i < inputs.size(); i++)
            inputs[i] = data[index].data[i % 28][i / 28] / 255.0f;

        for (int i = 0; i < num_hidden; i++) {
            float sum = 0.0f;

            for (int j = 0; j < 28 * 28; j++) {
                sum += hidden_weights[j + 28 * 28 * i] * inputs[j];
            }

            hidden_state[i] = std::tanh(sum / (28 * 28)) * 0.5f + 0.5f;
        }

        int max_index = 0;
        float max_act = -99999.0f;

        for (int i = 0; i < 10; i++) {
            float sum = 0.0f;

            for (int j = 0; j < num_hidden; j++) {
                sum += classifier[j + num_hidden * i] * hidden_state[j];
            }

            probs[i] = sum / num_hidden;

            if (probs[i] > max_act) {
                max_act = probs[i];
                max_index = i;
            }
        }

        accuracy += 0.001f * ((max_index == label) - accuracy);

        float total = 0.0f;

        for (int i = 0; i < 10; i++) {
            probs[i] = std::exp(probs[i] - max_act);

            total += probs[i];
        }

        std::fill(hidden_errors.begin(), hidden_errors.end(), 0.0f);

        for (int i = 0; i < 10; i++) {
            probs[i] /= std::max(0.0001f, total);

            float error = 0.5f * ((i == label) - probs[i]);

            for (int j = 0; j < num_hidden; j++) {
                hidden_errors[j] += classifier[j + num_hidden * i] * error;
            }

            for (int j = 0; j < num_hidden; j++) {
                classifier[j + num_hidden * i] += error * hidden_state[j];
            }
        }

        for (int i = 0; i < num_hidden; i++) {
            float error = hidden_errors[i] * (1.0f - hidden_state[i]) * hidden_state[i];

            for (int j = 0; j < 28 * 28; j++) {
                hidden_weights[j + 28 * 28 * i] += error * inputs[j];
            }
        }

        if (it % 1000 == 0) {
            std::cout << "Iteration " << it << " accuracy " << accuracy << std::endl;
        }
    }

    return 0;
}

