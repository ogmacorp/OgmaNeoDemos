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

class ART {
public:
    int num_input_columns;
    int input_column_size;
    int num_hidden_columns;
    int hidden_column_size;

    std::vector<unsigned char> weights;
    std::vector<int> weight_totals;
    std::vector<int> hidden_state;
    std::vector<bool> fasted;

    float choice = 0.01f;
    float vigilance = 0.9f;
    float lr = 0.5f;

    void init(
            int num_input_columns,
            int input_column_size,
            int num_hidden_columns,
            int hidden_column_size,
            std::mt19937 &rng
            ) {
        this->num_input_columns = num_input_columns;
        this->input_column_size = input_column_size;
        this->num_hidden_columns = num_hidden_columns;
        this->hidden_column_size = hidden_column_size;

        std::uniform_int_distribution<int> weight_dist(0, 5);

        weights.resize(num_input_columns * input_column_size * num_hidden_columns * hidden_column_size);

        for (int i = 0; i < weights.size(); i++)
            weights[i] = weight_dist(rng);

        weight_totals.resize(num_hidden_columns * hidden_column_size, 0);

        hidden_state.resize(num_hidden_columns, 0);

        fasted.resize(num_hidden_columns * hidden_column_size, false);
    }

    void step(
        const std::vector<int> &inputs,
        bool learn_enabled = true
    ) {
        int global_max_index = 0;
        float global_max_activation = 0.0f;
        bool learn_flag = false;

        for (int i = 0; i < num_hidden_columns; i++) {
            int max_index = -1;
            float max_activation = 0.0f;

            int max_complete_index = 0;
            float max_complete_activation = 0.0f;

            for (int j = 0; j < hidden_column_size; j++) {
                float sum = 0.0f;

                for (int k = 0; k < num_input_columns; k++) {
                    int in_index = inputs[k];

                    sum += weights[in_index + input_column_size * (k + num_input_columns * (j + hidden_column_size * i))] / 255.0f; 
                }

                float complemented = sum - weight_totals[j + hidden_column_size * i] / 255.0f + num_input_columns * (input_column_size - 1);

                float match = complemented / (num_input_columns * (input_column_size - 1));

                float activation = complemented / (choice + num_input_columns * input_column_size - weight_totals[j + hidden_column_size * i] / 255.0f);

                if (match >= vigilance && activation >= max_activation) {
                    max_activation = activation;
                    max_index = j;
                }

                if (activation >= max_complete_activation) {
                    max_complete_activation = activation;
                    max_complete_index = j;
                }
            }

            hidden_state[i] = (max_index == -1 ? max_complete_index : max_index);

            float global_activation = (max_index == -1 ? 0.0f : max_complete_activation);

            if (global_activation > global_max_activation) {
                global_max_activation = global_activation;
                global_max_index = i;

                learn_flag = (max_index != -1);
            }
        }

        if (learn_enabled && learn_flag) {
            int ij = hidden_state[global_max_index] + hidden_column_size * global_max_index;

            float rate = fasted[ij] ? lr : 1.0f;

            for (int k = 0; k < num_input_columns; k++) {
                int in_index = inputs[k];

                int wi = in_index + input_column_size * (k + num_input_columns * ij);

                unsigned char old_weight = weights[wi];

                unsigned char new_weight = std::min<int>(255, old_weight + std::ceil(rate * (255.0f - old_weight)));

                weights[wi] = new_weight;

                weight_totals[ij] += new_weight - old_weight;
            }

            fasted[ij] = true;
        }
    }
};

int main() {
    std::mt19937 rng(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::normal_distribution<float> nDist(0.0f, 1.0f);

    omp_set_num_threads(8);

    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    ART a;
    a.init(28 * 28, 2, 49, 64, rng);

    int num_hidden = 256;
    std::vector<float> classifier(10 * num_hidden);
    std::vector<float> hidden_weights(num_hidden * a.num_hidden_columns * a.hidden_column_size);
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

    std::vector<int> inputs(28 * 28, 0);

    float accuracy = 0.0f;

    for (int it = 0; it < 1000000; it++) {
        int index = dataDist(rng);

        int label = data[index].label;

        for (int i = 0; i < inputs.size(); i++)
            inputs[i] = (data[index].data[i % 28][i / 28] > 127);

        a.step(inputs);

        for (int i = 0; i < num_hidden; i++) {
            float sum = 0.0f;

            for (int j = 0; j < a.num_hidden_columns; j++) {
                sum += hidden_weights[a.hidden_state[j] + a.hidden_column_size * (j + a.num_hidden_columns * i)];
            }

            hidden_state[i] = std::tanh(sum / a.num_hidden_columns) * 0.5f + 0.5f;
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

            for (int j = 0; j < a.num_hidden_columns; j++) {
                hidden_weights[a.hidden_state[j] + a.hidden_column_size * (j + a.num_hidden_columns * i)] += error;
            }
        }

        if (it % 1000 == 0) {
            std::cout << "Iteration " << it << " accuracy " << accuracy << std::endl;
        }
    }

    return 0;
}

