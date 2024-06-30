#include "diffuser.h"

#include <omp.h>

void Diffuser::init_random(
    int width,
    int height,
    int cells_per_column,
    int radius
) {
    this->width = width;
    this->height = height;
    this->cells_per_column = cells_per_column;
    this->radius = radius;

    visible_cis.resize(width * height);
    std::fill(visible_cis.begin(), visible_cis.end(), 0);

    hidden_cis.resize(width * height);
    std::fill(hidden_cis.begin(), hidden_cis.end(), 0);

    int diam = radius * 2 + 1;
    int area = diam * diam;

    weights.resize(cells_per_column * area * cells_per_column); // shared weights

    std::uniform_real_distribution<float> dist_weight(0.99f, 1.0f);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = dist_weight(rng);

    deltas.resize(weights.size());
    std::fill(deltas.begin(), deltas.end(), 0.0f);

    visible_acts.resize(hidden_cis.size() * cells_per_column);
    hidden_acts.resize(visible_cis.size() * cells_per_column);
}

void Diffuser::step(
    const std::vector<int> &input_cis,
    float lr
) {
    int diam = radius * 2 + 1;

    //#pragma omp parallel for
    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            int i = y + x * height;

            int max_index = 0;
            float max_act = -999999.0f;

            for (int ci = 0; ci < cells_per_column; ci++) {
                float sum = 0.0f;
                int count = 0;

                for (int dy = -radius; dy <= radius; dy++)
                    for (int dx = -radius; dx <= radius; dx++) {
                        int ox = x + dx;
                        int oy = y + dy;

                        if (ox < 0 || oy < 0 || ox >= width || oy >= height)
                            continue;

                        int j = oy + ox * height;
                        
                        int input_ci = input_cis[j];

                        int wi = input_ci + cells_per_column * ((dx + radius) + diam * ((dy + radius) + diam * ci));

                        sum += weights[wi];
                        count++;
                    }

                sum /= count;

                hidden_acts[ci + i * cells_per_column] = sum;

                if (sum > max_act) {
                    max_act = sum;
                    max_index = ci;
                }
            }

            hidden_cis[i] = max_index;
        }

    if (lr != 0.0f) {
        // reconstruct and learn
        for (int x = 0; x < width; x++)
            for (int y = 0; y < height; y++) {
                int i = y + x * height;

                int max_index = 0;
                float max_act = -999999.0f;

                for (int ci = 0; ci < cells_per_column; ci++) {
                    float sum = 0.0f;
                    int count = 0;

                    for (int dy = -radius; dy <= radius; dy++)
                        for (int dx = -radius; dx <= radius; dx++) {
                            int ox = x + dx;
                            int oy = y + dy;

                            if (ox < 0 || oy < 0 || ox >= width || oy >= height)
                                continue;

                            int j = oy + ox * height;
                            
                            int wi = ci + cells_per_column * ((-dx + radius) + diam * ((-dy + radius) + diam * hidden_cis[j]));

                            sum += weights[wi];
                            count++;
                        }

                    sum /= count;

                    visible_acts[ci + i * cells_per_column] = sum;

                    if (sum > max_act) {
                        max_act = sum;
                        max_index = ci;
                    }
                }

                visible_cis[i] = max_index;

                if (max_index != input_cis[i]) {
                    for (int ci = 0; ci < cells_per_column; ci++) {
                        float delta = lr * ((ci == input_cis[i]) - std::exp(visible_acts[ci + i * cells_per_column] - 1.0f));

                        for (int dy = -radius; dy <= radius; dy++)
                            for (int dx = -radius; dx <= radius; dx++) {
                                int ox = x + dx;
                                int oy = y + dy;

                                if (ox < 0 || oy < 0 || ox >= width || oy >= height)
                                    continue;

                                int j = oy + ox * height;
                                
                                int wi = ci + cells_per_column * ((-dx + radius) + diam * ((-dy + radius) + diam * hidden_cis[j]));

                                deltas[wi] += delta;
                            }
                    }
                }
            }
    }
}

void Diffuser::apply_deltas() {
    for (int i = 0; i < weights.size(); i++)
        weights[i] += deltas[i];
}

void Diffuser::reconstruct(
    const std::vector<int> &recon_cis
) {
    int diam = radius * 2 + 1;

    for (int x = 0; x < width; x++)
        for (int y = 0; y < height; y++) {
            int i = y + x * height;

            int max_index = 0;
            float max_act = -999999.0f;

            for (int ci = 0; ci < cells_per_column; ci++) {
                float sum = 0.0f;
                int count = 0;

                for (int dy = -radius; dy <= radius; dy++)
                    for (int dx = -radius; dx <= radius; dx++) {
                        int ox = x + dx;
                        int oy = y + dy;

                        if (ox < 0 || oy < 0 || ox >= width || oy >= height)
                            continue;

                        int j = oy + ox * height;
                        
                        int wi = ci + cells_per_column * ((-dx + radius) + diam * ((-dy + radius) + diam * recon_cis[j]));

                        sum += weights[wi];
                        count++;
                    }

                sum /= count;

                visible_acts[ci + i * cells_per_column] = sum;

                if (sum > max_act) {
                    max_act = sum;
                    max_index = ci;
                }
            }

            visible_cis[i] = max_index;
        }
}
