#include <vector>

#include <random>

struct Diffuser {
    std::mt19937 rng;

    int width;
    int height;
    int cells_per_column;
    int radius;

    std::vector<float> weights;
    std::vector<float> deltas;

    std::vector<float> visible_acts;
    std::vector<float> hidden_acts;

    std::vector<int> visible_cis;
    std::vector<int> hidden_cis;

    void init_random(
        int width,
        int height,
        int cells_per_column,
        int radius
    );

    void step(
        const std::vector<int> &input_cis,
        float lr
    );

    void clear_deltas() {
        std::fill(deltas.begin(), deltas.end(), 0.0f);
    }

    void apply_deltas();

    void reconstruct(
        const std::vector<int> &recon_cis
    );
};
