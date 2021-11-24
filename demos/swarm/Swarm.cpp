#include "Swarm.h"

void Swarm::init(
    int width,
    int height,
    std::mt19937 &rng
) {
    this->width = width;
    this->height = height;

    cells.resize(width * height);

    std::uniform_real_distribution<float> valueDist(-0.001f, 0.001f);

    for (int i = 0; i < cells.size(); i++) {
        for (int j = 0; j < cells[i].values.size(); j++) {
            cells[i].values[j] = valueDist(rng);
            cells[i].traces[j] = 0.0f;
        }
    }
}

void Swarm::step(
    const std::vector<bool> &inputs,
    float reward,
    std::mt19937 &rng,
    bool learnEnabled
) {
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int i = y + x * height;

            int cellIndex0 = 0;

            for (int dx = -1; dx <= 1; dx++) {
                int ix = x + dx;

                if (ix < 0 || ix >= width)
                    continue;

                if (y == 0) {
                    if (inputs[ix])
                        cellIndex0 = cellIndex0 | (1 << (dx + 1));
                }
                else {
                    if (cells[(y - 1) + ix * height].on)
                        cellIndex0 = cellIndex0 | (1 << (dx + 1));
                }
            }

            int cellIndex1 = cellIndex0 | (1 << 3);

            if (dist01(rng) < epsilon)
                cells[i].on = dist01(rng) < 0.5f;
            else
                cells[i].on = cells[i].values[cellIndex1] > cells[i].values[cellIndex0];

            int cellIndex = cells[i].on ? cellIndex1 : cellIndex0;

            float targetQ = reward + discount * std::max(cells[i].values[cellIndex1], cells[i].values[cellIndex0]);

            // Traces
            for (int j = 0; j < cells[i].traces.size(); j++) {
                if (learnEnabled)
                    cells[i].values[j] += lr * (targetQ - cells[i].values[j]) * cells[i].traces[j];

                cells[i].traces[j] *= traceDecay;
            }

            if (cells[i].on) {
                cells[i].traces[cellIndex1] = 1.0f;
                cells[i].traces[cellIndex0] = 0.0f;
            }
            else {
                cells[i].traces[cellIndex0] = 1.0f;
                cells[i].traces[cellIndex1] = 0.0f;
            }
        }
    }
}
