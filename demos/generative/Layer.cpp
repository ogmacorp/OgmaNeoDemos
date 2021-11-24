#include "Layer.h"

#include <omp.h>

void Layer::initRandom(
    int inputWidth,
    int inputHeight,
    int inputDepth,
    int outputDepth,
    int columnSize,
    int stride,
    int radius
) {
    this->inputWidth = inputWidth;
    this->inputHeight = inputHeight;
    this->inputDepth = inputDepth;
    this->outputDepth = outputDepth;
    this->columnSize = columnSize;
    this->stride = stride;
    this->radius = radius;

    outputs = std::vector<float>((inputWidth / stride) * (inputHeight / stride) * outputDepth * columnSize, 0.0f);
    outputActivations = outputs;

    int diam = radius * 2 + 1;
    int area = diam * diam;
    int numWeightsPerCell = area * inputDepth;

    weights.resize(outputs.size() * numWeightsPerCell);

    std::uniform_real_distribution<float> distWeight(0.0f, 1.0f);

    for (int i = 0; i < weights.size(); i++)
        weights[i] = distWeight(rng);

    reconstruction = std::vector<float>(inputWidth * inputHeight * inputDepth, 0.0f);
    counts = reconstruction;
}

void Layer::step(
    const std::vector<float> &inputs,
    float lr
) {
    // Go through input and accumulate in output
    std::fill(outputActivations.begin(), outputActivations.end(), 0.0f);

    int outputWidth = getOutputWidth();
    int outputHeight = getOutputHeight();
    int outputDepthColumn = outputDepth * columnSize;

    int diam = radius * 2 + 1;

    //#pragma omp parallel for
    for (int ii = 0; ii < inputs.size(); ii++) {
        int ix = ii / (inputDepth * inputHeight);
        int iy = (ii / inputDepth) % inputHeight;
        int id = ii % inputDepth;

        if (inputs[ii] == 0.0f)
            continue;

        int cox = ix / stride;
        int coy = iy / stride;

        int projRadius = radius / stride + 1;
        
        for (int dx = -projRadius; dx <= projRadius; dx++)
            for (int dy = -projRadius; dy <= projRadius; dy++) {
                int ox = cox + dx;
                int oy = coy + dy;

                if (ox < 0 || ox >= outputWidth || oy < 0 || oy >= outputHeight)
                    continue;

                int cix = ox * stride;
                int ciy = oy * stride;

                if (ix >= cix - radius && ix <= cix + radius && iy >= ciy - radius && iy <= ciy + radius) {
                    for (int od = 0; od < outputDepthColumn; od++) {
                        int oi = od + outputDepthColumn * (oy + outputHeight * ox);

                        int wi = id + inputDepth * (ix - cix + radius + diam * (iy - ciy + radius + diam * oi));

                        //#pragma omp atomic
                        outputActivations[oi] += weights[wi] * inputs[ii];
                    }
                }
            }
    }
    
    std::fill(reconstruction.begin(), reconstruction.end(), 0.0f);
    std::fill(counts.begin(), counts.end(), 0.0f);

    // Inhibit
    //#pragma omp parallel for
    for (int oi = 0; oi < outputs.size(); oi++) {
        int ox = oi / (outputDepthColumn * outputHeight);
        int oy = (oi / outputDepthColumn) % outputHeight;
        int od = oi % outputDepthColumn;

        // If is highest in column
        int coli = od / columnSize;

        bool hasGreater = false;

        for (int ood = coli * columnSize; ood < (coli + 1) * columnSize; ood++) {
            int ooi = ood + outputDepthColumn * (oy + outputHeight * ox);

            if (outputActivations[ooi] > outputActivations[oi]) {
                hasGreater = true;

                break;
            }
        }
        
        outputs[oi] = hasGreater ? 0.0f : 1.0f;

        if (!hasGreater && lr != 0.0f) {
            // Reconstruct
            for (int dx = -radius; dx <= radius; dx++)
                for (int dy = -radius; dy <= radius; dy++) {
                    int ix = ox * stride + dx;
                    int iy = oy * stride + dy;

                    if (ix < 0 || ix >= inputWidth || iy < 0 || iy >= inputHeight)
                        continue;

                    for (int id = 0; id < inputDepth; id++) {
                        int ii = id + inputDepth * (iy + inputHeight * ix);

                        int wi = id + inputDepth * (dx + radius + diam * (dy + radius + diam * oi));

                        float strength = 1.0f - (std::abs(dx) + std::abs(dy)) / static_cast<float>(2 * radius + 1);
                            
                        //#pragma omp atomic
                        reconstruction[ii] += weights[wi] * strength;

                        //#pragma omp atomic
                        counts[ii] += strength;
                    }
                }
        }
    }

    // Rescale reconstruction
    for (int ii = 0; ii < reconstruction.size(); ii++)
        reconstruction[ii] /= std::max(0.0001f, counts[ii]);

    // Learn
    if (lr != 0.0f) {
        // Reconstruct
        //#pragma omp parallel for
        for (int oi = 0; oi < outputs.size(); oi++) {
            int ox = oi / (outputDepthColumn * outputHeight);
            int oy = (oi / outputDepthColumn) % outputHeight;
            int od = oi % outputDepthColumn;

            if (outputs[oi] != 0.0f) {
                // Reconstruct
                for (int dx = -radius; dx <= radius; dx++)
                    for (int dy = -radius; dy <= radius; dy++) {
                        int ix = ox * stride + dx;
                        int iy = oy * stride + dy;

                        if (ix < 0 || ix >= inputWidth || iy < 0 || iy >= inputHeight)
                            continue;

                        for (int id = 0; id < inputDepth; id++) {
                            int ii = id + inputDepth * (iy + inputHeight * ix);

                            int wi = id + inputDepth * (dx + radius + diam * (dy + radius + diam * oi));
                                
                            weights[wi] += lr * (inputs[ii] - reconstruction[ii]);
                        }
                    }
            }
        }
    }
}

void Layer::reconstruct(
    const std::vector<float> &reconOutputs
) {
    int outputWidth = getOutputWidth();
    int outputHeight = getOutputHeight();
    int outputDepthColumn = outputDepth * columnSize;

    std::fill(reconstruction.begin(), reconstruction.end(), 0.0f);
    std::fill(counts.begin(), counts.end(), 0.0f);

    int diam = radius * 2 + 1;

    //#pragma omp parallel for
    for (int oi = 0; oi < outputs.size(); oi++) {
        int ox = oi / (outputDepthColumn * outputHeight);
        int oy = (oi / outputDepthColumn) % outputHeight;
        int od = oi % outputDepthColumn;

        // If is highest in column
        int coli = od / columnSize;

        bool hasGreater = false;

        for (int ood = coli * columnSize; ood < (coli + 1) * columnSize; ood++) {
            int ooi = ood + outputDepthColumn * (oy + outputHeight * ox);

            if (reconOutputs[ooi] > reconOutputs[oi]) {
                hasGreater = true;

                break;
            }
        }

        if (!hasGreater) {
            // Reconstruct
            for (int dx = -radius; dx <= radius; dx++)
                for (int dy = -radius; dy <= radius; dy++) {
                    int ix = ox * stride + dx;
                    int iy = oy * stride + dy;

                    if (ix < 0 || ix >= inputWidth || iy < 0 || iy >= inputHeight)
                        continue;

                    for (int id = 0; id < inputDepth; id++) {
                        int ii = id + inputDepth * (iy + inputHeight * ix);

                        int wi = id + inputDepth * (dx + radius + diam * (dy + radius + diam * oi));
                            
                        float strength = 1.0f - (std::abs(dx) + std::abs(dy)) / static_cast<float>(2 * radius + 1);

                        //#pragma omp atomic
                        reconstruction[ii] += weights[wi] * strength;

                        //#pragma omp atomic
                        counts[ii] += strength;
                    }
                }
        }
    }

    // Rescale reconstruction
    for (int ii = 0; ii < reconstruction.size(); ii++)
        reconstruction[ii] /= std::max(0.0001f, counts[ii]);
}
