// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Settings.h"

#include <ctime>
#include <iostream>

#include <neo/Predictor.h>

using namespace ogmaneo;

int main() {
	std::mt19937 generator(std::time(nullptr));

    ComputeSystem cs;
    cs.create(ComputeSystem::_gpu);

    ComputeProgram prog;
    prog.loadMainKernel(cs);

    // Input image
    cl::Image2D inputImage = cl::Image2D(cs.getContext(), CL_MEM_READ_WRITE, cl::ImageFormat(CL_R, CL_FLOAT), 4, 4);

    // Hierarchy structure
    std::vector<FeatureHierarchy::LayerDesc> layerDescs(3);
    std::vector<Predictor::PredLayerDesc> pLayerDescs(3);

    layerDescs[0]._size = { 16, 16 };
    layerDescs[1]._size = { 16, 16 };
    layerDescs[2]._size = { 16, 16 };

    for (int l = 0; l < layerDescs.size(); l++) {
        layerDescs[l]._recurrentRadius = 6;
        layerDescs[l]._spActiveRatio = 0.04f;
        layerDescs[l]._spBiasAlpha = 0.01f;

        pLayerDescs[l]._alpha = 0.08f;
        pLayerDescs[l]._beta = 0.16f;
    }

    // Predictive hierarchy
    Predictor ph;

    ph.createRandom(cs, prog, { 4, 4 }, pLayerDescs, layerDescs, { -0.01f, 0.01f }, generator);

	std::uniform_int_distribution<int> item_dist(0, 9);

    std::vector<float> digitVec(16, 0.0f);

    bool recalled = true;
    int recallSteps = 0;

    std::vector<int> digits(10);
    assert(digits.size() <= digitVec.size());

    int numSequences = 100;
    int sequenceCount = 1;
    std::cout << "Testing recall for " << numSequences << " sequences" << std::endl;

    while(numSequences > 0) {
        if (recalled) {
            std::cout << "Sequence #: " << sequenceCount << std::endl;

            // Create 10 random digits [0-9]
            std::cout << "Input:      ";
            for (size_t show_iter = 0; show_iter < digits.size(); show_iter++) {
                digits[show_iter] = item_dist(generator);
                std::cout << digits[show_iter] << " ";
            }
            std::cout << std::endl;

            recalled = false;
        }

        // Show each digit in the sequence to the network as a bit vector
        for (size_t show_iter = 0; show_iter < digits.size(); show_iter++) {
			for (int i = 0; i < 16; i++)
				digitVec[i] = 0.0f;

			digitVec[digits[show_iter]] = 1.0f;

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 4, 4, 1 }, 0, 0, digitVec.data());
			ph.simStep(cs, inputImage, inputImage, generator);
		}

		std::vector<float> prediction(16, 0.0f);

        int recallCount = 0;

        // Query the hierarchy and count how many predicted digits match the input sequence
        std::cout << "Prediction: ";
        for (size_t recall_iter = 0; recall_iter < digits.size(); recall_iter++) {
			cs.getQueue().enqueueReadImage(ph.getPrediction(), CL_TRUE, { 0, 0, 0 }, { 4, 4, 1 }, 0, 0, prediction.data());

            // Determine the highest prediction
            float highestItem = -1.0f;
            int predictedDigit = -1;
			for (size_t i = 0; i < digits.size(); i++) {
                if (prediction[i] > highestItem)
                {
                    highestItem = prediction[i];
                    predictedDigit = static_cast<int>(i);
                }
			}
            std::cout << predictedDigit << " ";

            // Does the predicted digit match the input digit?
            if (predictedDigit == digits[recall_iter])
                recallCount++;

            // Present the next digit
			for (int i = 0; i < 16; i++)
				digitVec[i] = 0.0f;

			digitVec[digits[recall_iter]] = 1.0f;

			cs.getQueue().enqueueWriteImage(inputImage, CL_TRUE, { 0, 0, 0 }, { 4, 4, 1 }, 0, 0, digitVec.data());
			ph.simStep(cs, inputImage, inputImage, generator);
		}
        std::cout << std::endl;

        recallSteps++;

        // If the input sequence matches the predicted sequence, create a new random sequence of digits
        if (recallCount == 10)
        {
            std::cout << "Sequence recalled in " << recallSteps << " steps, resetting sequence" << std::endl << std::endl;
            recalled = true;
            recallSteps = 0;

            sequenceCount++;
            numSequences--;
        }
    }

	return 0;
}
