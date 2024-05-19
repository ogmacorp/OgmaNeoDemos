// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/hierarchy.h>
#include "aon_utils.hpp"

#include "vis/Plot.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

#if !defined(M_PI)
#define M_PI 3.141596f
#endif

using namespace aon;

#include "getopt.h"

#include "csdrScalarEncoder.hpp"

int main(int argc, char *argv[])
{
    std::string hFileName = "wavyLine.ohr";

    int numAdditionalStepsAhead = 5;
    int numInputs  = 2;

    int opt;
	while ((opt = getopt(argc, argv, "i:p:")) != -1) {  // for each option...
		switch (opt) {
		case 'i':			
			numInputs = std::stoi(optarg);
			break;        
		case 'p':
			numAdditionalStepsAhead = std::stoi(optarg);
			break;
		case '?':
			std::cerr << "valid option -i num_inputs -p numSteps!" << std::endl;
			break;
		}
	}

    // --------------------------- Create the window(s) ---------------------------

    unsigned int windowWidth = 1000;
    unsigned int windowHeight = 500;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "Wavy Test", sf::Style::Default);

    window.setVerticalSyncEnabled(false);
    //window.setFramerateLimit(60);

    int plotHeight = windowHeight / numInputs;

    vis::Plot plot[numInputs];
    for (auto i = 0; i < numInputs; ++i)
    {
        //plot[i].backgroundColor = sf::Color(64, 64, 64, 255);
        plot[i].plotXAxisTicks = false;
        plot[i].curves.resize(2 + numAdditionalStepsAhead);
        plot[i].curves[0].shadow = 0.f; // Input
        plot[i].curves[1].shadow = 0.f; // 1st step prediction
        if (numAdditionalStepsAhead)
            plot[i].curves[2].shadow = 0.f; // multi-step prediction
    }

    float minValue = -1.25f;
    float maxValue = 1.25f;

    sf::RenderTexture plotRT[numInputs];
    for (auto i = 0; i < numInputs; ++i)
    {
        plotRT[i].create(windowWidth, plotHeight);
        plotRT[i].setActive();
        plotRT[i].clear(sf::Color::White);
    }

    sf::Texture lineGradient;
    lineGradient.loadFromFile("resources/lineGradient.png");

    sf::Font tickFont;

#if defined(_WINDOWS)
    tickFont.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#elif defined(__APPLE__)
    tickFont.loadFromFile("/Library/Fonts/Courier New.ttf");
#else
    tickFont.loadFromFile("/usr/share/fonts/truetype/ttf-bitstream-vera/VeraMono.ttf");
#endif

    // --------------------------- Create the Hierarchy ---------------------------

    const int inputColumnSize = 64;
    const int eRadius = 2;
    const int dRadius = 2;
    const int historyCapacity = 64;

    set_num_threads(4);
printf("B0\n");
    Hierarchy h;
    Array<Hierarchy::IO_Desc> ioDescs(numInputs);
    for (auto i=0; i < numInputs; ++i)
        ioDescs[i] = Hierarchy::IO_Desc(Int3(1, 1, inputColumnSize), IO_Type::prediction, 4, eRadius, dRadius, historyCapacity);
printf("B1\n");
    const int numLayers = 6;    // the last layer updates its value every 2^(numLayers-1) = 32 steps
                                // each hidden layer has 4 x 4 elementsx, but we get only prediction by the 1st element
                                // What do other elements of hidden layers mean????
                                // update period of each hidden layer is fixed --> no context information here, because context should
                                // have different length over time (e.g. increasing phase of a signal)
    Array<Hierarchy::Layer_Desc> lds(numLayers);
    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int3(4, 4, 32);
        lds[i].num_dendrites_per_cell = 4;
    }
printf("B2\n");
    h.init_random(ioDescs, lds);
printf("B3\n");
    // Context analyse based on the top hidden layer in hierarchy
    // and colorize all data of the same context
    sf::Color inColors[2] = {sf::Color::Red, sf::Color::Magenta};       
    int colorIndx  = 0;

    int hStateSize = h.state_size();

    const int maxBufferSize = 300;

    bool quit = false;
    bool autoplay = true;
    bool spacePressedPrev = false;

    int index = -1;

    bool loadHierarchy = false;
    bool saveHierarchy = false;
    bool learnFlag     = true;

    // prediction index for 1-step and multi-step prediction
    int predIndice[numInputs], mPredIndice[numInputs];
    float predValues[numInputs];

    // Creat a random number generator
    std::mt19937 generator(time(nullptr));
    std::uniform_real_distribution<float> dist01(-1.0f, 1.0f);
    float noiseFactor = 0.f;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) learnFlag = false;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::L)) learnFlag = true;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::N)) noiseFactor = 0.01;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::C)) noiseFactor = 0.f;

            bool spacePressed = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);

            if (spacePressed && !spacePressedPrev)
                autoplay = !autoplay;

            spacePressedPrev = spacePressed;
        }

        if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            index++;

            if (index % 1000 == 0)
                std::cout << "Step: " << index << ", learn: " << learnFlag << ", noise: " << noiseFactor << std::endl;

            float inValues[numInputs];
//#define _FOR_BEST_CONTEXT_DEMO_
#ifdef _FOR_BEST_CONTEXT_DEMO_
            inValues[0] = std::sin(0.0125f * M_PI * index + 0.25f);
            for (auto i = 1; i < numInputs; ++i)
                inValues[i] = 0.8*std::cos(0.02 * i * M_PI * index); 
#else
            //inValues[0] = std::sin(0.0125f * M_PI * index + 0.25f) * std::sin(0.03f * M_PI * index + 1.5f) * std::sin(0.025f * M_PI * index - 0.1f);
            //inValues[0] = std::sin(0.0125f * M_PI * index + 0.25f);
            inValues[0] = std::sin(0.0125f * M_PI * index * 0.5 + 0.25f) * std::sin(0.03f * M_PI * index + 1.5f) * std::sin(0.025f * M_PI * index - 0.1f);
            for (auto i = 1; i < numInputs; ++i)
                inValues[i] = 0.8*std::cos(0.02 * i * M_PI * index) + 0.2*std::sin(0.05f * i * M_PI * index);

            // adding noises
            for (auto i = 0; i < numInputs; ++i) inValues[i] += noiseFactor*dist01(generator);
#endif
            Array<Int_Buffer_View> inputCIs(numInputs);
            Int_Buffer inBs[numInputs];
            for (auto i = 0; i < numInputs; ++i)
            {
                inBs[i] = Int_Buffer(1, simpleFloat2CSDR(inValues[i], inputColumnSize, minValue, maxValue));
                inputCIs[i] = inBs[i];
            }
printf("C0\n");            
            h.step(inputCIs, learnFlag);
printf("C1\n");
            if (numAdditionalStepsAhead > 1)
            {
                // do multiple step prediction ahead
                // 1. save the current states into buffer
                BufferWriter writer(hStateSize);
                h.write_state(writer);
printf("C2\n");
                // 2. multiple step prediction ahead
                for (int step = 1; step < numAdditionalStepsAhead; step++)
                {
                    Array<Int_Buffer_View> inputCIs_(numInputs);
                    for (auto i = 0; i < numInputs; ++i) inputCIs_[i] = h.get_prediction_cis(i);
printf("C3\n");
                    h.step(inputCIs_, false);
                }
printf("C4\n");
                // 3. get results of multistep prediction
                for (auto i = 0; i < numInputs; ++i)  mPredIndice[i] = h.get_prediction_cis(i)[0];

                // 4. copy the old states in buffer back to the hierarchy
                BufferReader reader;
                reader.buffer = &writer.buffer;                   
                h.read_state(reader);

                // end do multiple step prediction
            }
            else
            {
                for (auto i = 0; i < numInputs; ++i)  mPredIndice[i] = h.get_prediction_cis(i)[0];
            }
            
            // **********************************************
            // Analyzing the state of the top Hidden Layer
            //   1. find the input pattern (even though multiple input). It looks like fusion data
            //   2. then colorize the pattern
            // **********************************************
            // get CSDR of the top hidden layer and convert them into vector
            auto topCI = h.get_encoder(h.get_num_layers() - 1).get_hidden_cis();
            std::vector<int> thD; thD.reserve(topCI.size());
            for (auto i=0; i < topCI.size(); ++i) thD.push_back( topCI[i]);

            float cScores;
            int cMatchIndx, cMatchLen;
            std::tie(cScores, cMatchIndx, cMatchLen) = PatternAnalyse(thD, index);
            if (cMatchLen)
            {
                // pattern length is bigger than 0
                colorIndx    = !colorIndx;
            }
            
            sf::Color inColor = inColors[colorIndx];
            // **********************************************

            // Un-bin
            float anomalyScores[numInputs];
            for (auto i = 0; i < numInputs; ++i)
            {
                //predValues[i] = static_cast<float>(predIndice[i]) / static_cast<float>(inputColumnSize - 1) * (maxValue - minValue) + minValue;
                //predValues[i] = simpleCSDR2Float(predIndice[i], inputColumnSize, minValue, maxValue);
                anomalyScores[i] = 0; //(inValues[i] - predValues[i]) * (inValues[i] - predValues[i]);
            }
            // Plot target data
            window.clear();

            for (auto i = 0; i < numInputs; ++i)
            {
                vis::Point p;
                p.position.x = index;
                p.position.y = inValues[i];
                p.color = inColor;
                plot[i].curves[0].points.push_back(p);

                // Plot predicted data
                vis::Point p1;
                p1.position.x = index;
                p1.position.y = predValues[i];
                p1.color = sf::Color::Blue;
                plot[i].curves[1].points.push_back(p1);

                if (numAdditionalStepsAhead)
                {
                    //float mPredValue = static_cast<float>(mPredIndice[i]) / static_cast<float>(inputColumnSize - 1) * (maxValue - minValue) + minValue;
                    float mPredValue = simpleCSDR2Float(mPredIndice[i], inputColumnSize, minValue, maxValue);
                    vis::Point p2;
                    p2.position.x = index;
                    p2.position.y = mPredValue;
                    p2.color = sf::Color::Green;
                    plot[i].curves[2].points.push_back(p2);
                }

                if (plot[i].curves[0].points.size() > maxBufferSize) {
                    plot[i].curves[0].points.erase(plot[i].curves[0].points.begin());
                    int firstIndex = 0;
                    for (std::vector<vis::Point>::iterator it = plot[i].curves[0].points.begin(); it != plot[i].curves[0].points.end(); it++, firstIndex++)
                        (*it).position.x = (float)firstIndex;

                    plot[i].curves[1].points.erase(plot[i].curves[1].points.begin());
                    firstIndex = 0;
                    for (std::vector<vis::Point>::iterator it = plot[i].curves[1].points.begin(); it != plot[i].curves[1].points.end(); it++, firstIndex++)
                        (*it).position.x = (float)firstIndex;

                    if (numAdditionalStepsAhead)
                    {
                        plot[i].curves[2].points.erase(plot[i].curves[2].points.begin());
                        firstIndex = 0;
                        for (std::vector<vis::Point>::iterator it = plot[i].curves[2].points.begin(); it != plot[i].curves[2].points.end(); it++, firstIndex++)
                            (*it).position.x = (float)firstIndex;
                    }
                }

                plot[i].draw(plotRT[i], lineGradient, tickFont, 0.5f,
                    sf::Vector2f(0.0f, plot[i].curves[0].points.size()),
                    sf::Vector2f(minValue, maxValue), sf::Vector2f(48.0f, 48.0f),
                    sf::Vector2f(plot[i].curves[0].points.size() / 10.0f, (maxValue - minValue) / 10.0f),
                    2.0f, 4.0f, 2.0f, 6.0f, 2.0f, 4);

                plotRT[i].display();

                sf::Sprite plotSprite;
                plotSprite.setPosition(0,i * plotHeight);
                plotSprite.setTexture(plotRT[i].getTexture());

                window.draw(plotSprite);
            }

            //float mStateValue = static_cast<float>(mPredIndice[i]) / static_cast<float>(inputColumnSize - 1) * (maxValue - minValue) + minValue;
            //vis::Point p2;
            //p2.position.x = index;
            //p2.position.y = mPredValue;
            //p2.color = sf::Color::Green;
            //plot[i].curves[2].points.push_back(p2);


            window.display();
        }
    } while (!quit);

    return 0;
}
