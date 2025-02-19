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

class CustomerStreamReader : public aon::Stream_Reader
{
public:
  std::ifstream ins;

  void read( void* data, long len )
  {
    ins.read(static_cast<char*>(data), len);
  }
};

class CustomerStreamWriter : public aon::Stream_Writer
{
public:
  std::ofstream outs;

  void write( const void* data, long len )
  {
    outs.write(static_cast<const char*>(data), len);
  }
};

class BufferReader : public aon::Stream_Reader
{
public:
  int start;
  const std::vector<unsigned char>* buffer;

  BufferReader() : start(0), buffer(nullptr)
  {}

  void read( void* data, long len )
  {
    for (int i = 0; i < len; i++)
      static_cast<unsigned char*>(data)[i] = (*buffer)[start + i];

    start += len;
  }
};

class BufferWriter : public aon::Stream_Writer
{
public:
  int start;
  std::vector<unsigned char> buffer;

  BufferWriter(int size) : start(0)
  {
    buffer.resize(size);
  };

  void write( const void* data, long len )
  {
    assert(buffer.size() >= start + len);

    for (int i = 0; i < len; i++)
      buffer[start + i] = static_cast<const unsigned char*>(data)[i];

    start += len;
  }
};

int simpleFloat2CSDR(float x, int cells_per_column, float minVal = 0.f, float maxVal = 1.f)
{
    return static_cast<int>((x - minVal) / (maxVal - minVal) * (cells_per_column - 1) + 0.5f);
};

// decode single-point CSDR into float
float simpleCSDR2Float(int Index, int cells_per_column, float minVal = 0.f, float maxVal = 1.f)
{
    return static_cast<float>(Index) / static_cast<float>(cells_per_column - 1) * (maxVal - minVal) + minVal;
};

int main(int argc, char *argv[])
{

    std::string hFileName = "wavyLine.ohr";

    int numAdditionalStepsAhead = 5;
    int numInputs  = 2;
    bool loadHierarchy = false;

    int opt;
	while ((opt = getopt(argc, argv, "i:p:h")) != -1) {  // for each option...
		switch (opt) {
		case 'i':
			numInputs = std::stoi(optarg);
			break;
		case 'p':
			numAdditionalStepsAhead = std::stoi(optarg);
			break;
        case 'h':
            loadHierarchy = true;
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
        plot[i].curves.resize(2 + (numAdditionalStepsAhead > 1 ? 2 : 0));
        plot[i].curves[0].shadow = 0.f; // Input
        plot[i].curves[1].shadow = 0.f; // 1st step prediction
        if (numAdditionalStepsAhead > 1)
        {
            plot[i].curves[2].shadow = 0.f; // plot the last-step prediction

            plot[i].curves[3].shadow = 0.f; // plot only ahead predicted values from 1 to numAdditionalStepsAhead
            plot[i].curves[3].type   = 2; // positive value for circle display with radius = type, 0: line
        }
    }

    float minValue = -1.25f;
    float maxValue = 1.25f;

    sf::RenderTexture plotRT[numInputs];
    for (auto i = 0; i < numInputs; ++i)
    {
        plotRT[i].create(windowWidth, plotHeight, false);
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
    tickFont.loadFromFile("/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf");
#endif

    set_num_threads(8);

#define USE_SIMPLE_FLOAT_ENCODER_
    // for encoding/decoding scalar input
#ifdef USE_SIMPLE_FLOAT_ENCODER_
    const int inputColumnSize = 64;
#else
    const int inputColumnSize = 16;
#endif

    // --------------------------- Create the Hierarchy ---------------------------
    const int eRadius                   = 2;    // encoder radius
    const int dRadius                   = 2;    // decoder radius
    const int num_dendrites_per_cell    = 4;
    const int history_capacity          = 256;

    Hierarchy h;
    Array<Hierarchy::IO_Desc> ioDescs(numInputs);
#ifdef USE_SIMPLE_FLOAT_ENCODER_
    for (auto i=0; i < numInputs; ++i)
        ioDescs[i] = Hierarchy::IO_Desc(Int3(1, 1, inputColumnSize), IO_Type::prediction, num_dendrites_per_cell, eRadius, dRadius);

    const int numLayers = 2;    // the last layer updates its value every 2^(numLayers-1) = 32 steps
                                // each hidden layer has 4 x 4 elementsx, but we get only prediction by the 1st element
                                // What do other elements of hidden layers mean????
                                // update period of each hidden layer is fixed --> no context information here, because context should
                                // have different length over time (e.g. increasing phase of a signal)

#else
    for (auto i=0; i < numInputs; ++i)
        ioDescs[i] = Hierarchy::IO_Desc(Int3(1, 2, inputColumnSize), IO_Type::prediction, num_dendrites_per_cell, eRadius, dRadius);

    const int numLayers = 2;
#endif


    const int ticks_per_update = 2; // number of ticks a layer takes to update (relative to previous layer)
    const int temporal_horizon = 2; // temporal distance into the past addressed by the layer. should be greater than or equal to ticks_per_update

    Array<Hierarchy::Layer_Desc> lds(numLayers);
    for (int i = 0; i < lds.size(); i++) {
#ifdef USE_SIMPLE_FLOAT_ENCODER_
        lds[i].hidden_size              = Int3(5, 5, 16);
#else
        lds[i].hidden_size              = Int3(5, 5, 16);
#endif
        lds[i].num_dendrites_per_cell   = num_dendrites_per_cell;
        lds[i].temporal_size = 16;
        //lds[i].ticks_per_update         = ticks_per_update;
        //lds[i].temporal_horizon         = temporal_horizon;
    }

    bool learnFlag     = true;

    if (loadHierarchy)
    {
        std::cout << "load hierarchy file" << std::endl;
        CustomerStreamReader reader;
        reader.ins.open(hFileName.c_str(), std::ios::binary);
        h.read(reader);
        learnFlag = false;

        //h.clear_state();
    }
    else
    {
        std::cout << "randomly init hierarchy" << std::endl;
        h.init_random(ioDescs, lds);
    }

    std::cout << "...finished" << std::endl;

    // Context analyse based on the top hidden layer in hierarchy
    // and colorize all data of the same context
    sf::Color inColors[2] = {sf::Color::Red, sf::Color::Magenta};
    int colorIndx  = 0;

    int hStateSize = h.state_size();

    const int maxBufferSize = 300;

    bool quit = false;
    bool autoplay = true;
    bool spacePressedPrev = false;
    bool sPressedPrev = false;

    int index = -1;

    // prediction index for 1-step and multi-step prediction
    int predIndice[numInputs], mPredIndice[numInputs];
    float predValues[numInputs];

    // Creat a random number generator
    std::mt19937 generator(time(nullptr));
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> dist11(-1.0f, 1.0f);
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
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))   quit = true;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::P))        learnFlag = false;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::L))        learnFlag = true;

            if (sf::Keyboard::isKeyPressed(sf::Keyboard::N)) noiseFactor = 0.01;
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::C)) noiseFactor = 0.f;

            bool spacePressed = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);

            if (spacePressed && !spacePressedPrev)
                autoplay = !autoplay;

            spacePressedPrev = spacePressed;
        }

        if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            index++;

            if (dist01(generator) < 0.001f) {
                std::uniform_int_distribution<int> indexDist(0, 1000);

                index = indexDist(generator);

                std::cout << "Random jump" << std::endl;
            }

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
            for (auto i = 0; i < numInputs; ++i) inValues[i] += noiseFactor*dist11(generator);
#endif
            Array<Int_Buffer_View> inputCIs(numInputs);
            Int_Buffer inBs[numInputs];
            for (auto i = 0; i < numInputs; ++i)
            {
#ifdef USE_SIMPLE_FLOAT_ENCODER_
                inBs[i] = Int_Buffer(1, simpleFloat2CSDR(inValues[i], inputColumnSize, minValue, maxValue));
#else
                Int_Buffer sensorCIs(2, 0);
                auto sData = Unorm8ToCSDR(inValues[i], minValue, maxValue);
                sensorCIs[0] = sData[0];
                sensorCIs[1] = sData[1];
                inBs[i] = sensorCIs;
#endif
                inputCIs[i] = inBs[i];
            }

            h.step(inputCIs, learnFlag);

            for (int i = 0; i < h.get_encoder(0).get_hidden_cis().size(); i++)
                std::cout << h.get_encoder(0).get_hidden_cis()[i] << " ";

            std::cout << std::endl;

            // prediction index for 1-step prediction
            for (auto i = 0; i < numInputs; ++i)
                predIndice[i] = h.get_prediction_cis(i)[0];

            int predIndexData[numInputs][numAdditionalStepsAhead];

            if (numAdditionalStepsAhead > 1)
            {
                // do multiple step prediction ahead
                // 1. save the current states into buffer
                BufferWriter writer(hStateSize);
                h.write_state(writer);

                // 2. multiple step prediction ahead
                int step = 0;
                for (; step < numAdditionalStepsAhead-1; step++)
                {
                    Array<Int_Buffer_View> inputCIs_(numInputs);
                    for (auto i = 0; i < numInputs; ++i)
                    {
                        inputCIs_[i]           = h.get_prediction_cis(i);
                        predIndexData[i][step] = inputCIs_[i][0];
                    }

                    h.step(inputCIs_, false);
                }

                // 3. get results of multistep prediction
                for (auto i = 0; i < numInputs; ++i)
                {
                    mPredIndice[i]         = h.get_prediction_cis(i)[0];
                    predIndexData[i][step] = mPredIndice[i];
                }

                // 4. copy the old states in buffer back to the hierarchy
                BufferReader reader;
                reader.buffer = &writer.buffer;
                h.read_state(reader);

                // end do multiple step prediction
            }

            bool sPressed = sf::Keyboard::isKeyPressed(sf::Keyboard::S);
            if (sPressed && !sPressedPrev && learnFlag)
            {
                std::cout << "Save hierarchy file" << std::endl;
                CustomerStreamWriter writer;
                writer.outs.open(hFileName.c_str(), std::ios::out | std::ios::binary);
                h.write(writer);
            }
            sPressedPrev =  sPressed;


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
                // 1-step
                float mPredValue = simpleCSDR2Float(mPredIndice[i], inputColumnSize, minValue, maxValue);
                vis::Point p1;
                p1.position.x = index;
                p1.position.y = mPredValue;
                p1.color = sf::Color::Green;
                plot[i].curves[1].points.push_back(p1);

                if (numAdditionalStepsAhead)
                {
                    //float mPredValue = static_cast<float>(mPredIndice[i]) / static_cast<float>(inputColumnSize - 1) * (maxValue - minValue) + minValue;
                    float mPredValue = simpleCSDR2Float(mPredIndice[i], inputColumnSize, minValue, maxValue);
                    vis::Point p2;
                    p2.position.x = index;
                    p2.position.y = mPredValue;
                    p2.color = sf::Color::Blue;
                    plot[i].curves[2].points.push_back(p2);

                    auto myindex = plot[i].curves[0].points.size() > maxBufferSize ? (maxBufferSize) : (index+1);
                    plot[i].curves[3].points.clear();
                    for (auto j = 0; j < numAdditionalStepsAhead; ++j)
                    {
                        float mPredValue = simpleCSDR2Float(predIndexData[i][j], inputColumnSize, minValue, maxValue);
                        vis::Point p3;
                        p3.position.x = myindex+j;
                        p3.position.y = mPredValue;
                        p3.color = sf::Color::Cyan;
                        plot[i].curves[3].points.push_back(p3);
                    }
                }

                if (plot[i].curves[0].points.size() > maxBufferSize )
                {
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

                // multiple prediction values (numAdditionalStepsAhead points) will be appended in the input curve (with index = 0)
                auto maxX = plot[i].curves[0].points.size() + numAdditionalStepsAhead;

                plot[i].draw(plotRT[i], lineGradient, tickFont, 0.5f,
                    sf::Vector2f(0.0f, maxX), sf::Vector2f(minValue, maxValue), sf::Vector2f(48.0f, 48.0f),
                    sf::Vector2f(maxX / 10.0f, (maxValue - minValue) / 10.0f),
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
