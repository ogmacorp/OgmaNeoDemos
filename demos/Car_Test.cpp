// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/Hierarchy.h>
#include <aogmaneo/Helpers.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <cmath>

#include "getopt.h"

#include <iostream>
#include "vis/Plot.hpp"
#include <vis/guiControl_sfml.hpp>


#if !defined(M_PI)
#define M_PI 3.141596f
#endif

using namespace aon;

std::vector<int> Unorm8ToCSDR(float x)
{
    // get value in the interval [0, 1]
    x = std::min(1.0f, std::max(0.0f, x));
    // scaling it into [0, 255] and consider only the first byte
    int i = int(x * 255.0 + 0.5) & 0xff;
    // split the first byte into 2 parts, each has 4 bit or max value of pow(2,4) = 16
    std::vector<int> res = {int(i & 0x0f), int((i & 0xf0) >> 4) };
    return res;
};

// Reverse transform of IEEEToCSDR
float CSDRToUnorm8(std::vector<int> csdr)
{
    return (csdr[0] | (csdr[1] << 4)) / 255.0;
};

class CustomStreamReader : public aon::StreamReader {
public:
    std::ifstream ins;

    void read(
        void* data,
        int len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::StreamWriter {
public:
    std::ofstream outs;

    void write(
        const void* data,
        int len
    ) override {
        outs.write(static_cast<const char*>(data), len);
    }
};

std::vector<float> auto_label_data(std::vector<float> ts, std::vector<float> vs, float thresh=0.02, float label_decay=0.3, int window_size=16)
{
    int window_size2 = window_size/2;

    // Label
    float avg_accel = 0.0;
    int length = vs.size();

    std::vector<float> labels(length,0);

    for (auto i= 0; i < length; ++i)
    {
        float avg = 0.0;
        int count = 0;
        // Windowed average
        for (auto d = -window_size2; d < window_size2; ++d)
        {
            int w = i + d;

            if (w > 0 && w < length)
            {
                float dt = ts[w] - ts[w - 1];
                avg += (vs[w] - vs[w - 1]) / std::max(0.0001f, dt);
                count++;
            }
        }
        avg /= std::max(1, count);

        // Apply soft labels
        if (abs(avg) >= thresh)
        {
            for (auto d = -window_size2; d < window_size2; ++d)
            {
                int w = i + d;

                if (w >= 0 && w < length)
                {
                    if (avg > 0.0)
                        labels[w] = labels[w] + label_decay * (1 - labels[w]);
                    else
                        labels[w] = labels[w] + label_decay * (-1 - labels[w]);
                }
            }
        }
    }

    // Threshold into discrete labels
    for (auto &l : labels)
    {
        if      (l > thresh)    l = 1;
        else if (l < -thresh)   l = -1;
        else                    l = 0;
    }

    return labels;
};

int main(int argc, char *argv[]) 
{
    bool loadHierarchy  = true;
	bool learnFlag      = true;

    int opt;
	while ((opt = getopt(argc, argv, "l:h:")) != -1) {  // for each option...
		switch (opt) {
		case 'l':			
			learnFlag = std::stoi(optarg);
			break;        
		case 'h':
			loadHierarchy = std::stoi(optarg);
			break;
		case '?':
			std::cerr << "valid option -l 1 or -h 0!" << std::endl;
			break;
		}
	}

    std::string hFileName = "Car_Test.ohr";

#ifdef WORSE_SENSOR_
    std::string fname("/home/binh/experiments/sensors/videoMagnify/bin/data_good_label_bad.txt");

    std::cout<< "file: " << fname << "!" << std::endl;

    std::ifstream file(fname.c_str(), std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        std::cerr << "Could not open " << fname << "!" << std::endl;
        return 1;
    }
 
    std::vector<float> goodD, sensorData;
    std::vector<int> labels;
    std::string tab = std::string("") + '\t';
    while (file.peek() != EOF)
    {
        std::string str;
        getline( file, str);
        // get data
        std::vector<std::string> v;
        splitString(str, tab, v);
	    goodD.push_back(std::stof(v[0]));
        labels.push_back(std::stoi(v[1]));
        sensorData.push_back(std::stof(v[2]));
    }
    std::vector<int> timeStamp(sensorData.size());
    std::iota(timeStamp.begin(), timeStamp.end(),0);
#else
    std::string fname("training_camdataDetrend.txt");

    std::cout<< "file: " << fname << "!" << std::endl;

    std::ifstream file(fname.c_str(), std::ios::binary | std::ios::in);
    if (!file.is_open()) {
        std::cerr << "Could not open " << fname << "!" << std::endl;
        return 1;
    }
 
    std::vector<float> sensorData;
    std::vector<int> labels, timeStamp;
    std::string tab = std::string("") + '\t';
    while (file.peek() != EOF)
    {
        std::string str;
        getline( file, str);
        // get data
        std::vector<std::string> v;
        splitString(str, tab, v);
	    timeStamp.push_back(std::stoi(v[0]));
        sensorData.push_back(std::stof(v[1]));
        labels.push_back(std::stoi(v[2]));
    }
    //const auto [min, max] = std::minmax_element(sensorData.begin(), sensorData.end());
    //float delta = 1./(*max -*min);
    //for (auto &s : sensorData) s *= (s-*min)*delta;
#endif

    // --------------------------- Create the window(s) ---------------------------
    sf::RenderWindow renderWindow;
	int winW = 1000, winH=800, winMainW=1800;
	int numPlots = 2;
 
	int subWinH = static_cast<int>(winH / numPlots);
	int plotHeight = subWinH - 5;
    int plotWidth = winMainW -5;
    const int maxBufferSize = 500;

    sf::Vector2i screenDimensions(winMainW, winH);

	renderWindow.create(sf::VideoMode(winMainW, winH), "Car_Test Classification", sf::Style::Default);
	renderWindow.setFramerateLimit(60);

	guiControl guiC(screenDimensions);

    vis::Plot plot(sf::Vector2f(0,0), sf::Vector2i(plotWidth, plotHeight), 2, "index", "input/prediction", maxBufferSize);
	vis::Plot plotClass(sf::Vector2f(0,subWinH), sf::Vector2i(plotWidth, plotHeight), 2, "index", "pred/GT Class", maxBufferSize);

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
    // we encode a scalar input as 2 representative value in [0,16] for better sensisitive
    // so that we have 2 columns, each has 16 cells
    const int csdr_width  = 1;
    const int csdr_height = 2;
    const int csdr_num_columns = csdr_width * csdr_height;
    const int csdr_num_cells_per_column = 16;

    setNumThreads(4);

    Hierarchy h;

    Array<Hierarchy::LayerDesc> lds(5);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hiddenSize = Int3(4, 4, csdr_num_cells_per_column);
    }
   
    // here we use the measuring data in 1st input     --> InputType = prediction
	//             the labels data in 2nd input        --> InputType = prediction
    Array<Hierarchy::IODesc> ioDescs(2);

    ioDescs[0] = Hierarchy::IODesc(Int3(csdr_width, csdr_height, csdr_num_cells_per_column), IOType::prediction, 4, 2, 2);

    const int label_width = 1;
    const int label_height = 1;
    const int label_num_cells_per_column = 3;
    ioDescs[1] = Hierarchy::IODesc(Int3(label_width, label_height, label_num_cells_per_column), IOType::prediction, 2, 2, 2);

	if (loadHierarchy)
	{
		CustomStreamReader reader;
		reader.ins.open(hFileName.c_str(), std::ios::binary);
		h.read(reader);
		learnFlag = false;
	}
	else
	{
		h.initRandom(ioDescs, lds);
	}

    float train_test_split = 0.75;

    int train_set_size = int(timeStamp.size() * train_test_split);
    int test_set_size = timeStamp.size() - train_set_size;

    // Train
    const float avg_v_decay = 0.1;
    const float avg_v_decay2 = 0.01;
    const float variance_decay = 0.01;
    const float unorm_scale = 0.6;

    float avg_v = 0.0;
    float avg_v2 = 0.0;
    float variance = 1.0;
    int train_episodes = 100;
    if (learnFlag)
    {
        for (auto ep = 0; ep < train_episodes; ++ep)
        {
            std::cout << "Episode " << ep << std::endl;

            for (auto i2 = 0; i2 < train_set_size; ++i2)
            {
                // Encode sensor
                avg_v    += avg_v_decay  * (sensorData[i2] - avg_v);
                avg_v2   += avg_v_decay2 * (sensorData[i2] - avg_v2);
                variance += variance_decay * ((avg_v - avg_v2) * (avg_v - avg_v2) - variance);

                Array<const IntBuffer*> inputCIs(ioDescs.size());
                IntBuffer sensorCIs(2, 0); 
                std::vector<int> sData = Unorm8ToCSDR((avg_v - avg_v2) / std::max(0.0001f, std::sqrt(variance)) * unorm_scale * 0.5 + 0.5);
                sensorCIs[0] = sData[0];
                sensorCIs[1] = sData[1];
                inputCIs[0] = &sensorCIs;
                IntBuffer labelCI(1, int(labels[i2] + 1));
                inputCIs[1] = &labelCI;

                h.step(inputCIs, true);
            }
        }

        CustomStreamWriter writer;
        writer.outs.open(hFileName.c_str(), std::ios::out | std::ios::binary);
        h.write(writer);
    }

    std::cout << "do inference" << std::endl;
    // inference mode
    int i2 = train_set_size;
    float accuracy = 0.0;
    bool quit = false;
    bool autoplay = true;

    int index = -1;

    do
    {
        guiC.eventHandling(renderWindow, quit);
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))		quit     = true;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))		autoplay = false;
		if (sf::Keyboard::isKeyPressed(sf::Keyboard::C))			autoplay = true;

        if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right))
        {
            float value = sensorData[i2];
            int   label = labels[i2];
            std::cout << "indx: " << i2 << ", val: " << value << ", label: " << label << std::endl;
            // Encode sensor
            avg_v    += avg_v_decay  * (value - avg_v);
            avg_v2   += avg_v_decay2 * (value - avg_v2);
            variance += variance_decay * ((avg_v - avg_v2) * (avg_v - avg_v2) - variance);

            float value1 = (avg_v - avg_v2) / std::max(0.0001f, std::sqrt(variance)) * unorm_scale * 0.5 + 0.5;

            Array<const IntBuffer*> inputCIs(ioDescs.size());
            IntBuffer sensorCIs(2, 0); 
            std::vector<int> sData = Unorm8ToCSDR(value1);
            sensorCIs[0] = sData[0];
            sensorCIs[1] = sData[1];
            inputCIs[0] = &sensorCIs;
            inputCIs[1] = &h.getPredictionCIs(1);

            h.step(inputCIs, false);
            
            std::vector<int> csdr = {h.getPredictionCIs(0)[0], h.getPredictionCIs(0)[1]};
            float pred_value = CSDRToUnorm8(csdr);

            int pred_label = h.getPredictionCIs(1)[0] - 1;                    
            if (pred_label == labels[i2])  accuracy += 1.0;
                    
            renderWindow.clear();

            // plot target data
			vis::Point p;
            p._position.x = index;
			p._position.y = value1;
			p._color = sf::Color::Red;
			plot.updateBuffer(0, p);

			// plot predicted data
			vis::Point p1;
			p1._position.y = pred_value;
			p1._color = sf::Color::Blue;
			plot.updateBuffer(1, p1);
			plot.draw(renderWindow, sf::Vector2f(0, 3), 5);

            vis::Point pC;
            pC._position.x = index;
			pC._position.y = label;
			pC._color = sf::Color::Red;
			plotClass.updateBuffer(0, pC);

			// plot predicted data
			vis::Point pC1;
			pC1._position.y = pred_label;
			pC1._color = sf::Color::Blue;
			plotClass.updateBuffer(1, pC1);
			plotClass.draw(renderWindow, sf::Vector2f(-1.5, 1.5), 5);

            renderWindow.display();

            i2++;            
        }
    } while (!quit);
    

    return 0;
}