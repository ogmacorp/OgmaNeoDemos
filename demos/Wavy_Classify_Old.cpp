// ------------------------------------i----------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016-2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <aogmaneo/hierarchy.h>

#include "getopt.h"

#include <cmath>

#include <fstream>
#include <sstream>
#include <cmath>

#include <fstream>
#include <vector>
#include <iostream>
#include "vis/Plot.hpp"
#include "vis/guiControl_sfml.hpp"

#include <valarray>

#include <string>

using namespace aon;

const int S = 2048;
const int L = 32;

typedef Hierarchy<S, L> Hierarchy1;
typedef Vec<S, L> Vec1;
typedef Bundle<S, L> Bundle1;

Vec1 embedding1d(float x, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 2] * x + loc[i * 2 + 1];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

Vec1 embedding2d(float x, float y, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 3] * x + loc[i * 3 + 1] * y + loc[i * 3 + 2];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

Vec1 embedding3d(float x, float y, float z, const std::valarray<float> &loc) {
    // mat mul
    Vec1 p;

    for (int i = 0; i < S; i++) {
        float f = loc[i * 4] * x + loc[i * 4 + 1] * y + loc[i * 4 + 2] * z + loc[i * 4 + 3];

        if (f < 0.0f)
            f = 1.0f - std::fmod(-f, 1.0f);
        else
            f = std::fmod(f, 1.0f);

        int v = static_cast<int>(f * L);

        p[i] = v;
    }

    return p;
}

void print(const Vec1 &v) {
    std::cout << "[ ";

    for (int i = 0; i < v.segments(); i++)
        std::cout << static_cast<int>(v[i]) << " ";

    std::cout << " ]" << std::endl;
}

void print(const Bundle1 &b) {
    std::cout << "[ ";

    for (int i = 1; i < b.size(); i++)
        std::cout << b[i] << " ";

    std::cout << " ]" << std::endl;
}

void splitString(const std::string& s, std::string c, std::vector<std::string>& v)
{
    std::string::size_type i = 0;
    std::string::size_type j = s.find(c);

    const int len = c.length();

    while (j != std::string::npos) {
        v.push_back(s.substr(i, j-i));
        j += len;
        i = j;
        j = s.find(c, j);

        if (j == std::string::npos)
            v.push_back(s.substr(i, s.length()));
    }
}

class CustomStreamReader : public aon::Stream_Reader {
public:
    std::ifstream ins;

    void read(
        void* data,
        long len
    ) override {
        ins.read(static_cast<char*>(data), len);
    }
};

class CustomStreamWriter : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        long len
    ) override {
        outs.write(static_cast<const char*>(data), len);
    }
};

class BufferReader : public aon::Stream_Reader {
public:
    int start;
    const std::vector<unsigned char>* buffer;

    BufferReader() :
      start(0), buffer(nullptr)
    {}

    void read( void* data, long len ) override
    {
      for (int i = 0; i < len; i++)
        static_cast<unsigned char*>(data)[i] = (*buffer)[start + i];

      start += len;
    };
};

class BufferWriter : public aon::Stream_Writer {
public:
    int start;

    std::vector<unsigned char> buffer;

    BufferWriter(int size ) :
      start(0)
    {
        buffer.resize(size);
    }

    void write( const void* data, long len) override
    {
      assert(buffer.size() >= start + len);

      for (int i = 0; i < len; i++)
          buffer[start + i] = static_cast<const unsigned char*>(data)[i];

      start += len;
    };
};


// encode float into 2 integer, each has the max value 16
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

std::vector<int> fToCSDR(float x, int num_columns, int cells_per_column, float scale_factor=0.25f)
{
    std::vector<int> csdr;

    float scale = 1.0f;

    for (auto i = 0;  i< num_columns; ++i)
    {
        float factor = x > 0.0 ? 1.0f : -1.0;
        float s = std::fmod(x / scale, factor);

        csdr.push_back(int((s * 0.5 + 0.5) * (cells_per_column - 1) + 0.5));

        x -= scale * (float(csdr[i]) / float(cells_per_column - 1) * 2.0 - 1.0);

        scale *= scale_factor;
    }

    return csdr;
}

float CSDRToF(std::vector<int> csdr, int cells_per_column, float scale_factor=0.25f)
{
    float x = 0.0f;

    float scale = 1.0f;

    for (auto c : csdr)
    {
        x += scale * (float(c) / float(cells_per_column - 1) * 2.0 - 1.0);

        scale *= scale_factor;
    }

    return x;
}


int main(int argc, char *argv[])
{
    std::string hFileName = "wavyClass.ohr";

    std::cout << aon::min(3.0f, 4.0f) << " " << aon::max(3.0f, 4.0f) << std::endl;
    std::cout << aon::min(-3.0f, -8.0f) << " " << aon::max(-3.0f, -8.0f) << std::endl;

    bool loadHierarchy  = false;
    int numAdditionalStepsAhead = 0;

    int opt;
	while ((opt = getopt(argc, argv, "p:h:")) != -1) {  // for each option...
		switch (opt) {
		case 'p':			
			numAdditionalStepsAhead = std::stoi(optarg);
			break;        
		case 'h':
			loadHierarchy = std::stoi(optarg);
			break;
		case '?':
			std::cerr << "valid option -p num_prediction_steps or -h load__model!" << std::endl;
			break;
		}
	}

    bool saveHierarchy  = !loadHierarchy;
    // --------------------------- Create the window(s) ---------------------------

    sf::RenderWindow renderWindow;
	int winW = 1000, winH=800, winMainW=1800;
	int numPlots = 2;
 
	int subWinH = static_cast<int>(winH / numPlots);
	int plotHeight = subWinH - 5;
    int plotWidth = winMainW -5;
    const int maxBufferSize = 500;

    sf::Vector2i screenDimensions(winMainW, winH);

	renderWindow.create(sf::VideoMode(winMainW, winH), "wavy Classification", sf::Style::Default);
	renderWindow.setFramerateLimit(60);

	guiControl guiC(screenDimensions);

    vis::Plot plot(sf::Vector2f(0,0), sf::Vector2i(plotWidth, plotHeight), 2, "index", "input/prediction", maxBufferSize);
	vis::Plot plotClass(sf::Vector2f(0,subWinH), sf::Vector2i(plotWidth, plotHeight), 2, "index", "pred/GT Class", maxBufferSize);

    int cRadius  = 30;
    int cRadius2 = cRadius / 2;
    sf::CircleShape circle(cRadius);   // circle for learn state: red - learn, green - recognition
    circle.setPosition(sf::Vector2f( renderWindow.getSize().x - 2.3*cRadius, cRadius2));

    // the min ans max value of the input data
    float minY = -3.0f;
    float maxY = +3.0f;

    sf::Texture lineGradient;
    lineGradient.loadFromFile("resources/lineGradient.png");

    sf::Font tickFont;
    tickFont.loadFromFile("resources/Hack-Regular.ttf");

    //for (float x = 0.1f; x <= 10.0f; x += 0.1f)
    //    std::cout << std::log2(x) << std::endl;

    //for (float x = 0.1f; x <= 10.0f; x += 0.1f)
    //    std::cout << aon::log2f(x) << std::endl;

    //float mean = 0.0f;
    //float variance = 0.0f;

    //for (int i = 0; i < 1000; i++) {
    //    float y = aon::rand_normalf();
    //    std::cout << y << std::endl;

    //    mean += y;
    //    variance += y * y;
    //}

    //mean /= 1000.0f;
    //variance /= 1000.0f;

    //std::cout << mean << " " << variance << std::endl;

    //for (int i = 0; i < 100; i++) {
    //    float x = aon::randf(0.0f, 100.0f);

    //    std::cout << x << " -> " << aon::rand_roundf(x) << std::endl;
    //}

    // generate RNG image
    {
        sf::Image img;
        img.create(512, 512);

        for (int x = 0; x < img.getSize().x; x++)
            for (int y = 0; y < img.getSize().y; y++) {
                unsigned char r = aon::rand() % 256;
                unsigned char g = aon::rand() % 256;
                unsigned char b = aon::rand() % 256;

                img.setPixel(x, y, sf::Color(r, g, b));
            }

        img.saveToFile("rng_test.png");
    }

    // --------------------------- Create the Hierarchy ---------------------------

    const int inputColumnSize = 32;
//#define USE_SENSOR_DATA

#define SINGLE_COLUMN_ENCODER
#ifdef SINGLE_COLUMN_ENCODER
    const int numInputColumns = 1;
#else
    const int numInputColumns = 2;
#endif

    set_num_threads(8);

    Array<Layer_Desc> lds(1);

    for (int i = 0; i < lds.size(); i++) {
        lds[i].hidden_size = Int2(1, 1);
        lds[i].temporal_horizon = 32;
    }

    // here we use the measuring data in 1st input     --> InputType = prediction
	//             the labels data in 2nd input        --> InputType = prediction
    Array<IO_Desc> ioDescs(2);

    //ioDescs[0] = Hierarchy::IODesc(Int3(1, numInputColumns, inputColumnSize), IOType::prediction, 4, 2, 2, 32);
    ioDescs[0] = IO_Desc(Int2(1, 1), IO_Type::prediction);

    const int label_width  = 1;
    const int label_height = 1;
#ifdef USE_SENSOR_DATA    
    const int numLabels    = 3;
#else    
    const int numLabels    = 5;
#endif    
    const int label_num_cells_per_column = numLabels;   // same as number of classes
    //ioDescs[1] = Hierarchy::IODesc(Int3(label_width, label_height, label_num_cells_per_column), IOType::prediction, 2, 2, 2, 32);
    ioDescs[1] = IO_Desc(Int2(1, 1), IO_Type::prediction);

    Hierarchy1 h;
    bool learnFlag = true;

    if (loadHierarchy)
	{
        std::cout << "load hierarchy..";
		CustomStreamReader reader;
		reader.ins.open(hFileName.c_str(), std::ios::binary);
		h.read(reader);
        std::cout << "... finished" << std::endl;
        learnFlag = false;
	}
	else
	{
		h.init_random(ioDescs, lds);
	}

    Array<Vec1> loc_vecs(inputColumnSize);

    std::valarray<float> loc(2 * S);

    for (int i = 0; i < loc.size(); i++)
        loc[i] = aon::rand_normalf() * 1.0f;

    for (int i = 0; i < loc_vecs.size(); i++)
        loc_vecs[i] = embedding1d(static_cast<float>(i) / loc_vecs.size() * (maxY - minY) + minY, loc);

    Array<Vec1> lab_vecs(numLabels);

    for (int i = 0; i < lab_vecs.size(); i++)
        lab_vecs[i] = Vec1::randomized();

    Bundle<S, L> memory = 0.0f;
    Vec<S, L> input = 0;
    Vec<S, L> input_prev = 0;

    int hStateSize = h.state_size();

#ifdef USE_SENSOR_DATA
    std::string fname("resources/training_camdataDetrend.txt");

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
    minY = -8.f;
    maxY = +8.f;
#endif

    bool quit = false;
    bool autoplay = true;
    bool spacePressedPrev = false;

    int index = -1;
    int sequenceID = 0;
    int predIndex;
    int encodedInPrev = 0;

    do {
        guiC.eventHandling(renderWindow, quit);

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            quit = true;

        bool spacePressed = sf::Keyboard::isKeyPressed(sf::Keyboard::Space);

        if (spacePressed && !spacePressedPrev)
            autoplay = !autoplay;

        spacePressedPrev = spacePressed;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::L)) learnFlag = 1;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::P)) learnFlag = 0;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num0)) sequenceID = 0;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num1)) sequenceID = 1;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num2)) sequenceID = 2;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num3)) sequenceID = 3;
        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Num4)) sequenceID = 4;

        if (autoplay || sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
            index++;

            if (index % 1000 == 0)
                std::cout << "Step: " << index << std::endl;

            float value;
#ifdef USE_SENSOR_DATA
            value = sensorData[index];
            sequenceID = labels[index];
#else
            float in0 = std::sin(0.025f * pi * index + 0.25f);
            //float in0 = (index % 50 == 0 || index % 7 == 0 ? 1.0f : 0.0f);
            float in1 = std::sin(0.09f * pi * index + 1.5f);
            float in2 = std::sin(0.05f * pi * index - 0.1f);

            switch (sequenceID)
            {
            case 0:
                value = in0;
                break;
            case 1:
                value = in1;
                break;
            case 2:
                value = in2;
                break;
            case 3:
                value = in0 + in1;
                break;
            case 4:
                value = in0 + in1 + in2;
                break;
            }
#endif

            Array<Array_View<Vec<S, L>>> all_input_vecs(ioDescs.size());

            int encodedIn = static_cast<int>((value - minY) / (maxY - minY) * (inputColumnSize - 1) + 0.5f);

            Array<Vec<S, L>> loc_input(1);
            loc_input[0] = loc_vecs[encodedIn];

            Array<Vec<S, L>> lab_input(1);
            lab_input[0] = lab_vecs[sequenceID];

            all_input_vecs[0] = loc_input;
            all_input_vecs[1] = lab_input;

            input_prev = input;
            input = (loc_vecs[encodedIn] + input * loc_vecs[encodedIn]).thin();

            if (!learnFlag || sf::Keyboard::isKeyPressed(sf::Keyboard::P))
            {
                // Prediction mode
                std::cout << "Prediction mode" << std::endl;
                //inputCIs[0] = &h.get_prediction_cis(0); // memory prediction, independent in current input
                all_input_vecs[1] = h.get_prediction_vecs(1);
                h.step(all_input_vecs, false);

            }
            else {
                // training mode
                h.step(all_input_vecs, true);

                memory *= 0.999f;
                memory += input_prev.permute() * loc_vecs[encodedIn];
            }

            encodedInPrev = encodedIn;

            //Vec1 pred_loc_vec = h.get_prediction_vecs(0)[0];
            //Vec1 pred_lab_vec = h.get_prediction_vecs(1)[0];
            Vec1 pred_loc_vec = memory.thin() / input.permute();

            float predValue = 0.0f;
            int pred_label = 0;

            {
                // search for closest vector
                int max_index = 0;
                int max_similarity = 0;

                for (int i = 0; i < loc_vecs.size(); i++) {
                    int similarity = pred_loc_vec.dot(loc_vecs[i]);

                    if (similarity > max_similarity) {
                        max_similarity = similarity;
                        max_index = i;
                    }
                }

                predValue = static_cast<float>(max_index) / loc_vecs.size() * (maxY - minY) + minY;
            }

            renderWindow.clear();

            // plot target data
			vis::Point p;
            p._position.x = index;
			p._position.y = value;
			p._color = sf::Color::Red;
			plot.updateBuffer(0, p);

			// plot predicted data
			vis::Point p1;
			p1._position.y = predValue;
			p1._color = sf::Color::Blue;
			plot.updateBuffer(1, p1);
			plot.draw(renderWindow, sf::Vector2f(1.5*minY, 1.5*maxY), 5);

            vis::Point pC;
            pC._position.x = index;
			pC._position.y = sequenceID;
			pC._color = sf::Color::Red;
			plotClass.updateBuffer(0, pC);

			// plot predicted data
			vis::Point pC1;
			pC1._position.y = pred_label;
			pC1._color = sf::Color::Blue;
			plotClass.updateBuffer(1, pC1);
			plotClass.draw(renderWindow, sf::Vector2f(0, 5), 5);

            // display learn state + framerate
            if (learnFlag)
                circle.setFillColor(sf::Color(255, 0, 0));
            else
                circle.setFillColor(sf::Color(0, 255, 0  )); // inference

            renderWindow.draw(circle);

            if (!learnFlag || index % 1000 == 0) {
                renderWindow.display();
            }

        }
    } while (!quit);

    if (saveHierarchy)
    {
        std::cout << "save hierarchy..";
        CustomStreamWriter writer;
        writer.outs.open(hFileName.c_str(), std::ios::out | std::ios::binary);
        h.write(writer);
        std::cout << "... finished" << std::endl;
    }
    return 0;
}
