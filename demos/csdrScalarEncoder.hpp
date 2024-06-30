#ifndef _CSDR_ENCODER_HPP__
#define _CSDR_ENCODER_HPP__

#include <vector>
//#include <cassert>
#include <random>
#include <algorithm>


float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

//#########################################
// Encode (binning method)
int binningEncoder(float val, int numColumns, float squashingConstant = 3.f)
{
	return static_cast<int>(sigmoid(val * squashingConstant) * (numColumns - 1) + 0.5f);
};

std::vector<int> binningEncoder(std::vector<float> val, int numColumns, float squashingConstant = 3.f)
{
    std::vector<int> res; res.reserve(val.size());
    for (auto v : val) res.push_back(binningEncoder(v, numColumns, squashingConstant));
	return res;
};
//#########################################


//#########################################
class cpScalarEncoder
{
	std::vector<std::vector<std::vector<float>>> protos;

    int nScalars, nColumns, nCellsPerColumn;
    float lowerBound, upperBound;

public:
	cpScalarEncoder(int nScalars_, int nColumns_, int nCellsPerColumn_, float lowerBound_ = 0.f, float upperBound_ = 1.f)
        : nScalars(nScalars_), nColumns(nColumns_), nCellsPerColumn(nCellsPerColumn_), lowerBound(lowerBound_), upperBound(upperBound_)
	{
        // Creat a random number generator
        std::mt19937 generator(time(nullptr));
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        protos.reserve(nColumns);
		for (auto i = 0; i < nColumns; ++i)
		{
            std::vector<std::vector<float>> tmp; tmp.reserve(nCellsPerColumn);
			for (auto j = 0; j < nCellsPerColumn; ++j)
            {
                std::vector<float> tmp2; tmp2.reserve(nScalars);
                for (auto k = 0; k < nScalars; ++k) tmp2.push_back(dist01(generator) * (upperBound - lowerBound) + lowerBound );
                tmp.push_back(tmp2);
            }
            protos.push_back(tmp);
		}
	}

    std::vector<int> encode(std::vector<float> in)
    {
        // in has a length of nScalars
        std::vector<int> csdr; csdr.reserve(nColumns);
        for (auto p : protos)
        {
            std::vector<float> acts; acts.reserve(nCellsPerColumn);
            for (auto pp : p)
            {
                float act = 0.f;
                for (auto k = 0; k < nScalars; ++k) act += std::pow(in[k] - pp[k], 2);
                acts.push_back(-act);
            }

            auto res = std::max_element(acts.begin(), acts.end());
            csdr.push_back(std::distance(acts.begin(), res));
        }
        return csdr;
    }

    std::vector<float> decode(std::vector<int> csdr)
    {
        std::vector<float> acts(nScalars, 0.f);
        for (auto i = 0; i < nColumns; ++i)
        {
            for (auto j = 0; j < nScalars; ++j) acts[j] += protos[i][csdr[i]][j];
        }
        return acts;
    }
};
//#########################################


//#########################################
// encode float into single-point CSDR
int simpleFloat2CSDR(float x, int cells_per_column, float minVal = 0.f, float maxVal = 1.f)
{
    return static_cast<int>((x - minVal) / (maxVal - minVal) * (cells_per_column - 1) + 0.5f);
};

// decode single-point CSDR into float
float simpleCSDR2Float(int Index, int cells_per_column, float minVal = 0.f, float maxVal = 1.f)
{
    return static_cast<float>(Index) / static_cast<float>(cells_per_column - 1) * (maxVal - minVal) + minVal;
};
//#########################################

//#########################################
// encode float into 2 4bits-integer (each has the max value 16)
std::vector<int> Unorm8ToCSDR(float x, float minVal = 0., float maxVal = 1.)
{
    // make sure that input value must be in the interval [0, 1]
    x = (x - minVal) / (maxVal - minVal);
    // scaling it into [0, 255] and consider only the first byte
    int i = int(x * 255.0 + 0.5) & 0xff;
    // split the first byte into 2 parts, each has 4 bit or max value of pow(2,4) = 16
    std::vector<int> res = {int(i & 0x0f), int((i & 0xf0) >> 4) };
    return res;
};

// decode 4bits-integer CSDR into float
float CSDRToUnorm8(std::vector<int> csdr, float minVal = 0., float maxVal = 1.)
{
    auto xnorm = (csdr[0] | (csdr[1] << 4)) / 255.0;
    return xnorm * (maxVal - minVal) + minVal;
};
//#########################################

//#########################################
std::vector<int> fToCSDR(float x, int num_columns, int cells_per_column, float scale_factor=0.25f)
{
    std::vector<int> csdr;
    const float factor = x > 0.0 ? 1.0f : -1.0;

    float scale  = 1.0f;
    for (auto i = 0;  i< num_columns; ++i)
    {        
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
//#########################################


/*
union PackF{
   float i;
   char c[sizeof(float)];
};

union Pack{
   int i;
   char c[sizeof(int)];
};

Pack p = {};
p.i = 1234;
std::string packed(p.c, sizeof(int)); // "\xd2\x04\x00\0"

# Convert an IEEE float to 8 columns with 16 cells each (similar to first approach but on floating-point data)
def IEEEToCSDR(x : float):
    b = struct.pack("<f", x)

    csdr = []

    for i in range(4):
        csdr.append(b[i] & 0x0f)
        csdr.append((b[i] & 0xf0) >> 4)

    return csdr

def CSDRToIEEE(csdr):
    bs = []

    for i in range(4):
        bs.append(csdr[i * 2 + 0] | (csdr[i * 2 + 1] << 4))

    return struct.unpack("<f", bytes(bs))[0]
*/

#endif
