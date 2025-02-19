#ifndef _aogmaneo_utils_hpp
#define _aogmaneo_utils_hpp

#include <aogmaneo/helpers.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <tuple>

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


/**
 * PatternAnalyse
 * currentPatternData:  input array of the top hidden layer
 * frameIndex:          current frame number
 * output:              typle of matchScores, currentMatchIndex , currentMatchLength
*/
std::tuple<float, int, int> PatternAnalyse(std::vector<int> currentPatternData, int frameIndex)
{
  static std::vector<int> lastPatternData;       // the last CSDR pattern
  static std::vector<int> repeatPatternData;     // the first repeat CSDR-pattern

  // temporary data for finding the 1st repeated CSDR
  // after that they are cleared
  static std::vector<std::vector<int>> tmpPatterns;
  static std::vector<int> tmpIndices;

  static int currentMatchIndex  = 0;
  static int currentMatchLength = 0;

  const int topHSize = currentPatternData.size();

  float matchScores = 0.0;
  bool  firstMatch  = false;

  if (!lastPatternData.empty())
  {
    // this changes every fixed number of steps (here 32 steps, because numLayers = 6, 32 = 2 ^ 6) 
    // so that for identifying a input pattern
    // it repeats after every 25 changes:
    // That means:
    // Top Hidden Layer represents 25 different patterns
    //
    // By changing
    // inValues[0] = std::sin(0.0125f * M_PI * index * 0.5 + 0.25f) * std::sin(0.03f * M_PI * index + 1.5f) * std::sin(0.025f * M_PI * index - 0.1f);
    // we got Top Hidden Layer represents 50 different patterns
    bool identical = true;                
    for (auto i = 0; i < topHSize; ++i)
      if (lastPatternData[i] != currentPatternData[i]) {identical = false; break;}
    
    if (!identical)
    {
      // mark the time point the whole pattern changes (for each 32 steps) <--> new context
      if (!repeatPatternData.empty())
      {
        int numIdenticals = 0;
        for (auto i = 0; i < topHSize; ++i)
          if (repeatPatternData[i] == currentPatternData[i]) ++numIdenticals;
        
        matchScores = static_cast<float>(numIdenticals) / topHSize;                
      }
      else
      {
        // we really do not know how long it learns. Only after finish learning, the top hidden layer
        // can provide stable pattern
        // for this reason, we register temporary pattern and wait until any of temporary pattern can
        // be found as repeated
        if (!tmpPatterns.empty())
        {
          for (auto k = 0; k < tmpPatterns.size(); ++k)
          {
            int numIdenticals = 0;
            for (auto i = 0; i < topHSize; ++i)
              if (tmpPatterns[k][i] == currentPatternData[i]) ++numIdenticals;

            matchScores = static_cast<float>(numIdenticals) / topHSize;
            if (numIdenticals == topHSize) 
            {
              firstMatch          = true;
              currentMatchIndex   = frameIndex;
              currentMatchLength  = tmpIndices[tmpPatterns.size()-k] - tmpIndices[0];
              repeatPatternData   = currentPatternData;
              tmpPatterns.clear();
              tmpIndices.clear();                                
              break;
            }
          }
        }    
        if (repeatPatternData.empty())    
        {
          // no repeat pattern found, try to collect the current pattern               
          tmpPatterns.push_back(currentPatternData);
          tmpIndices.push_back(frameIndex);
        }
      }
    }
  }
  lastPatternData = currentPatternData;  

  if (matchScores == 1.0)
  {
    if (firstMatch)
    {
      // currentMatchLength, currentMatchIndex have been already estimated by checking 
      std::cout << "found 1st repeated CSDR at frame index: " << frameIndex << " and estimated period: " << currentMatchLength << std::endl;
    }
    else
    {
      auto tmpLen = frameIndex - currentMatchIndex;
      std::string tmpTxt = "";
      if (currentMatchLength > 0 && currentMatchLength != tmpLen)
          tmpTxt = " + updated period to " + std::to_string(tmpLen);
      currentMatchLength = tmpLen;
      currentMatchIndex  = frameIndex;
      std::cout << "found repeated CSDR at index: " << frameIndex << tmpTxt << std::endl;
    }
    return std::make_tuple(matchScores, currentMatchIndex , currentMatchLength);
  }
  else
    return std::make_tuple(matchScores, 0 , 0);
}

#endif
