#ifndef ENV_HEADER
#define ENV_HEADER
#include <random>
#include <string>
#include <vector>

#ifndef M_PIf32
#define M_PIf32 3.14159265358979323846
#endif

class Env 
{
protected:
  uint32_t      _seed;
  std::mt19937  _ran_generator;

  // currently, max_steps is used only for some applications,
  int max_steps = 500;

public:
  Env(uint32_t seed = 28) : _seed(seed)
  {
    _ran_generator = std::mt19937(_seed);
  };

  ~Env() {};

  void setMaxSteps (int steps) {max_steps = steps;};

  void setSeed(uint32_t seed)
  {
    _seed           = seed;
    _ran_generator  = std::mt19937(_seed);
  };

  uint32_t getSeed() {return _seed;};

  virtual std::tuple<std::vector<float>, double, bool> step(const int &action) {return {};};
  virtual std::tuple<std::vector<float>, double, bool> step(std::vector<double> &action) {return {};};
  virtual std::vector<float> reset() {return {};};
};
#endif
