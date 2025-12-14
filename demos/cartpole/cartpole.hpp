#ifndef CARTPOLE_HEADER
#define CARTPOLE_HEADER
#include <tuple>
#include <utility>
#include <vector>
#include <set>
#include <random>
#include <cmath>
#include "env.hpp"

class CartPoleEnv : public Env
{
  public:
    // Constants
    const float gravity = 9.8;
    const float masscart = 1.0;
    const float masspole = 0.1;
    const float total_mass = (masspole + masscart);
    const float length = 0.5;
    const float polemass_length = (masspole * length);
    const float force_mag = 10.0;
    const float tau = 0.02;
    const float theta_threshold_radians = 12.f * 2.f * M_PIf32 / 360.f;
    const float x_threshold = 2.4;
    const float fourthirds = 4.0f / 3.0f;
    const float polemassfrac = masspole / total_mass;
    const float polemasslengthfrac = polemass_length / total_mass;

    // State
    int steps_beyond_done = -1;
    std::vector<float> state; // contains position, velocity, angle, angular velocity.

    CartPoleEnv();
    CartPoleEnv(CartPoleEnv &other);
    ~CartPoleEnv() {};

    std::vector<float> reset();
    std::tuple<std::vector<float>, double, bool> step(const int &action);
};
#endif
