#include "cartpole.hpp"


// Original source from Python.
// https://github.com/openai/gym

CartPoleEnv::CartPoleEnv()
{
}

CartPoleEnv::CartPoleEnv(CartPoleEnv &other) {
  steps_beyond_done = other.steps_beyond_done;
  state             = other.state;
  _seed             = other._seed;
  _ran_generator    = other._ran_generator;
}

std::tuple<std::vector<float>, double, bool> CartPoleEnv::step(const int &action)
{
  float x         = state[0];
  float x_dot     = state[1];
  float theta     = state[2];
  float theta_dot = state[3];
  float force     = (action == 1) ? force_mag : -force_mag;

  const float costheta  = cos(theta);
  const float sintheta  = sin(theta);
  const float temp      = (force + polemass_length * pow(theta_dot, 2.f) * sintheta) / total_mass;
  const float thetaacc  = (gravity * sintheta - costheta * temp) / (length * (fourthirds - polemassfrac * pow(costheta, 2.f) ));
  const float xacc      = temp - polemasslengthfrac * thetaacc * costheta;

  x         += tau * x_dot;
  x_dot     += tau * xacc;
  theta     += tau * theta_dot;
  theta_dot += tau * thetaacc;

  state = {x, x_dot, theta, theta_dot };

  bool done = x < -x_threshold || x > x_threshold || theta < -theta_threshold_radians || theta > theta_threshold_radians;

  float reward = 0.f;
  if(!done)
    reward = 1.0f;
  else if (steps_beyond_done == -1) {
    steps_beyond_done = 0;
    reward = 1.0f;
  } else {
    steps_beyond_done++;
    reward = 0.0f;
  }

  return {state, reward, done};
}

std::vector<float> CartPoleEnv::reset()
{ 
  steps_beyond_done = -1;

  std::uniform_real_distribution<float> distribution(-.05, .05);
  state = {
    distribution(_ran_generator),
    distribution(_ran_generator),
    distribution(_ran_generator),
    distribution(_ran_generator)
  };
  return state;
}
