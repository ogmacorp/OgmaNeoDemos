#include <iostream>
#include <cmath>
#include <random>
#include <time.h>

struct CartPoleEnv {
    float x;
    float dx;
    float theta;
    float dtheta;

    CartPoleEnv()
    :
    x(0.0f),
    dx(0.0f),
    theta(0.0f),
    dtheta(0.0f)
    {}

    void step(float force) {
        static const float g = 9.81f;
        static const float mp = 1.0f;
        static const float mc = 2.0f;
        static const float l = 1.0f;
        static const float dt = 1.0f / 60.0f;
        static const float mpmc_inv = 1.0f / (mp + mc);
        static const float four_thirds = 4.0f / 3.0f;
        static const float mpl = mp * l;

        float cos_theta = std::cos(theta);
        float sin_theta = std::sin(theta);
        float dtheta2 = dtheta * dtheta;
        float dtheta2_sin_theta = dtheta2 * sin_theta;

        float ddtheta = (g * sin_theta + cos_theta * ((-force - mpl * dtheta2_sin_theta) * mpmc_inv)) / (l * (four_thirds - (mp * cos_theta * cos_theta) * mpmc_inv));
        float ddx = (force + mp * (dtheta2_sin_theta - ddtheta * cos_theta)) * mpmc_inv;

        dx += ddx * dt;
        dtheta += ddtheta * dt;

        x += dx * dt;
        theta += dtheta * dt;
    }
};

int main() {
    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> thetaDist(-0.03f, 0.03f);

    CartPoleEnv env;
    env.theta = thetaDist(rng);

    for (long i = 0; i < 1000000000; i++)
        env.step(0.05f);

    std::cout << "END" << std::endl;

    std::cout << env.x << std::endl;

    return 0;
}
