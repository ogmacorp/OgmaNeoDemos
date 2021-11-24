#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "catmouse/CatMouseEnv.h"

#include <cmath>

#include <torch/torch.h>

#include <time.h>
#include <iostream>
#include <fstream>
#include <random>

int main() {
    bool load = false;
    bool manualControl = false;
    float epsilon = 0.0f;

    torch::Tensor t = torch::eye(3);

    std::cout << t << std::endl;


    return 0;
}
