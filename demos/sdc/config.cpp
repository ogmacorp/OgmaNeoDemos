#include "config.h"

bool Dynamic_Config::load(
    const std::string &file_name
) {
    std::ifstream from_file(file_name);

    if (!from_file.is_open()) {
        std::cerr << "Could not open config! Using defaults" << std::endl;

        return false;
    }

    from_file >> trim >> motion_threshold;

    return true;
}

bool Dynamic_Config::save(
    const std::string &file_name
) {
    std::ofstream to_file(file_name);

    to_file << trim << " " << motion_threshold << std::endl;

    return true;
}

Dynamic_Config dcfg;
