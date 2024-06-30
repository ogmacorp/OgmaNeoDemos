#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include <openvr/openvr.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

std::string readFileIntoString(const std::string &path) {
    std::ifstream input_file(path);

    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '"<< path << "'" << std::endl;
        exit(EXIT_FAILURE);
    }

    return std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());
}

int main(int argc, char *argv[]) {
    std::mt19937 rng(time(nullptr));

    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    vr::EVRInitError eError = vr::VRInitError_None;
    vr::IVRSystem* hmd = vr::VR_Init(&eError, vr::VRApplication_Utility);

    if (eError != vr::VRInitError_None) {
        std::cerr << "Error: " << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;

        return 1;
    }

    std::string bindingsPath = "/home/ericl/Documents/Projects/OgmaNeoDemos/resources/openvr-binding-files/openvr_tracking_example_actions.json";

    vr::EVRInputError inputError = vr::VRInput()->SetActionManifestPath(bindingsPath.c_str());

    if (inputError != vr::VRInputError_None) {
        std::cerr << "Input error: " << inputError << std::endl;

        return 1;
    }

    vr::VRActionHandle_t leftHand;

    inputError = vr::VRInput()->GetActionHandle("/actions/demo/in/Hand_Left", &leftHand);

    if (inputError != vr::VRInputError_None) {
        std::cerr << "Input error - left hand!" << std::endl;

        return 1;
    }

    vr::InputPoseActionData_t poseData;
    vr::HmdMatrix34_t pose;

    // --------------------------- Create the window(s) ---------------------------

    unsigned int windowWidth = 1000;
    unsigned int windowHeight = 500;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "VR Test", sf::Style::Default);

    window.setVerticalSyncEnabled(false);
    window.setFramerateLimit(60);

    bool quit = false;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;
        }

        inputError = vr::VRInput()->GetPoseActionDataForNextFrame(leftHand, vr::TrackingUniverseStanding, &poseData, sizeof(poseData), vr::k_ulInvalidInputValueHandle);

	if (inputError == vr::VRInputError_None) {
            if (poseData.bActive) {
                vr::VRInputValueHandle_t activeOrigin = poseData.activeOrigin;

                pose = poseData.pose.mDeviceToAbsoluteTracking;

                std::cout << pose.m[0][0] << std::endl;
            }
	}
        else {
            std::cerr << "Pose error: " << inputError << std::endl;
        }

        window.clear();

        window.display();
    } while (!quit);

    vr::VR_Shutdown();

    return 0;
}
