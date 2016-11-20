// ----------------------------------------------------------------------------
//  OgmaNeoDemos
//  Copyright(c) 2016 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of OgmaNeoDemos is licensed to you under the terms described
//  in the OGMANEODEMOS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include <neo/Architect.h>
#include <neo/Hierarchy.h>
#include <neo/SparseFeaturesChunk.h>

#include <vis/Plot.h>

#include <time.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <random>
#include <algorithm>

using namespace ogmaneo;

// Temporary MNIST image buffer
struct Image {
    std::vector<sf::Uint8> _intensities;
};

// MNIST image loading
void loadMNISTimage(std::ifstream &fromFile, int index, Image &img) {
    const int headerSize = 16;
    const int imageSize = 28 * 28;

    fromFile.seekg(headerSize + index * imageSize);

    if (img._intensities.size() != 28 * 28)
        img._intensities.resize(28 * 28);

    fromFile.read(reinterpret_cast<char*>(img._intensities.data()), 28 * 28);
}

// MNIST label loading
int loadMNISTlabel(std::ifstream &fromFile, int index) {
    const int headerSize = 8;

    fromFile.seekg(headerSize + index * 1);

    char label;

    fromFile.read(&label, 1);

    return static_cast<int>(label);
}

int main() {
    const float moveSpeed = 1.0f; // Speed digits move across the screen
    const float spacing = 28.0f; // Spacing between digits
    const int targetLabel = 3; // Label of non-anomalous class
    const int totalMain = 5; // Amount of target (main) digits to load
    const int totalAnomalous = 2000; // Amount of anomalous digits to load
    const float anomalyRate = 0.1f; // Ratio of time which anomalies randomly appear
    const int imgPoolSize = 20; // Offscreen digit pool buffer size
    const float sensitivity = 1.1f; // Sensitivity to anomalies
    const float averageDecay = 0.01f; // Average activity decay
    const int numSuccessorsRequired = 4; // Number of successors before an anomaly is signalled
    const float spinRate = 5.0f; // How fast the digits spin
    const int okRange = 1; // Approximate range (in digits) where an anomaly flag can be compared to the actual anomaly outcome

    // Creat a random number generator
    std::mt19937 generator(time(nullptr));

    // --------------------------- Create the Hierarchy ---------------------------

    // Bottom input width and height
    int bottomWidth = 28;
    int bottomHeight = 28;

    std::shared_ptr<ogmaneo::Resources> res = std::make_shared<ogmaneo::Resources>();

    res->create(ogmaneo::ComputeSystem::_gpu);

    ogmaneo::Architect arch;
    arch.initialize(1234, res);

    // 1 input layer
    arch.addInputLayer(ogmaneo::Vec2i(bottomWidth, bottomHeight))
        .setValue("in_p_alpha", 0.02f)
        .setValue("in_p_radius", 12);

    // 8 layers using chunk encoders
    for (int l = 0; l < 8; l++)
        arch.addHigherLayer(ogmaneo::Vec2i(96, 96), ogmaneo::_chunk)
        .setValue("sfc_chunkSize", ogmaneo::Vec2i(6, 6))
        .setValue("sfc_ff_radius", 12)
        .setValue("hl_poolSteps", 2)
        .setValue("p_alpha", 0.08f)
        .setValue("p_beta", 0.16f)
        .setValue("p_radius", 12);

    // Generate the hierarchy
    std::shared_ptr<ogmaneo::Hierarchy> h = arch.generateHierarchy();

    // Input and prediction fields
    ogmaneo::ValueField2D inputField(ogmaneo::Vec2i(bottomWidth, bottomHeight));
    ogmaneo::ValueField2D predField(ogmaneo::Vec2i(bottomWidth, bottomHeight));

    // --------------------------- Create the Windows ---------------------------

    sf::RenderWindow window;

    window.create(sf::VideoMode(1024, 512), "MNIST Anomaly Detection");

    // --------------------------- Plot ---------------------------

    vis::Plot plot;
    plot._curves.resize(1);
    plot._curves[0]._shadow = 0.1f;	

    plot._curves[0]._name = "Prediction Error";

    const int overSizeMult = 6; // How many times the graph should extent past what is visible on-screen in terms of anomaly times

    plot._curves[0]._points.resize(bottomWidth * overSizeMult);

    // Initialize
    for (int i = 0; i < plot._curves[0]._points.size(); i++) {
        plot._curves[0]._points[i]._position = sf::Vector2f(i, 0.0f);
        plot._curves[0]._points[i]._color = sf::Color::Red;
    }

    // Render target for the plot
    sf::RenderTexture plotRT;
    plotRT.create(window.getSize().y, window.getSize().y, false);

    // Resources for the plot
    sf::Texture lineGradient;
    lineGradient.loadFromFile("resources/lineGradient.png");

    sf::Font tickFont;
    
#ifdef _WINDOWS
    tickFont.loadFromFile("C:/Windows/Fonts/Arial.ttf");
#else
#ifdef __APPLE__
    tickFont.loadFromFile("/Library/Fonts/Courier New.ttf");
#else
    tickFont.loadFromFile("/usr/share/fonts/truetype/freefont/FreeMono.ttf");
#endif
#endif

    // Uniform random in [0, 1]
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    // --------------------------- Digit rendering ---------------------------

    // Load MNIST
#if defined(_WINDOWS) || defined(__APPLE__)
    std::ifstream fromImageFile("resources/train-images.idx3-ubyte", std::ios::binary | std::ios::in);
#else
    std::ifstream fromImageFile("resources/train-images-idx3-ubyte", std::ios::binary | std::ios::in);
#endif
    if (!fromImageFile.is_open()) {
        std::cerr << "Could not open train-images.idx3-ubyte!" << std::endl;

        return 1;
    }

#if defined(_WINDOWS) || defined(__APPLE__)
    std::ifstream fromLabelFile("resources/train-labels.idx1-ubyte", std::ios::binary | std::ios::in);
#else
    std::ifstream fromLabelFile("resources/train-labels-idx1-ubyte", std::ios::binary | std::ios::in);
#endif
    if (!fromLabelFile.is_open()) {
        std::cerr << "Could not open train-labels.idx1-ubyte!" << std::endl;

        return 1;
    }

    // Main render target
    sf::RenderTexture rt;

    rt.create(bottomWidth, bottomHeight);

    // Positioning values
    const float boundingSize = (bottomWidth - 28) / 2;
    const float center = bottomWidth / 2;
    const float minimum = center - boundingSize;
    const float maximum = center + boundingSize;

    std::vector<int> mainIndices; // Indicies of main (target) class digits
    std::vector<int> anomalousIndices; // Indicies of anomalous class digits

    // Find indices for both main and anomalous
    int index = 0;

    while (mainIndices.size() < totalMain || anomalousIndices.size() < totalAnomalous) {
        int label = loadMNISTlabel(fromLabelFile, index);

        if (label == targetLabel) {
            if (mainIndices.size() < totalMain)
                mainIndices.push_back(index);
        }
        else {
            if (anomalousIndices.size() < totalAnomalous)
                anomalousIndices.push_back(index);
        }

        index++;
    }

    // Sampling distributions
    std::uniform_int_distribution<int> mainDist(0, mainIndices.size() - 1);
    std::uniform_int_distribution<int> anomolousDist(0, anomalousIndices.size() - 1);

    // Load first digits
    std::vector<sf::Texture> imgPool(imgPoolSize);
    std::vector<float> imgSpins(imgPoolSize, 0.0f);
    std::vector<bool> imgAnomolous(imgPoolSize, false);
    std::vector<int> imgLabels(imgPoolSize, 0);

    for (int i = 0; i < imgPool.size(); i++) {
        Image img;
        int index = mainDist(generator);

        loadMNISTimage(fromImageFile, mainIndices[index], img);
        int label = loadMNISTlabel(fromLabelFile, mainIndices[index]);

        // Convert to SFML image
        sf::Image digit;
        digit.create(28, 28);

        for (int x = 0; x < digit.getSize().x; x++)
            for (int y = 0; y < digit.getSize().y; y++) {
                int index = x + y * digit.getSize().x;

                sf::Color c = sf::Color::White;

                c.a = img._intensities[index];

                digit.setPixel(x, y, c);
            }

        imgPool[i].loadFromImage(digit);
        imgLabels[i] = label;
    }

    // Total width of image pool in pixels
    float imgsSize = spacing * imgPool.size();

    // Current position
    float position = 0.0f;

    // Average anomaly score
    float averageScore = 1.0f;

    // Number of anomalously flagged successors
    int successorCount = 0;

    // Whether the previous digit was flagged as anomalous
    float prevAnomalous = 0.0f;

    // Statistical counters
    int numTruePositives = 0;
    int numFalsePositives = 0;
    int totalSamples = 0;

    // Modes
    bool quit = false;

    bool trainMode = true;

    // Simulaiton loop
    while (!quit) {
        // Poll events
        sf::Event event;

        while (window.pollEvent(event)) {
            switch (event.type) {
            case sf::Event::Closed:
                quit = true;
                break;
            default:
                break;
            }
        }

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
            quit = true;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::R))
            trainMode = false;

        window.clear();

        // Move digits
        position += moveSpeed;

        // If new digit needs to be loaded
        if (position > spacing) {
            // Load new digit
            sf::Texture newImg;

            Image img;

            // Load digit, get anomalous status and label
            bool anomolous = false;

            int label;

            if (trainMode) {
                int index = mainDist(generator);

                loadMNISTimage(fromImageFile, mainIndices[index], img);
                label = loadMNISTlabel(fromLabelFile, mainIndices[index]);
            }
            else {
                if (dist01(generator) < anomalyRate) {
                    int index = anomolousDist(generator);

                    loadMNISTimage(fromImageFile, anomalousIndices[index], img);
                    label = loadMNISTlabel(fromLabelFile, anomalousIndices[index]);

                    anomolous = true;
                }
                else {
                    int index = mainDist(generator);

                    loadMNISTimage(fromImageFile, mainIndices[index], img);
                    label = loadMNISTlabel(fromLabelFile, mainIndices[index]);
                }
            }

            // Convert to SFML image
            sf::Image digit;
            digit.create(28, 28);

            for (int x = 0; x < digit.getSize().x; x++)
                for (int y = 0; y < digit.getSize().y; y++) {
                    int index = x + y * digit.getSize().x;

                    sf::Color c = sf::Color::White;

                    c.a = img._intensities[index];

                    digit.setPixel(x, y, c);
                }

            newImg.loadFromImage(digit);

            // Update pool information
            imgPool.insert(imgPool.begin(), newImg);
            imgPool.pop_back();

            imgSpins.insert(imgSpins.begin(), 0.0f);
            imgSpins.pop_back();

            imgAnomolous.insert(imgAnomolous.begin(), anomolous);
            imgAnomolous.pop_back();

            imgLabels.insert(imgLabels.begin(), label);
            imgLabels.pop_back();

            totalSamples++;

            // Reset position
            position = 0.0f;
        }

        // Render to render target
        rt.clear();

        float offset = -imgsSize * 0.5f;

        for (int i = 0; i < imgPool.size(); i++) {
            // Add spinning motion
            imgSpins[i] += spinRate;

            sf::Sprite s;

            s.setTexture(imgPool[i]);
            s.setPosition(offset + position + i * spacing + imgPool[i].getSize().x * 0.5f, rt.getSize().y * 0.5f - 14.0f + imgPool[i].getSize().y * 0.5f);
            s.setOrigin(imgPool[i].getSize().x * 0.5f, imgPool[i].getSize().y * 0.5f);
            s.setRotation(imgSpins[i]);

            rt.draw(s);
        }

        rt.display();

        // Get image from render target
        sf::Image rtImg = rt.getTexture().copyToImage();

        // ------------------------------------- Anomaly detection -------------------------------------

        // Retrieve prediction
        predField = h->getPredictions().front();

        // Copy image data
        std::vector<float> rtData(rtImg.getSize().x * rtImg.getSize().y);

        for (int x = 0; x < rtImg.getSize().x; x++)
            for (int y = 0; y < rtImg.getSize().y; y++) {
                sf::Color c = rtImg.getPixel(x, y);

                inputField.setValue(ogmaneo::Vec2i(x, y), 0.333f * (c.r / 255.0f + c.b / 255.0f + c.g / 255.0f));
            }

        // Compare input and pred fields (distance)
        float anomalyScore = 0.0f;

        for (int i = 0; i < inputField.getData().size(); i++) {
            float delta = inputField.getData()[i] - predField.getData()[i];

            anomalyScore += -delta * delta;
        }

        anomalyScore /= inputField.getData().size();

        // Detection
        float anomaly = anomalyScore < averageScore * sensitivity ? 1.0f : 0.0f;

        // Successor counting
        if (anomaly > 0.0f)
            successorCount++;
        else
            successorCount = 0;

        // If enough successors, there is a sustained anomaly that needs to be flagged
        float sustainedAnomaly = successorCount >= numSuccessorsRequired ? 1.0f : 0.0f;

        // Adjust average score if in training mode
        if (trainMode)
            averageScore = averageDecay * averageScore + (1.0f - averageDecay) * anomalyScore;

        // Hierarchy simulation step
        std::vector<ogmaneo::ValueField2D> inputVector = { inputField };
        h->simStep(inputVector, trainMode);

        // Shift plot y values
        for (int i = plot._curves[0]._points.size() - 1; i >= 1; i--)
            plot._curves[0]._points[i]._position.y = plot._curves[0]._points[i - 1]._position.y;

        // Add anomaly to plot
        plot._curves[0]._points.front()._position.y = sustainedAnomaly;

        // See if an anomaly is in range
        int center = imgPoolSize / 2;

        // Gather statistics
        bool anomalyInRange = false;

        for (int dx = -okRange; dx <= okRange; dx++)
            if (imgAnomolous[center + dx]) {
                anomalyInRange = true;
                break;
            }

        // First detection
        bool firstDetection = prevAnomalous == 0.0f && sustainedAnomaly == 1.0f;

        if (!trainMode && firstDetection) {
            if (anomalyInRange)
                numTruePositives++;
            else
                numFalsePositives++;
        }

        prevAnomalous = sustainedAnomaly;

        // Rendering
        sf::Sprite s;

        s.setTexture(rt.getTexture());

        float scale = window.getSize().y / static_cast<float>(bottomHeight);

        s.setScale(scale, scale);

        window.draw(s);

        // Show pool chain
        const float miniChainSpacing = 24.0f;

        for (int i = 0; i < imgPool.size(); i++) {
            sf::RectangleShape rs;
            rs.setPosition(i * miniChainSpacing + 4.0f, 4.0f);

            sf::Text text;
            text.setFont(tickFont);
            text.setCharacterSize(24);
            text.setString(std::to_string(imgLabels[i]));

            text.setPosition(rs.getPosition() + sf::Vector2f(2.0f, -2.0f));

            rs.setSize(sf::Vector2f(miniChainSpacing, miniChainSpacing));

            if (firstDetection)
                rs.setFillColor(sf::Color::Red);
            else
                rs.setFillColor(sf::Color::Green);

            window.draw(rs);
            window.draw(text);
        }

        // Show statistics
        {
            sf::Text text;
            text.setFont(tickFont);
            text.setCharacterSize(16);
            text.setString("Total samples: " + std::to_string(totalSamples));

            text.setPosition(4.0f, 32.0f);

            window.draw(text);

            text.setString("True Positives: " + std::to_string(numTruePositives));

            text.setPosition(4.0f, 32.0f + 32.0f);

            window.draw(text);

            text.setString("False Positives: " + std::to_string(numFalsePositives));

            text.setPosition(4.0f, 32.0f + 32.0f + 32.0f);

            window.draw(text);
        }

        // Render plot
        plotRT.clear(sf::Color::White);

        plot.draw(plotRT, lineGradient, tickFont, 0.5f, sf::Vector2f(0.0f, bottomWidth * overSizeMult), sf::Vector2f(-1.0f, 2.0f), sf::Vector2f(50.0f, 50.0f), sf::Vector2f(50.0f, 4.0f), 2.0f, 2.0f, 2.0f, 5.0f, 4.0f, 3);

        plotRT.display();

        // Anomaly flag
        if (sustainedAnomaly > 0.0f) {
            sf::RectangleShape rs;

            rs.setFillColor(sf::Color::Red);

            rs.setSize(sf::Vector2f(512.0f, 12.0f));

            window.draw(rs);
        }

        // Show plot
        sf::Sprite plotS;
        plotS.setTexture(plotRT.getTexture());

        plotS.setPosition(512.0f, 0.0f);

        window.draw(plotS);

        window.display();
    }

    return 0;
}