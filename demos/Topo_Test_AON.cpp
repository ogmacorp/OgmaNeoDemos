#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>

#include "constructs/Matrix3x3f.h"

#include <aogmaneo/encoder.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <random>

using namespace aon;

class Enc {
public:
    int numColumns;
    int numCells;
    int numInputs;
    std::vector<float> protos;
    std::vector<int> hiddenCIs;
    std::vector<float> hiddenActs;
    std::vector<float> hiddenRates;

    float vigilance;
    float lr;
    float boost;

    void initRandom(int numInputs, int numColumns, int numCells, std::mt19937 &rng) {
        this->numColumns = numColumns;
        this->numCells = numCells;
        this->numInputs = numInputs;

        protos.resize(numColumns * numCells * numInputs);

        std::normal_distribution<float> protoDist(0.0f, 1.0f);

        for (int i = 0; i < protos.size(); i++)
            protos[i] = protoDist(rng) * 0.5f;

        hiddenCIs = std::vector<int>(numColumns, 0);
        hiddenActs = std::vector<float>(numColumns, 0.0f);
        hiddenRates = std::vector<float>(numColumns * numCells, 1.0f);

        lr = 0.1f;
        boost = 0.001f;
        vigilance = 0.005f;
    }

    void step(const std::vector<float> &inputs, bool learnEnabled) {
        // Activate columns
        int maxOverallC = 0;
        float maxOverallActivation = -999999.0f;

        for (int c = 0; c < numColumns; c++) {
            int maxIndex = 0;
            float maxActivation = -999999.0f;

            for (int cc = 0; cc < numCells; cc++) {
                float sum = 0.0f;

                for (int i = 0; i < numInputs; i++) {
                    float delta = inputs[i] - protos[i + numInputs * (cc + numCells * c)];

                    sum -= delta * delta;
                }

                if (sum > maxActivation) {
                    maxActivation = sum;
                    maxIndex = cc;
                }
            }

            hiddenCIs[c] = maxIndex;
            hiddenActs[c] = maxActivation;

            if (maxActivation > maxOverallActivation) {
                maxOverallActivation = maxActivation;
                maxOverallC = c;
            }
        }

        if (learnEnabled) {
            // Learn only max overall strongly, others weakly
            for (int c = 0; c < numColumns; c++) {
                if (-hiddenActs[c] < vigilance)
                    continue;

                for (int dcc = -1; dcc <= 1; dcc++) {
                    int cc = hiddenCIs[c] + dcc;

                    if (cc < 0 || cc >= numCells)
                        continue;

                    float rate = hiddenRates[cc + numCells * c];// * std::exp(-falloff * diff * diff / std::max(0.0001f, hiddenRates[cc + numCells * c]));

                    if (c != maxOverallC)
                        rate *= boost;

                    for (int i = 0; i < numInputs; i++) {
                        int pri = i + numInputs * (cc + numCells * c);

                        float delta = inputs[i] - protos[pri];

                        protos[pri] += rate * delta;
                    }

                    hiddenRates[cc + numCells * c] -= lr * rate;
                }
            }
        }
    }
};

int main() {
    unsigned int windowWidth = 1280;
    unsigned int windowHeight = 720;

    sf::RenderWindow window;

    window.create(sf::VideoMode(windowWidth, windowHeight), "STDP Demo", sf::Style::Default);

    //window.setVerticalSyncEnabled(true);
    window.setFramerateLimit(60);

    std::mt19937 rng(time(nullptr));

    // Generate data
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::normal_distribution<float> nDist(0.0f, 1.0f);

    int resolution = 64;
    int numClusters = 8;
    int pointsPerCluster = 100;

    std::vector<Vec2f> data(numClusters * pointsPerCluster);

    for (int c = 0; c < numClusters; c++) {
        Vec2f pos(dist01(rng) * 2.0f - 1.0f, dist01(rng) * 2.0f - 1.0f);

        pos *= 0.9f;

        Matrix3x3f transform = Matrix3x3f::translateMatrix(pos) * Matrix3x3f::rotateMatrix(dist01(rng) * 3.1415f * 2.0f) * Matrix3x3f::scaleMatrix(Vec2f(dist01(rng), dist01(rng)));

        for (int p = 0; p < pointsPerCluster; p++) {
            Vec2f randVec(nDist(rng), nDist(rng));

            randVec *= 0.1f;

            Vec2f point = transform * randVec;

            data[p + pointsPerCluster * c] = point;
        }
    }

    std::uniform_int_distribution<int> dataDist(0, data.size() - 1);
    std::uniform_int_distribution<int> clusterDist(0, data.size() / numClusters - 1);

    aon::set_num_threads(8);

    // Encoder
    Array<Encoder::Visible_Layer_Desc> vlds(1);
    vlds[0].size = Int3(1, 2, resolution);

    Encoder e;
    e.init_random(Int3(4, 4, 16), vlds);
    Encoder::Params params;
    
    Int_Buffer inputs(2);

    Array<Int_Buffer_View> input_cis(1);
    input_cis[0] = inputs;

    bool quit = false;

    sf::View view = window.getDefaultView();

    const float dt = 0.017f;
    const float zoomRate = 0.4f;
    const float viewInterpolateRate = 20.0f;

    view.setCenter(0.0f, 0.0f);
    view.zoom(0.01f);

    sf::View newView = view;

    window.setView(view);

    std::cout << "Ready." << std::endl;

    std::vector<sf::Color> pallette1(numClusters);

    for (int i = 0; i < pallette1.size(); i++)
        pallette1[i] = sf::Color(dist01(rng) * 255.0f, dist01(rng) * 255.0f, dist01(rng) * 255.0f, 255);

    std::vector<sf::Color> pallette2(e.get_hidden_cis().size());

    for (int i = 0; i < pallette2.size(); i++)
        pallette2[i] = sf::Color(dist01(rng) * 255.0f, dist01(rng) * 255.0f, dist01(rng) * 255.0f, 255);

    int currentCluster = 0;

    do {
        sf::Event event;

        while (window.pollEvent(event)) {
            if (window.hasFocus()) {
                switch (event.type) {
                case sf::Event::Closed:
                    quit = true;
                    break;
                case sf::Event::MouseWheelMoved:
                    int dWheel = event.mouseWheel.delta;

                    newView = view;

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta0 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    newView.setSize(view.getSize() + newView.getSize() * (-zoomRate * dWheel));

                    window.setView(newView);

                    sf::Vector2f mouseZoomDelta1 = window.mapPixelToCoords(sf::Mouse::getPosition(window));

                    window.setView(view);

                    newView.setCenter(view.getCenter() + mouseZoomDelta0 - mouseZoomDelta1);

                    break;
                }
            }
        }

        if (window.hasFocus()) {
            if (sf::Keyboard::isKeyPressed(sf::Keyboard::Escape))
                quit = true;

            for (int k = 0; k < 10; k++) {
                if (sf::Keyboard::isKeyPressed(static_cast<sf::Keyboard::Key>(static_cast<int>(sf::Keyboard::Num0) + k)))
                    currentCluster = k;
            }

            currentCluster = std::min(numClusters - 1, currentCluster);
        }

        int iters = 1;

        if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space))
            iters = 100;

        for (int it = 0; it < iters; it++) {
            int dataIndex = clusterDist(rng) + currentCluster * pointsPerCluster;

            Vec2f point = data[dataIndex];

            inputs[0] = std::min(1.0f, std::max(0.0f, point.x * 0.5f + 0.5f)) * (resolution - 1) + 0.5f;
            inputs[1] = std::min(1.0f, std::max(0.0f, point.y * 0.5f + 0.5f)) * (resolution - 1) + 0.5f;

            e.step(input_cis, true, params);
        }

        view.setCenter(view.getCenter() + (newView.getCenter() - view.getCenter()) * viewInterpolateRate * dt);
        view.setSize(view.getSize() + (newView.getSize() - view.getSize()) * viewInterpolateRate * dt);

        window.setView(view);

        window.clear(sf::Color::Black);

        // Render datapoints
        sf::CircleShape cs;
        cs.setRadius(0.3f);
        cs.setOrigin(cs.getRadius(), cs.getRadius());

        for (int i = 0; i < data.size(); i++) {
            cs.setFillColor(pallette1[i / pointsPerCluster]);

            cs.setPosition(data[i].x * resolution, -data[i].y * resolution);

            window.draw(cs);
        }

        cs.setOutlineThickness(0.005f);
        cs.setOutlineColor(sf::Color::White);
        cs.setFillColor(sf::Color::Transparent);

        // Draw SOMs as lines
        Int3 hidden_size = e.get_hidden_size();
        int num_columns = e.get_hidden_cis().size();
        int num_cells_per_column = e.get_hidden_size().z;

        for (int c = 0; c < num_columns; c++) {
            Int2 columnPos(c / hidden_size.y, c % hidden_size.y);

            sf::VertexArray va(sf::LineStrip, num_cells_per_column);
            sf::Color color = pallette2[c];

            sf::CircleShape cs;
            cs.setFillColor(color);
            cs.setRadius(0.1f);
            cs.setOrigin(cs.getRadius(), cs.getRadius());

            for (int cc = 0; cc < num_cells_per_column; cc++) {
                int hidden_cell_index = cc + c * hidden_size.z;

                Encoder::Visible_Layer &vl = e.get_visible_layer(0);
                const Encoder::Visible_Layer_Desc &vld = e.get_visible_layer_desc(0);

                int diam = vld.radius * 2 + 1;
     
                // Projection
                Float2 hToV = Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
                    static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

                Int2 visibleCenter = project(columnPos, hToV);

                // Lower corner
                Int2 fieldLowerBound(visibleCenter.x - vld.radius, visibleCenter.y - vld.radius);

                // Bounds of receptive field, clamped to input size
                Int2 iterLowerBound(max(0, fieldLowerBound.x), max(0, fieldLowerBound.y));
                Int2 iterUpperBound(min(vld.size.x - 1, visibleCenter.x + vld.radius), min(vld.size.y - 1, visibleCenter.y + vld.radius));

                sf::Vector2f coord;
                int coordIndex = 0;

                for (int ix = iterLowerBound.x; ix <= iterUpperBound.x; ix++)
                    for (int iy = iterLowerBound.y; iy <= iterUpperBound.y; iy++) {
                        int visibleColumnIndex = address2(Int2(ix, iy), Int2(vld.size.x, vld.size.y));

                        Int2 offset(ix - fieldLowerBound.x, iy - fieldLowerBound.y);

                        int wi = cc + hidden_size.z * (offset.y + diam * (offset.x + diam * c));
                        //int wi = offset.y + diam * (offset.x + diam * hidden_cell_index);

                        float p = vl.means[wi];

                        if (coordIndex == 0)
                            coord.x = (p * 2.0f - 1.0f) * vld.size.z;
                        else if (coordIndex == 1)
                            coord.y = -(p * 2.0f - 1.0f) * vld.size.z;

                        coordIndex++;
                    }

                va[cc].position = coord;
                va[cc].color = color;

                cs.setPosition(coord);
                window.draw(cs);
            }

            window.draw(va);
        }
        
        window.display();
    } while (!quit);

    return 0;
}
