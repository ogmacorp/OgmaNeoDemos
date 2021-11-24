#include "Graph.h"

#include <iostream>

std::mt19937 rng;

float getSimilarity(
    const CSDR &left,
    const CSDR &right
) {
    float sim = 0.0f;
    
    for (int i = 0; i < left.size(); i++) {
        if (left[i] == right[i])
            sim += 1.0f;
    }

    sim /= std::max<int>(1, left.size());

    return sim;
}

float getSimilarity(
    const Matrix4x4f &left,
    const Matrix4x4f &right
) {
    float sim = 0.0f;
    
    for (int i = 0; i < 16; i++) {
        float d = left.elements[i] - right.elements[i];

        sim -= d * d;
    }

    sim /= 16.0f;

    return sim;
}

void merge(
    const CSDR &left,
    const CSDR &right,
    CSDR &result
) {
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

    if (result.size() != left.size())
        result.resize(left.size());

    for (int i = 0; i < left.size(); i++) {
        if (dist01(rng) < 0.5f)
            result[i] = left[i];
        else
            result[i] = right[i];
    }
}

void toward(
    const CSDR &target,
    QuasiCSDR &qcsdr,
    float rate
) {
    int columnSize = qcsdr.size() / target.size();

    for (int i = 0; i < target.size(); i++) {
        for (int j = 0; j < columnSize; j++) {
            qcsdr[j + columnSize * i] += rate * ((j == target[i] ? 1.0f : 0.0f) - qcsdr[j + columnSize * i]);
        }
    }
}

void inhibit(
    const QuasiCSDR &qcsdr,
    CSDR &csdr
) {
    assert(!csdr.empty());

    int columnSize = qcsdr.size() / csdr.size();

    for (int i = 0; i < csdr.size(); i++) {
        int maxIndex = 0;
        float maxActivation = -999999.0f;

        for (int j = 0; j < columnSize; j++) {
            if (qcsdr[j + columnSize * i] > maxActivation) {
                maxActivation = qcsdr[j + i * columnSize];
                maxIndex = j;
            }
        }

        csdr[i] = maxIndex;
    }
}

void initQuasiCSDR(
    const CSDR &csdr,
    QuasiCSDR &qcsdr
) {
    assert(!csdr.empty());

    int columnSize = qcsdr.size() / csdr.size();

    for (int i = 0; i < csdr.size(); i++) {
        for (int j = 0; j < columnSize; j++) {
            qcsdr[j + columnSize * i] = (csdr[i] == j ? 1.0f : 0.0f);
        }
    }
}

void Graph::init(
    int columnSize
) {
    this->columnSize = columnSize;

    estimated.setIdentity();

    accumDelta.setIdentity();
}

void Graph::step(
    const Matrix4x4f &delta,
    const CSDR &csdr
) {
    int oldLastN = lastN;

    estimated *= delta;

    accumDelta *= delta;

    // Update nodes based on closeness to estimated position
    CSDR temp(csdr.size());

    int maxN = 0;
    float maxSim = 0.0f;

    for (int n = 0; n < nodes.size(); n++) {
        inhibit(nodes[n].qcsdr, temp);

        float sim = getSimilarity(csdr, temp) * (1.0f - transWeight) + std::exp(getSimilarity(estimated, nodes[n].trans) * transScale) * transWeight;

        if (sim > maxSim) {
            maxSim = sim;
            maxN = n;
        }
    }

    // Find node with closest position top estimated position
    int minN = 0;
    float minDist = 999999.0f;

    Vec3f estimatedPos = estimated * Vec3f(0.0f, 0.0f, 0.0f);

    for (int n = 0; n < nodes.size(); n++) {
        Vec3f pos = nodes[n].trans * Vec3f(0.0f, 0.0f, 0.0f);

        float dist = (estimatedPos - pos).magnitude();

        if (dist < minDist) {
            minDist = dist;
            minN = n;
        }
    }

    if (lastN == -1 || maxSim < addThresh) {
        // Add new node
        Node node;
        node.trans = estimated;
        node.qcsdr.resize(csdr.size() * columnSize);

        initQuasiCSDR(csdr, node.qcsdr);

        if (lastN != -1) {
            Connection conn;
            conn.ni = lastN;

            conn.relTrans = accumDelta;

            node.connections.push_back(conn);
        }

        nodes.push_back(node);

        lastN = nodes.size() - 1;
    }
    else {
        // Check if connection already exists
        int usedC = -1;

        for (int c = 0; c < nodes[maxN].connections.size(); c++) {
            if (nodes[maxN].connections[c].ni == lastN) {
                usedC = c;

                break;
            }
        }

        if (usedC == -1) {
            if (maxN != lastN) {
                // Add new connection
                Connection conn;
                conn.ni = lastN;

                conn.relTrans = accumDelta;

                nodes[maxN].connections.push_back(conn);
            }
        }
        else {
            // Update relative transform
            for (int i = 0; i < 16; i++)
                nodes[maxN].connections[usedC].relTrans.elements[i] += (accumDelta.elements[i] - nodes[maxN].connections[usedC].relTrans.elements[i]) * elasticity;
        }

        // Tweak existing nodes around max node
        for (int n = 0; n < nodes.size(); n++) {
            Matrix4x4f transNNew;
            
            for (int i = 0; i < 16; i++)
                transNNew.elements[i] = 0.0f;

            for (int c = 0; c < nodes[n].connections.size(); c++) {
                Matrix4x4f predTransNNew = nodes[nodes[n].connections[c].ni].trans * nodes[n].connections[c].relTrans;

                for (int i = 0; i < 16; i++)
                    transNNew.elements[i] += predTransNNew.elements[i];
            }

            for (int i = 0; i < 16; i++)
                transNNew.elements[i] /= std::max<int>(1, nodes[n].connections.size());

            // Learn matrix (not the best way of doing this especially for rotation but whatever)
            for (int i = 0; i < 16; i++)
                nodes[n].trans.elements[i] += (transNNew.elements[i] - nodes[n].trans.elements[i]) * elasticity;
        }
        
        // Learn matrix (not the best way of doing this especially for rotation but whatever)
        for (int i = 0; i < 16; i++)
            estimated.elements[i] += (nodes[maxN].trans.elements[i] - estimated.elements[i]) * drift * maxSim * maxSim;

        lastN = maxN;
    }

    // Update QuasiCSDR
    toward(csdr, nodes[lastN].qcsdr, drift * maxSim * maxSim);

    if (oldLastN != lastN)
        accumDelta.setIdentity();
}

void Graph::renderXY(
    sf::RenderTarget &rt,
    float renderScale
) {
    // Compute all positions of transforms
    std::vector<Vec3f> positions(nodes.size());

    for (int n = 0; n < nodes.size(); n++)
        positions[n] = nodes[n].trans * Vec3f(0.0f, 0.0f, 0.0f);

    sf::VertexArray lines;

    lines.setPrimitiveType(sf::Lines);

    for (int n = 0; n < nodes.size(); n++) {
        for (int c = 0; c < nodes[n].connections.size(); c++) {
            sf::Vertex start;

            start.position = sf::Vector2f(positions[n].x, positions[n].y) * renderScale;
            start.color = sf::Color::Red;

            sf::Vertex end;

            end.position = sf::Vector2f(positions[nodes[n].connections[c].ni].x, positions[nodes[n].connections[c].ni].y) * renderScale;
            end.color = sf::Color::Red;

            lines.append(start);
            lines.append(end);
        }
    }

    rt.draw(lines);
    
    sf::CircleShape nodeShape;
    const float nodeRad = 0.005f * renderScale;
    nodeShape.setRadius(nodeRad);
    nodeShape.setOrigin(nodeRad, nodeRad);
    nodeShape.setFillColor(sf::Color::Green);

    for (int n = 0; n < nodes.size(); n++) {
        nodeShape.setPosition(sf::Vector2f(positions[n].x, positions[n].y) * renderScale);

        if (n == lastN) {
            nodeShape.setFillColor(sf::Color::Blue);
            
            rt.draw(nodeShape);

            nodeShape.setFillColor(sf::Color::Green);
        }
        else
            rt.draw(nodeShape);
    }
}
