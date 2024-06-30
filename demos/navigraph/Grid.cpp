#include "Grid.h"

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

void Grid::init(
    int columnSize
) {
    this->columnSize = columnSize;

    estimated.setIdentity();
}

void Grid::step(
    const Matrix4x4f &delta,
    const CSDR &csdr
) {
    int oldLastN = lastN;

    estimated *= delta;

    // Update nodes based on closeness to estimated position
    CSDR temp(csdr.size());

    int maxN = -1;
    float maxSim = 0.0f;

    for (int n = 0; n < nodes.size(); n++) {
        inhibit(nodes[n].qcsdr, temp);

        float sim = getSimilarity(csdr, temp);

        if (sim > maxSim) {
            maxSim = sim;
            maxN = n;
        }
    }

    float dist = 0.0f;

    if (lastN != -1)
        dist = (estimated * Vec3f(0.0f, 0.0f, 0.0f) - nodes[lastN].trans * Vec3f(0.0f, 0.0f, 0.0f)).magnitude();

    if (lastN == -1 || dist >= minDist) {
        if (maxSim >= minSim) {
            // Check if connection already exists
            int usedC = -1;

            auto it = connections.find(Pair{ lastN, maxN });
            auto revIt = connections.find(Pair{ maxN, lastN });

            if (it == connections.end()) {
                assert(revIt == connections.end());

                if (maxN != lastN) {
                    Connection conn0;
                    Connection conn1;

                    Matrix4x4f inv;

                    nodes[lastN].trans.inverse(inv);

                    conn0.relTrans = inv * estimated;
                    conn0.relTrans.inverse(conn1.relTrans);

                    connections[Pair{ lastN, maxN }] = conn0;
                    connections[Pair{ maxN, lastN }] = conn1;
                }
            }
            else {
                assert(revIt != connections.end());

                if (maxN != lastN) {
                    Matrix4x4f inv;

                    nodes[lastN].trans.inverse(inv);

                    Matrix4x4f relTrans = inv * estimated;
                    Matrix4x4f relTransInv;
                    relTrans.inverse(relTransInv);

                    // Update relative transforms
                    for (int i = 0; i < 16; i++)
                        (*it).second.relTrans.elements[i] += (relTrans.elements[i] - (*it).second.relTrans.elements[i]) * drift;

                    for (int i = 0; i < 16; i++)
                        (*revIt).second.relTrans.elements[i] += (relTransInv.elements[i] - (*revIt).second.relTrans.elements[i]) * drift;
                }
            }
            
            lastN = maxN;
        }
        else {
            // Add new node
            Node node;
            node.trans = estimated;
            node.qcsdr.resize(csdr.size() * columnSize);

            initQuasiCSDR(csdr, node.qcsdr);

            if (lastN != -1) {
                Connection conn0;
                Connection conn1;

                Matrix4x4f inv;

                nodes[lastN].trans.inverse(inv);

                conn0.relTrans = inv * estimated;
                conn0.relTrans.inverse(conn1.relTrans);

                maxN = static_cast<int>(nodes.size());

                connections[Pair{ lastN, maxN }] = conn0;
                connections[Pair{ maxN, lastN }] = conn1;
            }

            nodes.push_back(node);

            lastN = nodes.size() - 1;
        }
    }

    // Relax graph
    if (nodes.size() > 1) {
        // Zero trans and count temps
        for (int n = 0; n < nodes.size(); n++) {
            for (int i = 0; i < 16; i++)
                nodes[n].transTemp.elements[i] = 0.0f;

            nodes[n].countTemp = 0;
        }

        // Absolute phase
        for (auto it = connections.begin(); it != connections.end(); it++) {
            Pair p = (*it).first;

            Matrix4x4f predTransNNew = nodes[p.n0].trans * (*it).second.relTrans;
            
            for (int i = 0; i < 16; i++)
                nodes[p.n1].transTemp.elements[i] += predTransNNew.elements[i];

            nodes[p.n1].countTemp++;
        }

        // Update (double buffer)
        for (int n = 0; n < nodes.size(); n++) {
            assert(nodes[n].countTemp > 0);

            float scale = 1.0f / nodes[n].countTemp;

            for (int i = 0; i < 16; i++)
                nodes[n].trans.elements[i] += (nodes[n].transTemp.elements[i] * scale - nodes[n].trans.elements[i]) * elasticity;
        }

        // Relative phase
        for (auto it = connections.begin(); it != connections.end(); it++) {
            Pair p = (*it).first;

            Matrix4x4f inv;

            nodes[p.n0].trans.inverse(inv);

            Matrix4x4f newRelTrans = inv * nodes[p.n1].trans;

            for (int i = 0; i < 16; i++)
                (*it).second.relTrans.elements[i] += (newRelTrans.elements[i] - (*it).second.relTrans.elements[i]) * relElasticity;
        }
    }

    if (lastN != -1) {
        // Update matrix (not the best way of doing this especially for rotation but whatever)
        for (int i = 0; i < 16; i++)
            estimated.elements[i] += (nodes[lastN].trans.elements[i] - estimated.elements[i]) * drift;

        // Update QuasiCSDR
        toward(csdr, nodes[lastN].qcsdr, drift);
    }
}

void Grid::findPath(
    int startIndex,
    int endIndex,
    std::vector<int> &path
) {
    if (!path.empty())
        path.clear();

    std::vector<float> dists(nodes.size(), 999999.0f);
    std::vector<int> prev(nodes.size(), -1);

    std::unordered_set<int> q;

    for (int v = 0; v < nodes.size(); v++)
        q.insert(v);

    dists[startIndex] = 0.0f;

    while (!q.empty()) {
        std::unordered_set<int>::iterator cit = q.begin();

        int u = *cit;
        float minDist = dists[u];
        
        cit++;

        for (; cit != q.end(); cit++) {
            if (dists[*cit] < minDist) {
                minDist = dists[*cit];
                u = *cit;
            }
        }

        if (u == endIndex) {
            path.push_back(u);

            while (prev[u] != -1) {
                path.push_back(prev[u]);
                u = prev[u];
            }

            return;
        }

        q.erase(u);

        cit = q.begin();

        for (; cit != q.end(); cit++) {
            float dist = 999999.0f;

            // Find connection
            Pair p{ *cit, u };

            auto it = connections.find(p);

            if (it != connections.end())
                dist = ((*it).second.relTrans * Vec3f(0.0f, 0.0f, 0.0f)).magnitude();

            float alt = dists[u] + dist; // Slight decay for transition cost
            
            if (alt < dists[*cit]) {
                dists[*cit] = alt;

                prev[*cit] = u;
            }
        }
    }
}

void Grid::renderXY(
    sf::RenderTarget &rt,
    float renderScale,
    const std::vector<int> &path
) {
    // Compute all positions of transforms
    std::vector<Vec3f> positions(nodes.size());

    for (int n = 0; n < nodes.size(); n++)
        positions[n] = nodes[n].trans * Vec3f(0.0f, 0.0f, 0.0f);

    sf::VertexArray lines;

    lines.setPrimitiveType(sf::Lines);

    for (auto it = connections.begin(); it != connections.end(); it++) {
        Pair p = (*it).first;

        sf::Vertex start;

        start.position = sf::Vector2f(positions[p.n0].x, positions[p.n0].y) * renderScale;
        start.color = sf::Color::Red;

        sf::Vertex end;

        end.position = sf::Vector2f(positions[p.n1].x, positions[p.n1].y) * renderScale;
        end.color = sf::Color::Red;

        lines.append(start);
        lines.append(end);
    }

    rt.draw(lines);

    lines.clear();

    for (int i = 0; i < static_cast<int>(path.size()) - 1; i++) {
        sf::Vertex start;

        start.position = sf::Vector2f(positions[path[i]].x, positions[path[i]].y) * renderScale;
        start.color = sf::Color::Green;

        sf::Vertex end;

        end.position = sf::Vector2f(positions[path[i + 1]].x, positions[path[i + 1]].y) * renderScale;
        end.color = sf::Color::Green;

        lines.append(start);
        lines.append(end);
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

