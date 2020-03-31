#pragma once

#include <SFML/System.hpp>
#include <SFML/Network.hpp>
#include <ogmaneo/Hierarchy.h>
#include <ogmaneo/ImageEncoder.h>

using namespace ogmaneo;

struct Caret {
    sf::Uint16 layer;
    sf::Vector3i pos;

    Caret()
    : layer(0),
    pos(-1, -1, -1)
    {}
};

class VisAdapter {
private:
    sf::TcpListener listener;

    std::vector<std::unique_ptr<sf::TcpSocket>> clients;

    Caret caret;

public:
    VisAdapter(unsigned short port = 54000);

    void update(ComputeSystem &cs, const Hierarchy &h, const std::vector<const ImageEncoder*> &encs);
};