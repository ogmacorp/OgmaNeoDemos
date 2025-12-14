// ----------------------------------------------------------------------------
//  NeoVis
//  Copyright(c) 2017-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of NeoVis is licensed to you under the terms described
//  in the NEOVIS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include <SFML/System.hpp>
#include <SFML/Network.hpp>
#include <aogmaneo/hierarchy.h>
#include <aogmaneo/image_encoder.h>
#include <vector>
#include <memory>

using namespace aon;

struct Caret {
    sf::Uint16 layer;
    sf::Vector3i pos;

    Caret()
    : layer(0),
    pos(-1, -1, -1)
    {}
};

class Vis_Adapter {
private:
    sf::TcpListener listener;

    std::vector<std::unique_ptr<sf::TcpSocket>> clients;

    Caret caret;

public:
    Vis_Adapter(unsigned short port = 54000);

    void update(const Hierarchy &h, const std::vector<const Image_Encoder*> &encs);
};
