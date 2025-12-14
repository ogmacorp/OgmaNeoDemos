// ----------------------------------------------------------------------------
//  NeoVis
//  Copyright(c) 2017-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of NeoVis is licensed to you under the terms described
//  in the NEOVIS_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "visadapter.h"

#include <iostream>

const int field_name_size = 64;

void get_receptive_field(
    const Image_Encoder &enc,
    int vli,
    const Int3 &pos,
    std::vector<unsigned char> &field,
    Int3 &field_size
) {
    int num_visible_layers = enc.get_num_visible_layers();

    const aon::Image_Encoder::Visible_Layer &vl = enc.get_visible_layer(vli);
    const aon::Image_Encoder::Visible_Layer_Desc &vld = enc.get_visible_layer_desc(vli);

    const aon::Int3 &hidden_size = enc.get_hidden_size();

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    aon::Int2 column_pos(pos.x, pos.y);

    int hidden_column_index = aon::address2(column_pos, aon::Int2(hidden_size.x, hidden_size.y));
    int hidden_cells_start = hidden_size.z * hidden_column_index;

    // projection
    aon::Float2 h_to_v = aon::Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

    aon::Int2 visible_center = project(column_pos, h_to_v);

        // lower corner
    aon::Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

        // bounds of receptive field, clamped to input size
    aon::Int2 iter_lower_bound(aon::max(0, field_lower_bound.x), aon::max(0, field_lower_bound.y));
    aon::Int2 iter_upper_bound(aon::min(vld.size.x - 1, visible_center.x + vld.radius), aon::min(vld.size.y - 1, visible_center.y + vld.radius));

    int field_count = area * vld.size.z;

    field = std::vector<unsigned char>(field_count, 0);

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int visible_column_index = address2(aon::Int2(ix, iy), aon::Int2(vld.size.x, vld.size.y));

            aon::Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

            int wi_start_partial = vld.size.z * (offset.y + diam * (offset.x + diam * hidden_column_index));

            for (int vc = 0; vc < vld.size.z; vc++) {
                int wi = pos.z + hidden_size.z * (vc + wi_start_partial);

                field[vc + vld.size.z * (offset.y + diam * offset.x)] = vl.weights[wi];
            }
        }

    field_size = Int3(diam, diam, vld.size.z);
}

void get_encoder_receptive_field(
    const Hierarchy &h,
    int l,
    int vli,
    const Int3 &pos,
    std::vector<unsigned char> &field,
    Int3 &field_size
) {
    const aon::Encoder &enc = h.get_encoder(l);

    int num_visible_layers = enc.get_num_visible_layers();

    const aon::Int3 &hidden_size = enc.get_hidden_size();

    const aon::Encoder::Visible_Layer &vl = enc.get_visible_layer(vli);
    const aon::Encoder::Visible_Layer_Desc &vld = enc.get_visible_layer_desc(vli);

    int diam = vld.radius * 2 + 1;
    int area = diam * diam;

    aon::Int2 column_pos(pos.x, pos.y);

    int hidden_column_index = aon::address2(column_pos, aon::Int2(hidden_size.x, hidden_size.y));
    int hidden_cells_start = hidden_size.z * hidden_column_index;

    // projection
    aon::Float2 h_to_v = aon::Float2(static_cast<float>(vld.size.x) / static_cast<float>(hidden_size.x),
            static_cast<float>(vld.size.y) / static_cast<float>(hidden_size.y));

    aon::Int2 visible_center = project(column_pos, h_to_v);

        // lower corner
    aon::Int2 field_lower_bound(visible_center.x - vld.radius, visible_center.y - vld.radius);

        // bounds of receptive field, clamped to input size
    aon::Int2 iter_lower_bound(aon::max(0, field_lower_bound.x), aon::max(0, field_lower_bound.y));
    aon::Int2 iter_upper_bound(aon::min(vld.size.x - 1, visible_center.x + vld.radius), aon::min(vld.size.y - 1, visible_center.y + vld.radius));

    int hidden_stride = vld.size.z * area;

    int field_count = area * vld.size.z;

    field = std::vector<unsigned char>(field_count, 0);

    int hidden_cell_index = pos.z + hidden_cells_start;

    for (int ix = iter_lower_bound.x; ix <= iter_upper_bound.x; ix++)
        for (int iy = iter_lower_bound.y; iy <= iter_upper_bound.y; iy++) {
            int visible_column_index = address2(aon::Int2(ix, iy), aon::Int2(vld.size.x, vld.size.y));

            aon::Int2 offset(ix - field_lower_bound.x, iy - field_lower_bound.y);

            int wi_start = vld.size.z * (offset.y + diam * offset.x) + hidden_cell_index * hidden_stride;

            for (int vc = 0; vc < vld.size.z; vc++) {
                int wi = vc + wi_start;

                field[vc + vld.size.z * (offset.y + diam * offset.x)] = vl.weights[wi];
            }
        }

    field_size = Int3(diam, diam, vld.size.z);
}

void add(std::vector<unsigned char> &data, size_t size) {
    data.resize(data.size() + size);
}

template<class T>
void push(std::vector<unsigned char> &data, T value) {
    size_t start = data.size();

    add(data, sizeof(T));

    *reinterpret_cast<T*>(&data[start]) = value;
}

Vis_Adapter::Vis_Adapter(unsigned short port) {
    listener.setBlocking(false);

    listener.listen(port);
}

void Vis_Adapter::update(const Hierarchy &h, const std::vector<const Image_Encoder*> &encs) {
    // Check for new connections
    std::unique_ptr<sf::TcpSocket> socket = std::make_unique<sf::TcpSocket>();

    if (listener.accept(*socket) == sf::Socket::Done) {
        socket->setBlocking(false);

        clients.push_back(std::move(socket));

        std::cout << "Client connected from " << clients.back()->getRemoteAddress() << std::endl;
    }

    // Send data to clients
    for (int i = 0; i < clients.size();) {
        std::vector<unsigned char> data(16);

        size_t size;
        size_t total_received = 0;

        bool disconnected = false;

        if (clients[i]->receive(data.data(), data.size(), size) == sf::Socket::Done) {
            // --------------------------- Receive ----------------------------

            total_received += size;
            
            while (total_received < data.size()) {
                if (clients[i]->receive(&data[total_received], data.size() - total_received, size) == sf::Socket::Done)
                    total_received += size;
                else
                    sf::sleep(sf::seconds(0.001f));
            }

            caret = *reinterpret_cast<Caret*>(data.data());

            // Read away remaining data
            while (clients[i]->receive(data.data(), data.size(), size) == sf::Socket::Done);
        }

        // ----------------------------- Send -----------------------------

        data.clear();

        push<sf::Uint16>(data, static_cast<sf::Uint16>(h.get_num_layers() + encs.size()));
        push<sf::Uint16>(data, static_cast<sf::Uint16>(encs.size()));

        // Add encoder CSDRs
        for (int j = 0; j < encs.size(); j++) {
            Int3 s = encs[j]->get_hidden_size();

            push<sf::Uint16>(data, static_cast<sf::Uint16>(s.x));
            push<sf::Uint16>(data, static_cast<sf::Uint16>(s.y));
            push<sf::Uint16>(data, static_cast<sf::Uint16>(s.z));

            for (int k = 0; k < encs[j]->get_hidden_cis().size(); k++)
                push<sf::Uint16>(data, static_cast<sf::Uint16>(encs[j]->get_hidden_cis()[k]));
        }

        // Add layer SDRs
        for (int j = 0; j < h.get_num_layers(); j++) {
            Int3 s = h.get_encoder(j).get_hidden_size();

            push<sf::Uint16>(data, static_cast<sf::Uint16>(s.x));
            push<sf::Uint16>(data, static_cast<sf::Uint16>(s.y));
            push<sf::Uint16>(data, static_cast<sf::Uint16>(s.z));
            
            for (int k = 0; k < h.get_encoder(j).get_hidden_cis().size(); k++)
                push<sf::Uint16>(data, static_cast<sf::Uint16>(h.get_encoder(j).get_hidden_cis()[k]));
        }

        int num_fields = 0;
        int layer_index = 0;

        // If was initialized
        if (caret.pos.x != -1) {
            layer_index = caret.layer;
            
            if (layer_index < encs.size()) {
                int enc_index = layer_index;

                const Image_Encoder* enc = encs[enc_index];

                bool in_bounds = caret.pos.x >= 0 && caret.pos.y >= 0 && caret.pos.z >= 0 &&
                    caret.pos.x < enc->get_hidden_size().x && caret.pos.y < enc->get_hidden_size().y && caret.pos.z < enc->get_hidden_size().z;

                num_fields = in_bounds ? enc->get_num_visible_layers() : 0;
            }
            else {
                bool in_bounds = caret.pos.x >= 0 && caret.pos.y >= 0 && caret.pos.z >= 0 &&
                    caret.pos.x < h.get_encoder(layer_index - encs.size()).get_hidden_size().x && caret.pos.y < h.get_encoder(layer_index - encs.size()).get_hidden_size().y && caret.pos.z < h.get_encoder(layer_index - encs.size()).get_hidden_size().z;

                num_fields = in_bounds ? h.get_encoder(layer_index - encs.size()).get_num_visible_layers() : 0;
            }
        }

        push<sf::Uint16>(data, static_cast<sf::Uint16>(num_fields));

        if (layer_index < encs.size()) {
            int enc_index = layer_index;

            const Image_Encoder* enc = encs[enc_index];

            for (int j = 0; j < num_fields; j++) {
                std::string field_name = "field " + std::to_string(j);

                size_t start = data.size();

                add(data, field_name_size);

                for (int k = 0; k < field_name_size; k++)
                    *reinterpret_cast<char*>(&data[start + k]) = (k < field_name.length() ? field_name[k] : '\0');

                std::vector<unsigned char> field;
                Int3 field_size;

                get_receptive_field(*enc, j, Int3(caret.pos.x, caret.pos.y, caret.pos.z), field, field_size);

                push<sf::Int32>(data, static_cast<sf::Int32>(field_size.x));
                push<sf::Int32>(data, static_cast<sf::Int32>(field_size.y));
                push<sf::Int32>(data, static_cast<sf::Int32>(field_size.z));
            
                for (int k = 0; k < field.size(); k++)
                    push<unsigned char>(data, field[k]);
            }
        }
        else {
            for (int j = 0; j < num_fields; j++) {
                std::string field_name = "field " + std::to_string(j);

                size_t start = data.size();
                
                add(data, field_name_size);

                for (int k = 0; k < field_name_size; k++)
                    *reinterpret_cast<char*>(&data[start + k]) = (k < field_name.length() ? field_name[k] : '\0');

                std::vector<unsigned char> field;
                Int3 field_size;

                get_encoder_receptive_field(h, layer_index - encs.size(), j, Int3(caret.pos.x, caret.pos.y, caret.pos.z), field, field_size);

                push<sf::Int32>(data, static_cast<sf::Int32>(field_size.x));
                push<sf::Int32>(data, static_cast<sf::Int32>(field_size.y));
                push<sf::Int32>(data, static_cast<sf::Int32>(field_size.z));
            
                for (int k = 0; k < field.size(); k++)
                    push<unsigned char>(data, field[k]);
            }
        }

        size_t index = 0;
        size_t total_sent = 0;

        while (total_sent < data.size()) {
            sf::TcpSocket::Status status = clients[i]->send(&data[total_sent], data.size() - total_sent, size);

            if (status == sf::Socket::Disconnected) {
                std::cout << "Client disconnected." << std::endl;

                clients.erase(clients.begin() + i);

                disconnected = true;

                break;
            }

            total_sent += size;
        }
        
        if (!disconnected)
            i++;
    }
}
