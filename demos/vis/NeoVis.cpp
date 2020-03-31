#include "NeoVis.h"

#include <iostream>

const int fieldNameSize = 64;

std::vector<float> getReceptiveField(
    ComputeSystem &cs,
    const ImageEncoder &enc,
    int i,
    const Int3 &hiddenPosition,
    Int3 &size
) {
    // Determine bounds
    Int3 minPos(999999, 999999, 999999);
    Int3 maxPos(0, 0, 0);

    const SparseMatrix &sm = enc.getVisibleLayer(i).weights;

    int row = address3(Int3(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z), enc.getHiddenSize());
    //int nextIndex = row + 1;

    std::vector<int> js(2);
    js[0] = sm.rowRanges[row];
    js[1] = sm.rowRanges[row + 1];

    int numValues = js[1] - js[0];

    if (numValues == 0)
        return {};
  
    std::vector<int> columnIndices(numValues);
    std::vector<float> nonZeroValues(numValues);

    for (int i = 0; i < numValues; i++) {
        columnIndices[i] = sm.columnIndices[js[0] + i];
        nonZeroValues[i] = sm.nonZeroValues[js[0] + i];
    }
    
	for (int j = js[0]; j < js[1]; j++) {
        int index = columnIndices[j - js[0]];

        int inZ = index % enc.getVisibleLayerDesc(i).size.z;
        index /= enc.getVisibleLayerDesc(i).size.z;

        int inY = index % enc.getVisibleLayerDesc(i).size.y;
        index /= enc.getVisibleLayerDesc(i).size.y;

        int inX = index % enc.getVisibleLayerDesc(i).size.x;

		minPos.x = std::min(minPos.x, inX);
		minPos.y = std::min(minPos.y, inY);
		minPos.z = std::min(minPos.z, inZ);

        maxPos.x = std::max(maxPos.x, inX + 1);
		maxPos.y = std::max(maxPos.y, inY + 1);
		maxPos.z = std::max(maxPos.z, inZ + 1);
    }

    size.x = maxPos.x - minPos.x;
    size.y = maxPos.y - minPos.y;
    size.z = maxPos.z - minPos.z;

    int totalSize = size.x * size.y * size.z;
    
    std::vector<float> field(totalSize, 0.0f);

    for (int j = js[0]; j < js[1]; j++) {
        int index = columnIndices[j - js[0]];

        int inZ = index % enc.getVisibleLayerDesc(i).size.z;
        index /= enc.getVisibleLayerDesc(i).size.z;

        int inY = index % enc.getVisibleLayerDesc(i).size.y;
        index /= enc.getVisibleLayerDesc(i).size.y;

        int inX = index % enc.getVisibleLayerDesc(i).size.x;

		field[address3(Int3(inX - minPos.x, inY - minPos.y, inZ - minPos.z), Int3(size.x, size.y, size.z))] = nonZeroValues[j - js[0]];
    }

    return field;
}

std::vector<float> getSCReceptiveField(
    ComputeSystem &cs,
    const Hierarchy &h,
    int l,
    int i,
    const Int3 &hiddenPosition,
    Int3 &size
) {
    // Determine bounds
    Int3 minPos(999999, 999999, 999999);
    Int3 maxPos(0, 0, 0);

    const SparseMatrix &sm = h.getSCLayer(l).getVisibleLayer(i).weights;

    int row = address3(Int3(hiddenPosition.x, hiddenPosition.y, hiddenPosition.z), h.getSCLayer(l).getHiddenSize());
    //int nextIndex = row + 1;

    std::vector<int> js(2);
    js[0] = sm.rowRanges[row];
    js[1] = sm.rowRanges[row + 1];

    int numValues = js[1] - js[0];

    if (numValues == 0)
        return {};
  
    std::vector<int> columnIndices(numValues);
    std::vector<float> nonZeroValues(numValues);

    for (int i = 0; i < numValues; i++) {
        columnIndices[i] = sm.columnIndices[js[0] + i];
        nonZeroValues[i] = sm.nonZeroValues[js[0] + i];
    }

	for (int j = js[0]; j < js[1]; j++) {
        int index = columnIndices[j - js[0]];

        int inZ = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.z;
        index /= h.getSCLayer(l).getVisibleLayerDesc(i).size.z;

        int inY = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.y;
        index /= h.getSCLayer(l).getVisibleLayerDesc(i).size.y;

        int inX = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.x;

		minPos.x = std::min(minPos.x, inX);
		minPos.y = std::min(minPos.y, inY);
		minPos.z = std::min(minPos.z, inZ);

        maxPos.x = std::max(maxPos.x, inX + 1);
		maxPos.y = std::max(maxPos.y, inY + 1);
		maxPos.z = std::max(maxPos.z, inZ + 1);
    }

    size.x = maxPos.x - minPos.x;
    size.y = maxPos.y - minPos.y;
    size.z = maxPos.z - minPos.z;

    int totalSize = size.x * size.y * size.z;
    
    std::vector<float> field(totalSize, 0.0f);

    for (int j = js[0]; j < js[1]; j++) {
        int index = columnIndices[j - js[0]];

        int inZ = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.z;
        index /= h.getSCLayer(l).getVisibleLayerDesc(i).size.z;

        int inY = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.y;
        index /= h.getSCLayer(l).getVisibleLayerDesc(i).size.y;

        int inX = index % h.getSCLayer(l).getVisibleLayerDesc(i).size.x;

		field[address3(Int3(inX - minPos.x, inY - minPos.y, inZ - minPos.z), Int3(size.x, size.y, size.z))] = nonZeroValues[j - js[0]];
    }

    return field;
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

VisAdapter::VisAdapter(unsigned short port) {
    listener.setBlocking(false);

    listener.listen(port);
}

void VisAdapter::update(ComputeSystem &cs, const Hierarchy &h, const std::vector<const ImageEncoder*> &encs) {
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
        size_t totalReceived = 0;

        bool disconnected = false;

        if (clients[i]->receive(data.data(), data.size(), size) == sf::Socket::Done) {
            // --------------------------- Receive ----------------------------

            totalReceived += size;
            
            while (totalReceived < data.size()) {
                if (clients[i]->receive(&data[totalReceived], data.size() - totalReceived, size) == sf::Socket::Done)
                    totalReceived += size;
                else
                    sf::sleep(sf::seconds(0.001f));
            }

            caret = *reinterpret_cast<Caret*>(data.data());

            // Read away remaining data
            while (clients[i]->receive(data.data(), data.size(), size) == sf::Socket::Done);

            // ----------------------------- Send -----------------------------

            data.clear();

            push<sf::Uint16>(data, static_cast<sf::Uint16>(h.getNumLayers() + encs.size()));
            push<sf::Uint16>(data, static_cast<sf::Uint16>(encs.size()));

            // Add encoder SDRs
            for (int j = 0; j < encs.size(); j++) {
                Int3 s = encs[j]->getHiddenSize();

                push<sf::Uint16>(data, static_cast<sf::Uint16>(s.x));
                push<sf::Uint16>(data, static_cast<sf::Uint16>(s.y));
                push<sf::Uint16>(data, static_cast<sf::Uint16>(s.z));

                for (int k = 0; k < encs[j]->getHiddenCs().size(); k++)
                    push<sf::Uint16>(data, static_cast<sf::Uint16>(encs[j]->getHiddenCs()[k]));
            }

            // Add layer SDRs
            for (int j = 0; j < h.getNumLayers(); j++) {
                Int3 s = h.getSCLayer(j).getHiddenSize();

                push<sf::Uint16>(data, static_cast<sf::Uint16>(s.x));
                push<sf::Uint16>(data, static_cast<sf::Uint16>(s.y));
                push<sf::Uint16>(data, static_cast<sf::Uint16>(s.z));
                
                for (int k = 0; k < h.getSCLayer(j).getHiddenCs().size(); k++)
                    push<sf::Uint16>(data, static_cast<sf::Uint16>(h.getSCLayer(j).getHiddenCs()[k]));
            }

            int numFields = 0;
            int layerIndex = 0;

            // If was initialized to value value
            if (caret.pos.x != -1) {
                layerIndex = caret.layer;
                
                if (layerIndex < encs.size()) {
                    int encIndex = layerIndex;

                    const ImageEncoder* enc = encs[encIndex];

                    bool inBounds = caret.pos.x >= 0 && caret.pos.y >= 0 && caret.pos.z >= 0 &&
                        caret.pos.x < enc->getHiddenSize().x && caret.pos.y < enc->getHiddenSize().y && caret.pos.z < enc->getHiddenSize().z;

                    numFields = inBounds ? enc->getNumVisibleLayers() : 0;
                }
                else {
                    bool inBounds = caret.pos.x >= 0 && caret.pos.y >= 0 && caret.pos.z >= 0 &&
                        caret.pos.x < h.getSCLayer(layerIndex - encs.size()).getHiddenSize().x && caret.pos.y < h.getSCLayer(layerIndex - encs.size()).getHiddenSize().y && caret.pos.z < h.getSCLayer(layerIndex - encs.size()).getHiddenSize().z;

                    numFields = inBounds ? h.getSCLayer(layerIndex - encs.size()).getNumVisibleLayers() : 0;
                }
            }

            push<sf::Uint16>(data, static_cast<sf::Uint16>(numFields));

            if (layerIndex < encs.size()) {
                int encIndex = layerIndex;

                const ImageEncoder* enc = encs[encIndex];

                for (int j = 0; j < numFields; j++) {
                    std::string fieldName = "field " + std::to_string(j);

                    size_t start = data.size();

                    add(data, fieldNameSize);

                    for (int k = 0; k < fieldNameSize; k++)
                        *reinterpret_cast<char*>(&data[start + k]) = (k < fieldName.length() ? fieldName[k] : '\0');

                    Int3 fieldSize;

                    std::vector<float> field = getReceptiveField(cs, *enc, j, Int3(caret.pos.x, caret.pos.y, caret.pos.z), fieldSize);

                    push<sf::Int32>(data, static_cast<sf::Int32>(fieldSize.x));
                    push<sf::Int32>(data, static_cast<sf::Int32>(fieldSize.y));
                    push<sf::Int32>(data, static_cast<sf::Int32>(fieldSize.z));
                
                    for (int k = 0; k < field.size(); k++)
                        push<float>(data, field[k]);
                }
            }
            else {
                for (int j = 0; j < numFields; j++) {
                    std::string fieldName = "field " + std::to_string(j);

                    size_t start = data.size();
                    
                    add(data, fieldNameSize);

                    for (int k = 0; k < fieldNameSize; k++)
                        *reinterpret_cast<char*>(&data[start + k]) = (k < fieldName.length() ? fieldName[k] : '\0');

                    Int3 fieldSize;

                    std::vector<float> field = getSCReceptiveField(cs, h, layerIndex - encs.size(), j, Int3(caret.pos.x, caret.pos.y, caret.pos.z), fieldSize);

                    push<sf::Int32>(data, static_cast<sf::Int32>(fieldSize.x));
                    push<sf::Int32>(data, static_cast<sf::Int32>(fieldSize.y));
                    push<sf::Int32>(data, static_cast<sf::Int32>(fieldSize.z));
                
                    for (int k = 0; k < field.size(); k++)
                        push<float>(data, field[k]);
                }
            }

            size_t index = 0;
            size_t totalSent = 0;

            while (totalSent < data.size()) {
                sf::TcpSocket::Status status = clients[i]->send(&data[totalSent], data.size() - totalSent, size);

                if (status != sf::Socket::Done) {
                    std::cout << "Client disconnected." << std::endl;

                    clients.erase(clients.begin() + i);

                    disconnected = true;
                }

                totalSent += size;
            }
        }
        else {
            std::cout << "Client disconnected." << std::endl;

            clients.erase(clients.begin() + i);

            disconnected = true;
        }

        if (!disconnected)
            i++;
    }
}