#pragma once

#include <aogmaneo/hierarchy.h>
#include <aogmaneo/image_encoder.h>

#include <fstream>

class File_Reader : public aon::Stream_Reader {
public:
    std::ifstream ins;

    void read(
        void* data,
        long len
    ) override;
};

class File_Writer : public aon::Stream_Writer {
public:
    std::ofstream outs;

    void write(
        const void* data,
        long len
    ) override;
};

