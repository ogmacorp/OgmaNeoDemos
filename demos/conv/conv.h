#pragma once

#include <aogmaneo/helpers.h>

using namespace aon;

class ConvAE {
private:
    Int3 input_size;

    int filter_size_enc0;
    int pool_size_enc0;
    int filter_size_enc1;
    int filter_size_dec;

    Byte_Buffer filter_enc0;
    Byte_Buffer filter_enc1;
    Byte_Buffer filter_dec;

public:
    void init_random(
        const Int3 &input_size,
        int filter_size_enc0 = 2,
        int pool_size_enc0 = 2,
        int filter_size_enc1 = 2,
        int filter_size_dec = 2
    );
};
