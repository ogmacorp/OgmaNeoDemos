#include "conv.h"

void ConvAE::init_random(
    const Int3 &input_size,
    int filter_size_enc0,
    int pool_size_enc0,
    int filter_size_enc1,
    int filter_size_dec
) {
     this->input_size = input_size;
     this->filter_size_enc0 = filter_size_enc0;
     this->pool_size_enc0 = pool_size_enc0;
     this->filter_size_enc1 = filter_size_enc1;
     this->filter_size_dec = filter_size_dec;
}
