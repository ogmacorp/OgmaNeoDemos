#pragma once

#include "helpers.h"

using namespace aon;

class Self_Attention_Layer {
private:
    int token_size;
    int num_tokens;
    int head_size;
    int num_heads;

    Float_Buffer qkv;

    Float_Buffer queries;
    Float_Buffer keys;
    Float_Buffer values;

public:
    void init_random(
        int token_size,
        int num_tokens,
        int head_size,
        int num_heads
    );

    void forward(
        Float_Buffer_View tokens
    );
};

