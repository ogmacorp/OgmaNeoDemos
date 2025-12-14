#include "vit.h"

void Self_Attention_Layer::init_random(
    int token_size,
    int num_tokens,
    int head_size,
    int num_heads
) {
    this->token_size = token_size;
    this->num_tokens = num_tokens;
    this->head_size = head_size;
    this->num_heads = num_heads;

    qkv.resize(3 * num_heads * head_size * num_tokens * token_size);

    for (int i = 0; i < qkv.size(); i++)
        qkv[i] = rand_normalf();

    queries = Float_Buffer(num_tokens * head_size, 0.0f);
    keys = Float_Buffer(num_tokens * head_size, 0.0f);
    values = Float_Buffer(num_tokens * head_size, 0.0f);
}

void Self_Attention_Layer::forward(
    Float_Buffer_View tokens
) {
    for (int h = 0; h < num_heads; h++) {
        queries.fill(0.0f);
        keys.fill(0.0f);
        values.fill(0.0f);

        PARALLEL_FOR
        for (int t = 0; t < num_tokens; t++) {
            for (int hi = 0; hi < head_size; hi++) {
                int out_index = hi + head_size * t;

                for (int ti = 0; ti < token_size; ti++) {

                    int qkv_start = 3 * (ti + token_size * (hi + head_size * t));

                    queries[out_index] += qkv[0 + qkv_start];
                    keys[out_index] += qkv[1 + qkv_start];
                    values[out_index] += qkv[2 + qkv_start];
                }
            }
        }

        // attention matrix
    }
}
