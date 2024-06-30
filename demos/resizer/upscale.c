#include "upscale.h"

void horizontal_upscale_bilinear_4f32(f32 src[], f32 dst[], i32 src_width, i32 src_height, i32 dst_width, i32 dst_height) {
    f32 ratio_x = (f32)(src_width - 1) / (f32)dst_width;
    f32 ratio_y = (f32)(src_height - 1) / (f32)dst_height;

    i32 dst_size = dst_width * dst_height;

    for (i32 dst_i = 0; dst_i < dst_size; dst_i++) {
        i32 dst_x = dst_i / dst_height;
        i32 dst_y = dst_i % dst_height;
        f32 src_x_f = dst_x * ratio_x;
        f32 src_y_f = dst_y * ratio_y;
        i32 src_x = (i32)src_x_f;
        i32 src_y = (i32)src_y_f;
        f32 interp_x = src_x_f - src_x;
        f32 interp_y = src_y_f - src_y;

        i32 dst_start = 4 * dst_i;

        i32 src_start00 = 4 * (src_y + src_height * src_x);
        i32 src_start01 = src_start00 + 4;
        i32 src_start10 = src_start00 + src_height * 4;
        i32 src_start11 = src_start10 + 4;

        __m128 ix = _mm_set1_ps(interp_x);
        __m128 ix1 = _mm_set1_ps(1.0f - interp_x);
        __m128 iy = _mm_set1_ps(interp_y);
        __m128 iy1 = _mm_set1_ps(1.0f - interp_y);

        __m128 p00, p01, p10, p11;
        p00 = _mm_load_ps(src + src_start00);
        p01 = _mm_load_ps(src + src_start01);
        p10 = _mm_load_ps(src + src_start10);
        p11 = _mm_load_ps(src + src_start11);

        p00 = _mm_add_ps(_mm_mul_ps(p00, ix1), _mm_mul_ps(p10, ix));
        p01 = _mm_add_ps(_mm_mul_ps(p01, ix1), _mm_mul_ps(p11, ix));

        p00 = _mm_add_ps(_mm_mul_ps(p00, iy1), _mm_mul_ps(p01, iy));

        _mm_storeu_ps(dst + dst_start, p00);
    }
}

