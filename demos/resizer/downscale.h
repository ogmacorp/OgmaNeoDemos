void resize_nearest_4f32(f32 src[], f32 dst[], i32 src_width, i32 src_height, i32 dst_width, i32 dst_height) {
    f32 ratio_x = (f32)src_width / (f32)dst_width;
    f32 ratio_y = (f32)src_height / (f32)dst_height;

    i32 dst_size = dst_width * dst_height;

    for (i32 dst_x = 0; dst_x < dst_width; dst_x++) {
        for (i32 dst_y = 0; dst_y < dst_height; dst_y++) {
            i32 src_x = (i32)((dst_x + 0.5f) * ratio_x);
            i32 src_y = (i32)((dst_y + 0.5f) * ratio_y);

            memcpy(&dst[4 * (dst_y + dst_height * dst_x)], &src[4 * (src_y + src_height * src_x)], RGBA32F_SIZE);
        }
    }
}

void resize_bilinear_4f32(f32 src[], f32 dst[], i32 src_width, i32 src_height, i32 dst_width, i32 dst_height) {
    f32 ratio_x = (f32)src_width / (f32)dst_width;
    f32 ratio_y = (f32)src_height / (f32)dst_height;

    if (((size_t)src | (size_t)dst) & 0xf) {
        for (i32 dst_x = 0; dst_x < dst_width; dst_x++) {
            for (i32 dst_y = 0; dst_y < dst_height; dst_y++) {
                f32 src_x_f = (dst_x + 0.5f) * ratio_x;
                f32 src_y_f = (dst_y + 0.5f) * ratio_y;
                i32 src_x = (i32)(src_x_f + 0.5f) - 1;
                i32 src_y = (i32)(src_y_f + 0.5f) - 1;
                f32 interp_x = src_x_f - 0.5f - src_x;
                f32 interp_y = src_y_f - 0.5f - src_y;

                i32 dst_start = 4 * (dst_y + dst_height * dst_x);

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
    }
    else {
        for (i32 dst_x = 0; dst_x < dst_width; dst_x++) {
            for (i32 dst_y = 0; dst_y < dst_height; dst_y++) {
                f32 src_x_f = (dst_x + 0.5f) * ratio_x;
                f32 src_y_f = (dst_y + 0.5f) * ratio_y;
                i32 src_x = (i32)(src_x_f + 0.5f) - 1;
                i32 src_y = (i32)(src_y_f + 0.5f) - 1;
                f32 interp_x = src_x_f - 0.5f - src_x;
                f32 interp_y = src_y_f - 0.5f - src_y;

                i32 dst_start = 4 * (dst_y + dst_height * dst_x);

                i32 src_start00 = 4 * (src_y + src_height * src_x);
                i32 src_start01 = src_start00 + 4;
                i32 src_start10 = src_start00 + src_height * 4;
                i32 src_start11 = src_start10 + 4;

                __m128 ix = _mm_set1_ps(interp_x);
                __m128 ix1 = _mm_set1_ps(1.0f - interp_x);
                __m128 iy = _mm_set1_ps(interp_y);
                __m128 iy1 = _mm_set1_ps(1.0f - interp_y);

                __m128 p00, p01, p10, p11;
                p00 = _mm_loadu_ps(src + src_start00);
                p01 = _mm_loadu_ps(src + src_start01);
                p10 = _mm_loadu_ps(src + src_start10);
                p11 = _mm_loadu_ps(src + src_start11);

                p00 = _mm_add_ps(_mm_mul_ps(p00, ix1), _mm_mul_ps(p10, ix));
                p01 = _mm_add_ps(_mm_mul_ps(p01, ix1), _mm_mul_ps(p11, ix));

                p00 = _mm_add_ps(_mm_mul_ps(p00, iy1), _mm_mul_ps(p01, iy));

                _mm_storeu_ps(dst + dst_start, p00);
            }
        }
    }
}

