// What a Fern (gern) user writes by hand for the Figure 1 pipeline: the concrete data structures (views) and the
// black-box subroutines, in plain C++. The Fern generator (fern_pipeline_gen.cpp) only knows their annotations; the
// code it generates (generated/) includes this header, allocates and queries the data structures through the methods
// named in the annotations and calls the subroutines.
//
// FKL's conventions (common/bench_reference.h, Conventions::Fkl):
//   NV12 -> RGB float, BT.601 limited range, no clamping, chroma replicated (UV of (px, py) at (px/2, py/2)), FKL's
//   matrix (fk::ccMatrix<Limited, bt601, YCbCr2RGB, p8bit>, checked against FKL's by the benchmark at run time);
//   bilinear resize with fx = x * (w / 64), fy = y * (h / 32), clamped to [0, w - 1] / [0, h - 1];
//   16 black rows on top and bottom; out = (v * (1/255) - SUB[c]) / DIV[c] on every row (border rows too).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>

namespace fern_bvf {

// ---- Pipeline constants (the parameters a library call would take: resize size, border rows, normalization) ----
constexpr int64_t RES_W = 64, RES_H = 32;         // resize target
constexpr int64_t OUT_W = 64, OUT_H = 64;         // letterboxed plane
constexpr int64_t BORDER_TOP = (OUT_H - RES_H) / 2;
constexpr float MUL = 1.f / 255.f;
constexpr float SUB[3] = {0.485f, 0.456f, 0.406f};
constexpr float DIV[3] = {0.229f, 0.224f, 0.225f};
// FKL's BT.601 limited range 8-bit YCbCr -> RGB matrix, exact float values
constexpr float M[3][3] = {{1.164383531e+00f, 0.000000000e+00f, 1.596026659e+00f},
                           {1.164383531e+00f, -3.917622864e-01f, -8.129676580e-01f},
                           {1.164383531e+00f, 2.017231941e+00f, 0.000000000e+00f}};
constexpr float Y_SUB = 16.f, C_SUB = 128.f;

// ---- Scratch memory for the intermediates Fern allocates -------------------------------------------------------
// Fern's generated code calls <Type>::allocate(...) for every intermediate and <obj>.destroy() when it is dead. How
// that memory is obtained is up to the data structure: by default a per-thread stack arena (no malloc in the timed
// loop; a block freed on top of the stack is reused at once, so a per-band intermediate reuses the same cache lines
// in every band); -DFERN_BVF_MALLOC uses malloc/free like gern's own test data structures
// (test/library/matrix/impl/cpu-matrix.h).
namespace detail {
struct Arena {
    static constexpr size_t BYTES = size_t(1) << 20;
    static constexpr int MAX_BLOCKS = 64;
    alignas(64) unsigned char buf[BYTES];
    size_t start[MAX_BLOCKS];
    bool freed[MAX_BLOCKS];
    int blocks = 0;
    size_t top = 0;
};
inline Arena &arena() {
    static thread_local Arena a;
    return a;
}
inline float *alloc_floats(size_t n) {
#ifdef FERN_BVF_MALLOC
    return static_cast<float *>(std::malloc(n * sizeof(float)));
#else
    Arena &a = arena();
    const size_t bytes = (n * sizeof(float) + 63) & ~size_t(63);
    if (a.top + bytes > Arena::BYTES || a.blocks == Arena::MAX_BLOCKS) throw std::bad_alloc();
    a.start[a.blocks] = a.top;
    a.freed[a.blocks++] = false;
    float *p = reinterpret_cast<float *>(a.buf + a.top);
    a.top += bytes;
    return p;
#endif
}
inline void free_floats(float *p) {
#ifdef FERN_BVF_MALLOC
    std::free(p);
#else
    Arena &a = arena();
    const size_t off = size_t(reinterpret_cast<unsigned char *>(p) - a.buf);
    for (int b = a.blocks - 1; b >= 0; --b) {
        if (a.start[b] == off) {
            a.freed[b] = true;
            break;
        }
    }
    while (a.blocks > 0 && a.freed[a.blocks - 1]) a.top = a.start[--a.blocks];
#endif
}
} // namespace detail

// ---- Data structures ----------------------------------------------------------------------------------------------

// A rectangle of pixels of an NV12 frame (luma coordinates). query() returns a view that keeps the absolute origin
// (r0, c0), so the chroma of every pixel is found from its absolute position whatever the view's offset.
struct NV12View {
    const uint8_t *y;   // luma plane, row 0 col 0 of the frame
    const uint8_t *uv;  // interleaved chroma plane, row 0 of the frame
    int64_t pitch;      // bytes per row (both planes)
    int64_t r0, c0;     // absolute origin of the view
    int64_t row, col;   // extents
    NV12View query(int64_t r, int64_t c, int64_t l_r, int64_t l_c) const {
        return NV12View{y, uv, pitch, r0 + r, c0 + c, l_r, l_c};
    }
    void insert(int64_t, int64_t, int64_t, int64_t, NV12View) {}
    void destroy() {}
};

// The region of one call: run-time data that changes every iteration. Fern reads its fields in the annotations
// (rect.x, rect.y, rect.w, rect.h, ADTMember), never as scalars.
struct Rect {
    int64_t x, y, w, h;
    Rect query(int64_t) const { return *this; }
    void insert(int64_t, Rect) {}
    void destroy() {}
};

// A float3 (interleaved RGB) image or a view of one. ld is the row stride in pixels. (r0, c0) is the origin of the
// view in the image it belongs to: allocate() records the subset Fern asked for, query() accumulates offsets.
struct ImageF3 {
    float *data;
    int64_t row, col, ld;
    int64_t r0, c0;
    static ImageF3 allocate(int64_t r, int64_t c, int64_t l_r, int64_t l_c) {
        return ImageF3{detail::alloc_floats(size_t(l_r * l_c * 3)), l_r, l_c, l_c, r, c};
    }
    ImageF3 query(int64_t r, int64_t c, int64_t l_r, int64_t l_c) const {
        return ImageF3{data + (r * ld + c) * 3, l_r, l_c, ld, r0 + r, c0 + c};
    }
    // Queries are views (insertQuery() is false in the annotation), so Fern never calls insert
    void insert(int64_t, int64_t, int64_t, int64_t, ImageF3) {}
    void destroy() { detail::free_floats(data); }
    float *px(int64_t i, int64_t j) const { return data + (i * ld + j) * 3; }
};

// ---- Black-box subroutines ----------------------------------------------------------------------------------------

// NV12 -> RGB float of every pixel of `in` (same extents as `out`). Pointwise: any sub-rectangle can be converted on
// its own, so Fern may tile it freely; the crop is not a subroutine, it is the offset of the subset Fern queries.
inline void nv12_to_rgb(NV12View in, ImageF3 out) {
    for (int64_t i = 0; i < out.row; ++i) {
        const int64_t py = in.r0 + i;
        const uint8_t *yrow = in.y + py * in.pitch;
        const uint8_t *uvrow = in.uv + (py >> 1) * in.pitch;
        float *o = out.px(i, 0);
        for (int64_t j = 0; j < out.col; ++j) {
            const int64_t px = in.c0 + j;
            const float yf = float(yrow[px]) - Y_SUB;
            const float uf = float(uvrow[(px >> 1) * 2]) - C_SUB;
            const float vf = float(uvrow[(px >> 1) * 2 + 1]) - C_SUB;
            o[3 * j + 0] = yf * M[0][0] + uf * M[0][1] + vf * M[0][2];
            o[3 * j + 1] = yf * M[1][0] + uf * M[1][1] + vf * M[1][2];
            o[3 * j + 2] = yf * M[2][0] + uf * M[2][1] + vf * M[2][2];
        }
    }
}

// Bilinear resize of the whole image `in` (the crop, w x h) to the whole image `out` (64 x 32), FKL's convention.
// A whole-image subroutine, like cv::resize: its annotation consumes all of `in`, and it is never tiled (a tile of
// the output would need the global output size and the tile's origin, and its input footprint has a clamp that
// Fern's subset expressions cannot write).
inline void resize_bilinear(ImageF3 in, ImageF3 out) {
    const int64_t w = in.col, h = in.row;
    const float sx = float(w) / float(out.col);
    const float sy = float(h) / float(out.row);
    const float maxX = float(w - 1), maxY = float(h - 1);
    for (int64_t y = 0; y < out.row; ++y) {
        float fy = float(y) * sy;
        fy = fy < 0.f ? 0.f : (fy > maxY ? maxY : fy);
        const int64_t y0 = int64_t(fy);
        const int64_t y1 = y0 + 1 < h - 1 ? y0 + 1 : h - 1;
        const float ay = fy - float(y0);
        float *o = out.px(y, 0);
        for (int64_t x = 0; x < out.col; ++x) {
            float fx = float(x) * sx;
            fx = fx < 0.f ? 0.f : (fx > maxX ? maxX : fx);
            const int64_t x0 = int64_t(fx);
            const int64_t x1 = x0 + 1 < w - 1 ? x0 + 1 : w - 1;
            const float ax = fx - float(x0);
            const float *c00 = in.px(y0, x0), *c01 = in.px(y0, x1);
            const float *c10 = in.px(y1, x0), *c11 = in.px(y1, x1);
            for (int c = 0; c < 3; ++c) {
                o[3 * x + c] = (c00[c] * (1.f - ax) + c01[c] * ax) * (1.f - ay) +
                               (c10[c] * (1.f - ax) + c11[c] * ax) * ay;
            }
        }
    }
}

// Letterbox: rows [BORDER_TOP, BORDER_TOP + in.row) of the 64 x 64 plane are `in`, the others black. Works on any
// band of rows of the output (out.r0 is the band's first row in the plane), so the band can be a tile.
inline void letterbox(ImageF3 in, ImageF3 out) {
    for (int64_t i = 0; i < out.row; ++i) {
        const int64_t r = out.r0 + i - BORDER_TOP;
        float *o = out.px(i, 0);
        if (r >= 0 && r < in.row) {
            const float *s = in.px(r, out.c0);
            for (int64_t j = 0; j < out.col * 3; ++j) o[j] = s[j];
        } else {
            for (int64_t j = 0; j < out.col * 3; ++j) o[j] = 0.f;
        }
    }
}

// (v * (1/255) - SUB[c]) / DIV[c], pointwise
inline void normalize(ImageF3 in, ImageF3 out) {
    for (int64_t i = 0; i < out.row; ++i) {
        const float *s = in.px(i, 0);
        float *o = out.px(i, 0);
        for (int64_t j = 0; j < out.col; ++j) {
            for (int c = 0; c < 3; ++c) o[3 * j + c] = (s[3 * j + c] * MUL - SUB[c]) / DIV[c];
        }
    }
}

} // namespace fern_bvf
