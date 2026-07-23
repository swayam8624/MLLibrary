export module MLLibrary.Matrix;

export import MLLibrary.Base;
export import MLLibrary.Arena;

export struct Matrix {
    u32 rows;
    u32 cols;
    f32* data;
};

export Matrix* mat_create(MemArena* arena, u32 rows, u32 cols);
export b32 mat_copy(Matrix* dst, const Matrix* src);

export void mat_clear(Matrix* mat);
export void mat_fill(Matrix* mat, f32 x);
export void mat_fill_rand(Matrix* mat, f32 lower, f32 upper);
export void mat_scale(Matrix* mat, f32 scale);

export f32 mat_sum(const Matrix* mat);
export u64 mat_argmax(const Matrix* mat);

export b32 mat_add(Matrix* out, const Matrix* a, const Matrix* b);
export b32 mat_sub(Matrix* out, const Matrix* a, const Matrix* b);
export b32 mat_mul(Matrix* out, const Matrix* a, const Matrix* b,
                   b32 zero_out, b32 transpose_a, b32 transpose_b);

export b32 mat_relu(Matrix* out, const Matrix* in);
export b32 mat_softmax(Matrix* out, const Matrix* in);
export b32 mat_cross_entropy(Matrix* out, const Matrix* p, const Matrix* q);

export b32 mat_relu_add_grad(Matrix* out, const Matrix* in, const Matrix* grad);
export b32 mat_softmax_add_grad(Matrix* out, const Matrix* softmax_out, const Matrix* grad);
export b32 mat_cross_entropy_add_grad(Matrix* p_grad, Matrix* q_grad,
                                      const Matrix* p, const Matrix* q,
                                      const Matrix* grad);
