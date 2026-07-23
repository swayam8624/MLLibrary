export module MLLibrary.Data;

export import MLLibrary.Arena;
export import MLLibrary.Matrix;

export Matrix* mat_load(MemArena* arena, u32 rows, u32 cols, const char* filename);
export void draw_mnist_digit(const f32* data);
export void one_hot_encode(Matrix* out, const Matrix* labels, u32 num_classes);
