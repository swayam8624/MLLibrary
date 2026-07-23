export module MLLibrary.PRNG;

export import MLLibrary.Base;

export class PRNG {
public:
    PRNG() = default;
    PRNG(u64 initstate, u64 initseq) { seed(initstate, initseq); }

    void seed(u64 initstate, u64 initseq);
    u32 rand();
    f32 randf();

private:
    u64 state_ = 0;
    u64 inc_ = 0;
};

export void prng_seed(u64 initstate, u64 initseq);
export u32 prng_rand();
export f32 prng_randf();
