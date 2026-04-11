//
//  prng.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 11/04/26.
//

//
// prng.cpp
//

#include "prng.h"

//======================
// Global RNG
//======================

static PRNG s_prng_state{
    0x853c49e6748fea9bULL,
    0xda3e39cb94b95bdbULL
};

//======================
// PRNG implementation
//======================

void PRNG::seed(u64 initstate, u64 initseq) {
    state_ = 0u;
    inc_   = (initseq << 1u) | 1u;

    rand();               // advance state
    state_ += initstate;
    rand();
}

u32 PRNG::rand() {
    u64 oldstate = state_;

    state_ = oldstate * 6364136223846793005ULL + inc_;

    u32 xorshifted = static_cast<u32>(((oldstate >> 18u) ^ oldstate) >> 27u);
    u32 rot        = static_cast<u32>(oldstate >> 59u);

    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

f32 PRNG::randf() {
    // better than UINT32_MAX division (avoids bias)
    return (rand() >> 8) * (1.0f / 16777216.0f);
}

//======================
// Global wrappers
//======================

void prng_seed(u64 initstate, u64 initseq) {
    s_prng_state.seed(initstate, initseq);
}

u32 prng_rand() {
    return s_prng_state.rand();
}

f32 prng_randf() {
    return s_prng_state.randf();
}
