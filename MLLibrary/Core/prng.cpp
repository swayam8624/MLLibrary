//
//  prng.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 11/04/26.
//

//
// prng.cpp
//

// TEACHING RNG — compare docs/MNIST_TRAINING_FROM_FIRST_PRINCIPLES.md.
// Matrices initialized through matrix.cpp::mat_fill_rand use the GLOBAL
// wrappers below; training.cpp uses a separate local PRNG for its shuffle.
// Reproducibility needs BOTH initial weights and batch sequence, and the
// single global generator is not synchronized for concurrent callers.
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

// Reinitialize PCG-style LCG state/odd stream increment, then advance twice.
// Matching seed pair yields the same subsequent sequence on the same code.
void PRNG::seed(u64 initstate, u64 initseq) {
    state_ = 0u;
    inc_   = (initseq << 1u) | 1u;

    rand();               // advance state
    state_ += initstate;
    rand();
}

// Update 64-bit LCG state and permute bits to a rotated 32-bit result.
// Not cryptographically secure. Training's bounded_random additionally
// rejects values to avoid modulo bias in Fisher-Yates shuffles.
u32 PRNG::rand() {
    u64 oldstate = state_;

    state_ = oldstate * 6364136223846793005ULL + inc_;

    u32 xorshifted = static_cast<u32>(((oldstate >> 18u) ^ oldstate) >> 27u);
    u32 rot        = static_cast<u32>(oldstate >> 59u);

    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

// Convert high 24 PRNG bits to Float32 in [0,1); high bits fit precisely in
// Float32's 24-bit significand before scaling by 2^-24.
f32 PRNG::randf() {
    // better than UINT32_MAX division (avoids bias)
    return (rand() >> 8) * (1.0f / 16777216.0f);
}

//======================
// Global wrappers
//======================

// Seed the shared process-global generator used by mat_fill_rand.
void prng_seed(u64 initstate, u64 initseq) {
    s_prng_state.seed(initstate, initseq);
}

// Global-wrapper integer sample; no mutex/atomic protects this shared state.
u32 prng_rand() {
    return s_prng_state.rand();
}

// Global-wrapper float sample, typically used for parameter initialization.
f32 prng_randf() {
    return s_prng_state.randf();
}
