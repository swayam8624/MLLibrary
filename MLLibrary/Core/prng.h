//
//  prng.h
//  MLLibrary
//
//  Created by Swayam Singal on 11/04/26.
//

#pragma once

#include <cstdint>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using f32 = float;

//======================
// PRNG (PCG-style)
//======================

class PRNG {
public:
    // Constructors
    PRNG() = default;
    PRNG(u64 initstate, u64 initseq) {
        seed(initstate, initseq);
    }

    // Seed
    void seed(u64 initstate, u64 initseq);

    // Random integers
    u32 rand();

    // Random float [0, 1)
    f32 randf();

private:
    u64 state_ = 0;
    u64 inc_   = 0;
};

//======================
// Global RNG (optional)
//======================

void prng_seed(u64 initstate, u64 initseq);
u32  prng_rand();
f32  prng_randf();
