//
//  base.h
//  MLLibrary
//
//  Created by Swayam Singal on 11/04/26.
//

#pragma once

#include <cstdint>
#include <cmath>
#include <string>
#include <algorithm>
#include <iostream>

//======================
// Fixed-width types
//======================

using i8  = std::int8_t;
using i16 = std::int16_t;
using i32 = std::int32_t;
using i64 = std::int64_t;

using u8  = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;

using f32 = float;
using f64 = double;

// Boolean aliases (optional usage)
using b8  = i8;
using b32 = i32;

//======================
// Memory size helpers
//======================

constexpr u64 KiB(u64 n) noexcept {
    return n << 10;
}

constexpr u64 MiB(u64 n) noexcept {
    return n << 20;
}

constexpr u64 GiB(u64 n) noexcept {
    return n << 30;
}

//======================
// Math helpers
//======================

// Safer replacements for macros
template <typename T>
constexpr T Min(const T& a, const T& b) noexcept {
    return (a < b) ? a : b;
}

template <typename T>
constexpr T Max(const T& a, const T& b) noexcept {
    return (a > b) ? a : b;
}

// Align n up to next multiple of p (p must be power of 2)
constexpr u64 AlignUpPow2(u64 n, u64 p) noexcept {
    return (n + (p - 1)) & ~(p - 1);
}

//======================
// Utilities
//======================

constexpr bool IsPowerOfTwo(u64 x) noexcept {
    return x && ((x & (x - 1)) == 0);
}
