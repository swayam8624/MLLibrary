module;

#include <cstdint>

export module MLLibrary.Base;

export using i8 = std::int8_t;
export using i16 = std::int16_t;
export using i32 = std::int32_t;
export using i64 = std::int64_t;

export using u8 = std::uint8_t;
export using u16 = std::uint16_t;
export using u32 = std::uint32_t;
export using u64 = std::uint64_t;

export using f32 = float;
export using f64 = double;

export using b8 = i8;
export using b32 = i32;

export constexpr u64 KiB(u64 n) noexcept { return n << 10; }
export constexpr u64 MiB(u64 n) noexcept { return n << 20; }
export constexpr u64 GiB(u64 n) noexcept { return n << 30; }

export template <typename T>
constexpr T Min(const T& a, const T& b) noexcept
{
    return (a < b) ? a : b;
}

export template <typename T>
constexpr T Max(const T& a, const T& b) noexcept
{
    return (a > b) ? a : b;
}

export constexpr u64 AlignUpPow2(u64 n, u64 p) noexcept
{
    return (n + (p - 1)) & ~(p - 1);
}

export constexpr bool IsPowerOfTwo(u64 x) noexcept
{
    return x && ((x & (x - 1)) == 0);
}
