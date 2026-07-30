#include <cstdint>
#include <cstdio>

#include "arena.h"
#include "base.h"

namespace
{
    struct alignas(64) CacheLineValue final
    {
        std::uint64_t words[8];
    };

    bool is_aligned(const void* pointer, std::size_t alignment)
    {
        return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
    }
}

int main()
{
    MemArena* arena = MemArena::create(MiB(2), KiB(64));
    if (!arena)
    {
        std::fputs("arena creation failed\n", stderr);
        return 1;
    }

    const auto* prefix = push_struct<std::uint8_t>(arena);
    const auto* value = push_struct<CacheLineValue>(arena);
    const auto* values = push_array<CacheLineValue>(arena, 2);

    const bool passed = prefix != nullptr
        && value != nullptr
        && values != nullptr
        && is_aligned(value, alignof(CacheLineValue))
        && is_aligned(values, alignof(CacheLineValue))
        && value->words[0] == 0
        && values[1].words[7] == 0
        && arena->push_aligned(sizeof(CacheLineValue), 3) == nullptr;

    MemArena::destroy(arena);

    if (!passed)
    {
        std::fputs("arena alignment test failed\n", stderr);
        return 1;
    }
    return 0;
}
