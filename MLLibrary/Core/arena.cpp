//
//  arena.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 11/04/26.
//

#include "arena.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

//======================
// Internal helpers
//======================

namespace
{
    bool align_up(u64 value, u64 alignment, u64& result) noexcept
    {
        if (alignment == 0)
            return false;

        const u64 remainder = value % alignment;
        if (remainder == 0)
        {
            result = value;
            return true;
        }

        const u64 increment = alignment - remainder;
        if (value > std::numeric_limits<u64>::max() - increment)
            return false;

        result = value + increment;
        return true;
    }

    constexpr bool is_power_of_two(u64 value) noexcept
    {
        return value != 0 && (value & (value - 1)) == 0;
    }

    constexpr u64 min_u64(u64 a, u64 b) noexcept
    {
        return (a < b) ? a : b;
    }

    using u8 = std::uint8_t;
}

//======================
// Scratch arenas
//======================

static thread_local MemArena *scratch_arenas[2] = {nullptr, nullptr};

//======================
// MemArena implementation
//======================

MemArena *MemArena::create(u64 reserve_size, u64 commit_size)
{
    if (reserve_size == 0 || commit_size == 0)
        return nullptr;

    const u32 page_size = plat_get_pagesize();
    if (page_size == 0)
        return nullptr;

    u64 aligned_reserve = 0;
    u64 aligned_commit = 0;
    if (!align_up(std::max<u64>(reserve_size, sizeof(MemArena)), page_size, aligned_reserve)
        || !align_up(commit_size, page_size, aligned_commit))
    {
        return nullptr;
    }

    aligned_commit = min_u64(aligned_commit, aligned_reserve);

    void *mem = plat_mem_reserve(aligned_reserve);
    if (!mem)
        return nullptr;

    if (!plat_mem_commit(mem, aligned_commit))
    {
        plat_mem_release(mem, aligned_reserve);
        return nullptr;
    }

    MemArena *arena = new (mem) MemArena();

    arena->reserve_size_ = aligned_reserve;
    arena->commit_size_ = aligned_commit;
    arena->pos_ = sizeof(MemArena);
    arena->commit_pos_ = aligned_commit;

    return arena;
}

void MemArena::destroy(MemArena *arena)
{
    if (!arena)
        return;
    plat_mem_release(arena, arena->reserve_size_);
}

void *MemArena::push(u64 size, bool zero)
{
    return push_aligned(size, alignof(std::max_align_t), zero);
}

void *MemArena::push_aligned(u64 size, u64 alignment, bool zero)
{
    if (!is_power_of_two(alignment))
        return nullptr;

    u64 pos_aligned = 0;
    if (!align_up(pos_, alignment, pos_aligned)
        || pos_aligned > reserve_size_
        || size > reserve_size_ - pos_aligned)
    {
        return nullptr;
    }

    const u64 new_pos = pos_aligned + size;

    if (new_pos > commit_pos_)
    {
        u64 new_commit_pos = 0;
        if (!align_up(new_pos, commit_size_, new_commit_pos))
            return nullptr;
        new_commit_pos = min_u64(new_commit_pos, reserve_size_);

        u8 *mem = reinterpret_cast<u8 *>(this) + commit_pos_;
        const u64 bytes_to_commit = new_commit_pos - commit_pos_;

        if (bytes_to_commit != 0 && !plat_mem_commit(mem, bytes_to_commit))
            return nullptr;

        commit_pos_ = new_commit_pos;
    }

    pos_ = new_pos;

    u8 *out = reinterpret_cast<u8 *>(this) + pos_aligned;
    if (zero && size != 0)
        std::memset(out, 0, size);

    return out;
}

void MemArena::pop(u64 size)
{
    const u64 used = pos_ - sizeof(MemArena);
    pos_ -= min_u64(size, used);
}

void MemArena::pop_to(u64 pos)
{
    const u64 minimum = sizeof(MemArena);
    if (pos < minimum)
        pos = minimum;
    if (pos < pos_)
        pos_ = pos;
}

void MemArena::clear()
{
    pos_ = sizeof(MemArena);
}

//======================
// Temp
//======================

MemArena::Temp MemArena::begin_temp()
{
    return Temp(this);
}

//======================
// Scratch
//======================

MemArena::Temp MemArena::scratch_get(MemArena **conflicts, u32 num_conflicts)
{
    i32 index = -1;

    for (i32 i = 0; i < 2; ++i)
    {
        bool conflict = false;

        for (u32 j = 0; j < num_conflicts; ++j)
        {
            if (scratch_arenas[i] == conflicts[j])
            {
                conflict = true;
                break;
            }
        }

        if (!conflict)
        {
            index = i;
            break;
        }
    }

    if (index == -1)
        return Temp(nullptr);

    MemArena *&selected = scratch_arenas[index];

    if (!selected)
    {
        selected = MemArena::create(64ull << 20, 1ull << 20);
        if (!selected)
            return Temp(nullptr);
    }

    return selected->begin_temp();
}

#if defined(__APPLE__) || defined(__linux__) || defined(__unix__)

#include <sys/mman.h>
#include <unistd.h>

u32 plat_get_pagesize(void)
{
    const long page_size = sysconf(_SC_PAGESIZE);
    return page_size > 0 ? static_cast<u32>(page_size) : 0;
}

void *plat_mem_reserve(u64 size)
{
#if defined(MAP_ANONYMOUS)
    constexpr int anonymous_flag = MAP_ANONYMOUS;
#else
    constexpr int anonymous_flag = MAP_ANON;
#endif
    void *out = mmap(nullptr, size, PROT_NONE, MAP_PRIVATE | anonymous_flag, -1, 0);
    if (out == MAP_FAILED)
        return nullptr;
    return out;
}

b32 plat_mem_commit(void *ptr, u64 size)
{
    return size == 0 || mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
}

b32 plat_mem_decommit(void *ptr, u64 size)
{
    if (size == 0)
        return true;
    if (mprotect(ptr, size, PROT_NONE) != 0)
        return false;
#if defined(__APPLE__) && defined(MADV_FREE)
    return madvise(ptr, size, MADV_FREE) == 0;
#else
    return madvise(ptr, size, MADV_DONTNEED) == 0;
#endif
}

b32 plat_mem_release(void *ptr, u64 size)
{
    return munmap(ptr, size) == 0;
}

#else
#error "MemArena requires a POSIX virtual-memory implementation on this platform."
#endif
