//
//  arena.cpp
//  MLLibrary
//
//  Created by Swayam Singal on 11/04/26.
//

//
// arena.cpp
//

#include "arena.h"

#include <cstring>
#include <algorithm>

//======================
// Internal helpers
//======================

namespace
{
    constexpr u64 align_up_pow2(u64 n, u64 p) noexcept
    {
        return (n + (p - 1)) & ~(p - 1);
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
    u32 pagesize = plat_get_pagesize();

    reserve_size = align_up_pow2(reserve_size, pagesize);
    commit_size = align_up_pow2(commit_size, pagesize);

    void *mem = plat_mem_reserve(reserve_size);
    if (!mem)
        return nullptr;

    if (!plat_mem_commit(mem, commit_size))
    {
        plat_mem_release(mem, reserve_size);
        return nullptr;
    }

    MemArena *arena = new (mem) MemArena();

    arena->reserve_size_ = reserve_size;
    arena->commit_size_ = commit_size;
    arena->pos_ = sizeof(MemArena);
    arena->commit_pos_ = commit_size;

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
    u64 pos_aligned = align_up_pow2(pos_, alignof(void *));
    u64 new_pos = pos_aligned + size;

    if (new_pos > reserve_size_)
        return nullptr;

    if (new_pos > commit_pos_)
    {
        u64 new_commit_pos = align_up_pow2(new_pos, commit_size_);
        new_commit_pos = min_u64(new_commit_pos, reserve_size_);

        u8 *mem = reinterpret_cast<u8 *>(this) + commit_pos_;
        u64 commit_size = new_commit_pos - commit_pos_;

        if (!plat_mem_commit(mem, commit_size))
        {
            return nullptr;
        }

        commit_pos_ = new_commit_pos;
    }

    pos_ = new_pos;

    u8 *out = reinterpret_cast<u8 *>(this) + pos_aligned;

    if (!zero)
    {
        std::memset(out, 0, size);
    }

    return out;
}

void MemArena::pop(u64 size)
{
    size = min_u64(size, pos_ - sizeof(MemArena));
    pos_ -= size;
}

void MemArena::pop_to(u64 pos)
{
    if (pos < pos_)
    {
        pos_ = pos;
    }
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
    {
        return Temp(nullptr);
    }

    MemArena *&selected = scratch_arenas[index];

    if (!selected)
    {
        selected = MemArena::create(64ull << 20, 1ull << 20);
    }

    return selected->begin_temp();
}

#if defined(__APPLE__)

#include <unistd.h>
#include <sys/mman.h>

u32 plat_get_pagesize(void)
{
    return (u32)sysconf(_SC_PAGESIZE);
}

void *plat_mem_reserve(u64 size)
{
    void *out = mmap(NULL, size, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (out == MAP_FAILED)
        return nullptr;
    return out;
}

b32 plat_mem_commit(void *ptr, u64 size)
{
    return mprotect(ptr, size, PROT_READ | PROT_WRITE) == 0;
}

b32 plat_mem_decommit(void *ptr, u64 size)
{
    if (mprotect(ptr, size, PROT_NONE) != 0)
        return false;
    return madvise(ptr, size, MADV_FREE) == 0;
}

b32 plat_mem_release(void *ptr, u64 size)
{
    return munmap(ptr, size) == 0;
}

#endif