#pragma once

#include <cstdint>
#include <cstddef>

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using i32 = std::int32_t;
using b32 = std::int32_t;


//======================
// Platform API (forward)
//======================

u32  plat_get_pagesize();

void* plat_mem_reserve(u64 size);
b32   plat_mem_commit(void* ptr, u64 size);
b32   plat_mem_decommit(void* ptr, u64 size);
b32   plat_mem_release(void* ptr, u64 size);

//======================
// Arena Allocator
//======================

class MemArena {
public:
    class Temp;   // ✅ FIX: forward declaration

    // Creation / destruction
    static MemArena* create(u64 reserve_size, u64 commit_size);
    static void destroy(MemArena* arena);

    // Core allocation
    void* push(u64 size, bool zero = false);
    void  pop(u64 size);
    void  pop_to(u64 pos);
    void  clear();

    // Temp scope
    Temp begin_temp();

    // Scratch
    static Temp scratch_get(MemArena** conflicts, u32 num_conflicts);

    // Getters
    u64 position() const { return pos_; }
    u64 capacity() const { return reserve_size_; }

private:
    MemArena() = default;

private:
    u64 reserve_size_ = 0;
    u64 commit_size_  = 0;

    u64 pos_        = 0;
    u64 commit_pos_ = 0;
};

//======================
// Temp Arena (RAII)
//======================

class MemArena::Temp {
public:
    Temp(MemArena* arena)
        : arena_(arena), start_pos_(arena ? arena->pos_ : 0) {}

    Temp(const Temp&) = delete;
    Temp& operator=(const Temp&) = delete;

    Temp(Temp&& other) noexcept {
        arena_ = other.arena_;
        start_pos_ = other.start_pos_;
        other.arena_ = nullptr;
    }

    Temp& operator=(Temp&& other) noexcept {
        if (this != &other) {
            release();
            arena_ = other.arena_;
            start_pos_ = other.start_pos_;
            other.arena_ = nullptr;
        }
        return *this;
    }

    ~Temp() { release(); }

    void release() {
        if (arena_) {
            arena_->pop_to(start_pos_);
            arena_ = nullptr;
        }
    }

private:
    MemArena* arena_ = nullptr;
    u64 start_pos_   = 0;
};

//======================
// Helpers
//======================

template <typename T>
inline T* push_struct(MemArena* arena, bool zero = false) {
    return static_cast<T*>(arena->push(sizeof(T), zero));
}

template <typename T>
inline T* push_array(MemArena* arena, u64 count, bool zero = false) {
    return static_cast<T*>(arena->push(sizeof(T) * count, zero));
}
