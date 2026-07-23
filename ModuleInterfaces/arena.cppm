export module MLLibrary.Arena;

export import MLLibrary.Base;

export u32 plat_get_pagesize();

export void* plat_mem_reserve(u64 size);
export b32 plat_mem_commit(void* ptr, u64 size);
export b32 plat_mem_decommit(void* ptr, u64 size);
export b32 plat_mem_release(void* ptr, u64 size);

export class MemArena {
public:
    class Temp;

    static MemArena* create(u64 reserve_size, u64 commit_size);
    static void destroy(MemArena* arena);

    void* push(u64 size, bool zero = true);
    void pop(u64 size);
    void pop_to(u64 pos);
    void clear();

    Temp begin_temp();
    static Temp scratch_get(MemArena** conflicts, u32 num_conflicts);

    u64 position() const { return pos_; }
    u64 capacity() const { return reserve_size_; }

private:
    MemArena() = default;

private:
    u64 reserve_size_ = 0;
    u64 commit_size_ = 0;
    u64 pos_ = 0;
    u64 commit_pos_ = 0;
};

export class MemArena::Temp {
public:
    Temp(MemArena* arena)
        : arena_(arena), start_pos_(arena ? arena->pos_ : 0) {}

    Temp(const Temp&) = delete;
    Temp& operator=(const Temp&) = delete;

    Temp(Temp&& other) noexcept
    {
        arena_ = other.arena_;
        start_pos_ = other.start_pos_;
        other.arena_ = nullptr;
    }

    Temp& operator=(Temp&& other) noexcept
    {
        if (this != &other) {
            release();
            arena_ = other.arena_;
            start_pos_ = other.start_pos_;
            other.arena_ = nullptr;
        }
        return *this;
    }

    ~Temp() { release(); }

    MemArena* arena() const { return arena_; }

    void release()
    {
        if (arena_) {
            arena_->pop_to(start_pos_);
            arena_ = nullptr;
        }
    }

private:
    MemArena* arena_ = nullptr;
    u64 start_pos_ = 0;
};

export template <typename T>
T* push_struct(MemArena* arena, bool zero = true)
{
    if (!arena) {
        return nullptr;
    }
    return static_cast<T*>(arena->push(sizeof(T), zero));
}

export template <typename T>
T* push_array(MemArena* arena, u64 count, bool zero = true)
{
    if (!arena) {
        return nullptr;
    }
    return static_cast<T*>(arena->push(sizeof(T) * count, zero));
}
