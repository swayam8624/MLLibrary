#include "checkpoint.hpp"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

constexpr char kMagic[] = "KMLCP01";
constexpr u32 kVersion = 1;

void set_error(std::string* error, const char* message)
{
    if (error) *error = message;
}

std::vector<ModelVar*> parameters(ModelContext* model)
{
    std::vector<ModelVar*> result;
    if (!model) return result;
    for (u32 index = 0; index < model->cost_prog.size; ++index)
    {
        ModelVar* variable = model->cost_prog.vars[index];
        if (variable && (variable->flags & MV_FLAG_PARAMETER)) result.push_back(variable);
    }
    return result;
}

bool write_exact(FILE* file, const void* data, std::size_t size)
{
    return std::fwrite(data, 1, size, file) == size;
}

bool read_exact(FILE* file, void* data, std::size_t size)
{
    return std::fread(data, 1, size, file) == size;
}

} // namespace

bool model_save_checkpoint(const ModelContext* model, const char* path, std::string* error)
{
    if (!model || !path || !*path) { set_error(error, "Checkpoint save requires model and path."); return false; }
    const std::vector<ModelVar*> vars = parameters(const_cast<ModelContext*>(model));
    if (vars.empty()) { set_error(error, "Model has no compiled trainable parameters."); return false; }
    const std::filesystem::path destination(path);
    const std::filesystem::path temporary = destination.string() + ".tmp";
    FILE* file = std::fopen(temporary.c_str(), "wb");
    if (!file) { set_error(error, "Cannot open checkpoint temporary file."); return false; }
    const u32 count = static_cast<u32>(vars.size());
    bool success = write_exact(file, kMagic, sizeof(kMagic) - 1)
        && write_exact(file, &kVersion, sizeof(kVersion))
        && write_exact(file, &count, sizeof(count));
    for (ModelVar* variable : vars)
    {
        const u32 rows = variable->val->rows;
        const u32 columns = variable->val->cols;
        const u64 values = static_cast<u64>(rows) * columns;
        success = success && write_exact(file, &rows, sizeof(rows)) && write_exact(file, &columns, sizeof(columns))
            && write_exact(file, variable->val->data, sizeof(f32) * values);
    }
    success = success && std::fclose(file) == 0;
    if (!success) { std::filesystem::remove(temporary); set_error(error, "Checkpoint write failed."); return false; }
    std::error_code renameError;
    std::filesystem::rename(temporary, destination, renameError);
    if (renameError) { std::filesystem::remove(temporary); set_error(error, "Checkpoint rename failed."); return false; }
    return true;
}

bool model_load_checkpoint(ModelContext* model, const char* path, std::string* error)
{
    if (!model || !path || !*path) { set_error(error, "Checkpoint load requires model and path."); return false; }
    const std::vector<ModelVar*> vars = parameters(model);
    if (vars.empty()) { set_error(error, "Model has no compiled trainable parameters."); return false; }
    FILE* file = std::fopen(path, "rb");
    if (!file) { set_error(error, "Cannot open checkpoint file."); return false; }
    char magic[sizeof(kMagic) - 1]{};
    u32 version = 0, count = 0;
    bool success = read_exact(file, magic, sizeof(magic)) && std::memcmp(magic, kMagic, sizeof(magic)) == 0
        && read_exact(file, &version, sizeof(version)) && version == kVersion
        && read_exact(file, &count, sizeof(count)) && count == vars.size();
    std::vector<std::vector<f32>> loaded;
    loaded.reserve(vars.size());
    for (ModelVar* variable : vars)
    {
        u32 rows = 0, columns = 0;
        success = success && read_exact(file, &rows, sizeof(rows)) && read_exact(file, &columns, sizeof(columns));
        if (!success || rows != variable->val->rows || columns != variable->val->cols) { success = false; break; }
        std::vector<f32> values(static_cast<std::size_t>(rows) * columns);
        success = read_exact(file, values.data(), sizeof(f32) * values.size());
        if (!success) break;
        loaded.push_back(std::move(values));
    }
    std::fclose(file);
    if (!success || loaded.size() != vars.size()) { set_error(error, "Checkpoint is malformed or incompatible with this model."); return false; }
    for (std::size_t index = 0; index < vars.size(); ++index)
    {
        std::memcpy(vars[index]->val->data, loaded[index].data(), sizeof(f32) * loaded[index].size());
    }
    return true;
}
