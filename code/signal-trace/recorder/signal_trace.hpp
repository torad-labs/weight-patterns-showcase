// Observation only. The scheduler calls before each requested operation and
// synchronizes that operation before the after callback. No tensor is written.
#pragma once
#include "gguf.h"
#include <chrono>
#include <dlfcn.h>

struct signal_trace {
    std::ofstream data, events;
    json weights = json::object();
    std::unordered_map<std::string, json> chunks;
    std::unordered_map<const ggml_tensor *, uint64_t> producers;
    std::unordered_map<std::string, uint64_t> cache_versions;
    std::map<std::string, size_t> operations;
    uint64_t bytes = 0, count = 0, sequence = 0;
    int last_call = -1;
    uint64_t execution = 0;
    const ggml_tensor *pending_tensor = nullptr;
    using synchronize_fn = int (*)();
    synchronize_fn synchronize_device = nullptr;
    bool persist = false;
    json pending;

    void bind_model(const fs::path &model, const fs::path &out, const fs::path &backends) {
        // This instrument is explicitly single-GPU. Reading on CUDA's readback
        // stream alone does not wait for staged input copies on the compute
        // stream. Complete all work on the current device before a preimage.
        void *runtime = dlopen((backends / "libcudart.so.13").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!runtime) throw std::runtime_error("trace CUDA runtime binding");
        synchronize_device = reinterpret_cast<synchronize_fn>(dlsym(runtime, "cudaDeviceSynchronize"));
        if (!synchronize_device) throw std::runtime_error("trace device synchronization unavailable");
        ggml_context *ctx = nullptr;
        auto *g = gguf_init_from_file(model.c_str(), {true, &ctx});
        if (!g) throw std::runtime_error("trace GGUF metadata");
        for (int64_t i = 0; i < gguf_get_n_tensors(g); ++i) {
            const std::string name = gguf_get_tensor_name(g, i);
            auto *t = ggml_get_tensor(ctx, name.c_str());
            weights[name] = {{"name", name}, {"native_type", ggml_type_name(t->type)},
                {"shape", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}},
                {"offset_bytes", gguf_get_data_offset(g) + gguf_get_tensor_offset(g, i)},
                {"bytes", ggml_nbytes(t)}};
        }
        save(out / "weight-bindings.json", {{"model", fs::absolute(model).string()},
            {"model_bytes", fs::file_size(model)}, {"tensors", weights},
            {"meaning", "Static native tensor identities and byte ranges; no semantic labels or per-weight attribution."}});
        gguf_free(g); ggml_free(ctx);
    }

    void begin(const fs::path &dir, bool keep) {
        persist = keep; bytes = count = sequence = execution = 0; last_call = -1; pending_tensor = nullptr;
        chunks.clear(); producers.clear(); cache_versions.clear(); operations.clear();
        if (persist) { data.open(dir / "signal-data.bin", std::ios::binary); events.open(dir / "signal-events.jsonl"); }
    }

    static const ggml_tensor *root(const ggml_tensor *t) {
        while (t->view_src) t = t->view_src;
        return t;
    }

    static json layout(const ggml_tensor *t) {
        json params = json::array(); for (auto p : t->op_params) params.push_back(p);
        return {{"name", t->name}, {"type", ggml_type_name(t->type)},
            {"shape", {t->ne[0], t->ne[1], t->ne[2], t->ne[3]}},
            {"strides", {t->nb[0], t->nb[1], t->nb[2], t->nb[3]}},
            {"operation", ggml_op_name(t->op)}, {"op_params_i32", params}};
    }

    static std::string sha(const unsigned char *p, size_t n) {
        unsigned char h[SHA256_DIGEST_LENGTH]; SHA256(p, n, h);
        return hex(std::string(reinterpret_cast<char *>(h), SHA256_DIGEST_LENGTH));
    }

    json snapshot(const ggml_tensor *t) {
        json result = layout(t), blocks = json::array();
        const size_t n = ggml_nbytes(t);
        std::vector<unsigned char> raw(n);
        ggml_backend_tensor_get(t, raw.data(), 0, n);
        for (size_t start = 0; start < n; start += 65536) {
            const size_t len = std::min<size_t>(65536, n - start);
            const auto key = sha(raw.data() + start, len);
            auto it = chunks.find(key);
            if (it == chunks.end()) {
                json b = {{"sha256", key}, {"offset", bytes}, {"bytes", len}};
                data.write(reinterpret_cast<const char *>(raw.data() + start), len);
                bytes += len; it = chunks.emplace(key, std::move(b)).first;
            }
            blocks.push_back(it->second);
        }
        if (!data) throw std::runtime_error("signal snapshot write");
        result["bytes"] = n; result["chunks"] = blocks;
        return result;
    }

    json operand(const ggml_tensor *t) {
        if (!t) return nullptr;
        const auto *base = root(t);
        json result = {{"layout", layout(t)}};
        const std::string name = t->name, base_name = base->name;
        if (weights.contains(name)) result["weight"] = weights[name];
        else if (weights.contains(base_name)) {
            result["weight"] = weights[base_name];
            result["weight_view_offset"] = uintptr_t(t->data) - uintptr_t(base->data);
        } else {
            result["snapshot"] = snapshot(t);
            if (auto p = producers.find(t); p != producers.end()) result["producer_event"] = p->second;
            else {
                result["producer_event"] = nullptr;
                result["source_status"] = std::string(t->name).find('#') != std::string::npos
                    ? "backend copy; transfer-origin occurrence not exposed by this callback"
                    : "runtime input or mutable storage; inspect cache lineage where present";
            }
        }
        if (base_name.rfind("cache_", 0) == 0) {
            result["cache_storage"] = base_name;
            result["cache_root_layout"] = layout(base);
            result["cache_view_offset"] = uintptr_t(t->data) - uintptr_t(base->data);
            // Version points to the latest actual SET_ROWS occurrence, not a
            // graph node reused at another time. Its predecessor chain retains
            // earlier writes. Raw native bytes retain the exact read, too.
            result["cache_state"] = cache_versions.count(base_name) ? json(cache_versions.at(base_name)) : json(nullptr);
        }
        return result;
    }

    static json row_indices(const ggml_tensor *t) {
        if (t->type != GGML_TYPE_I64 && t->type != GGML_TYPE_I32)
            throw std::runtime_error("unhandled cache index format");
        std::vector<unsigned char> raw(ggml_nbytes(t)); ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        json indices = json::array();
        for (int64_t w = 0; w < t->ne[3]; ++w) for (int64_t z = 0; z < t->ne[2]; ++z)
        for (int64_t y = 0; y < t->ne[1]; ++y) for (int64_t x = 0; x < t->ne[0]; ++x) {
            const auto *p = raw.data() + w*t->nb[3] + z*t->nb[2] + y*t->nb[1] + x*t->nb[0];
            int64_t v; if (t->type == GGML_TYPE_I64) std::memcpy(&v, p, 8);
            else { int32_t a; std::memcpy(&a, p, 4); v = a; }
            indices.push_back(v);
        }
        return indices;
    }

    bool observe(ggml_tensor *t, bool ask, int call, const std::string &phase, int position) {
        // Sham makes the same synchronization requests as a persisted capture.
        if (ask && synchronize_device() != 0) throw std::runtime_error("trace preimage synchronization failed");
        if (!persist) return true;
        if (ask) {
            if (call != last_call) { producers.clear(); last_call = call; }
            // In the pinned Gemma graph embd is the first callback node of
            // each microbatch execution (not just each llama_decode call).
            if (std::string(t->name) == "embd" && t->op == GGML_OP_GET_ROWS) {
                producers.clear(); ++execution;
            }
            if (execution == 0) throw std::runtime_error("unrecognized graph-execution entry");
            if (pending_tensor) throw std::runtime_error("nested signal callback");
            pending_tensor = t;
            pending = {{"event", count}, {"call", call}, {"phase", phase},
                {"graph_execution", execution},
                {"input_position", position}, {"name", t->name},
                {"operation", ggml_op_name(t->op)}, {"before", sequence++}};
            json inputs = json::array(); for (auto *s : t->src) inputs.push_back(operand(s));
            pending["inputs"] = inputs;
            if (t->op == GGML_OP_SET_ROWS) {
                const std::string storage = root(t)->name;
                pending["cache_write"] = {{"storage", storage}, {"slots", row_indices(t->src[1])},
                    {"previous_event", cache_versions.count(storage) ? json(cache_versions.at(storage)) : json(nullptr)},
                    {"source_input", 0}, {"index_input", 1}, {"destination_input", 2}};
            }
        } else {
            if (pending.at("event") != count || pending_tensor != t) throw std::runtime_error("trace callback order");
            pending["output"] = snapshot(t); pending["after"] = sequence++;
            if (t->op == GGML_OP_SET_ROWS) cache_versions[root(t)->name] = count;
            producers[t] = count;
            events << pending.dump(-1, ' ', false, json::error_handler_t::replace) << '\n';
            if (!events) throw std::runtime_error("signal event write");
            ++operations[ggml_op_name(t->op)]; ++count; pending_tensor = nullptr;
        }
        return true;
    }

    json finish() {
        if (persist) { data.close(); events.close(); }
        return {{"events", count}, {"unique_native_bytes", bytes}, {"operations", operations},
            {"storage", "Exact native byte snapshots in original strides; deduplicated 64 KiB chunks, not precision conversion."},
            {"boundary", "Every scheduler callback operation, with inputs captured before and output after. Fused flash-attention internals are opaque."},
            {"preimage_completion", "cudaDeviceSynchronize before each ask in this single-GPU instrument, including sham."},
            {"graph_execution_identity", "Pinned Gemma first embd callback begins each execution; transient producer identities reset then, cache versions persist."},
            {"continuity", "Source snapshots are actual pre-operation values. producer_event is null where an exact producer occurrence has not been resolved; cache_state links mutable storage to versioned SET_ROWS writes."},
            {"position_boundary", "input_position is the last token of a prefill call, not a label for every row. Runtime input/position/index tensors and layouts retain row identity; sequence ID is the single fresh context."}};
    }
};
