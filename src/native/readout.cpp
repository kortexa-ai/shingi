// Native first-position logits from Prism's ternary llama.cpp runtime.
// No sampling, generated answer, or truncated top-k distribution is involved.
#include "llama.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

static std::vector<llama_token> tokenize(const llama_vocab *vocab, const std::string &s,
                                        bool special) {
    int n = llama_tokenize(vocab, s.data(), s.size(), nullptr, 0, special, special);
    std::vector<llama_token> tokens(std::abs(n));
    n = llama_tokenize(vocab, s.data(), s.size(), tokens.data(), tokens.size(), special, special);
    if (n < 0) throw std::runtime_error("tokenization failed");
    tokens.resize(n);
    return tokens;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: readout MODEL.gguf CONTEXT_TOKENS [ADAPTER.gguf]\n";
        return 2;
    }
    const bool vocab_only = std::getenv("SHINGI_VOCAB_ONLY") != nullptr;
    const char *gpu = std::getenv("CUDA_VISIBLE_DEVICES");
    bool permitted = gpu && std::regex_match(gpu, std::regex("GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"));
    if (!permitted && !vocab_only) {
        std::cerr << "Set CUDA_VISIBLE_DEVICES to exactly one full GPU UUID\n";
        return 2;
    }
    int context = std::stoi(argv[2]);
    if (context < 512 || context > 16384) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    if (!vocab_only && !llama_supports_gpu_offload()) return 2;
    if (!vocab_only) {
        size_t gpu_count = 0;
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i)
            if (ggml_backend_dev_type(ggml_backend_dev_get(i)) == GGML_BACKEND_DEVICE_TYPE_GPU) ++gpu_count;
        if (gpu_count != 1) {
            std::cerr << "Expected one visible GPU; refusing CPU fallback or multiple devices\n";
            return 2;
        }
    }
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = vocab_only ? 0 : 99;
    mp.vocab_only = vocab_only;
    auto *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = context;
    cp.n_batch = 512;
    cp.n_ubatch = 512;
    cp.n_threads = 12;
    cp.n_threads_batch = 12;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    bool fp16_kv = std::getenv("SHINGI_KV_F16") != nullptr;
    cp.type_k = fp16_kv ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    cp.type_v = fp16_kv ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
    auto *ctx = vocab_only ? nullptr : llama_init_from_model(model, cp);
    if (!ctx && !vocab_only) { llama_model_free(model); return 1; }
    llama_adapter_lora *adapter = nullptr;
    if (argc == 4 && !vocab_only) {
        adapter = llama_adapter_lora_init(model, argv[3]);
        float scale = 1.0f;
        if (!adapter || llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0) return 1;
    }
    const auto *vocab = llama_model_get_vocab(model);
    std::cout << json({{"ready", true}, {"context_tokens", vocab_only ? context : llama_n_ctx(ctx)},
                       {"vocab_size", llama_vocab_n_tokens(vocab)}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            auto request = json::parse(line);
            std::string prompt = request.at("prompt");
            auto tokens = tokenize(vocab, prompt, true);
            if (tokens.empty() || (!vocab_only && tokens.size() > llama_n_ctx(ctx)))
                throw std::invalid_argument("prompt exceeds context or is empty; never truncated");
            std::vector<llama_token> ids;
            std::set<llama_token> unique;
            for (const auto &label : request.at("labels")) {
                auto t = tokenize(vocab, label.get<std::string>(), false);
                if (t.size() != 1 || !unique.insert(t[0]).second)
                    throw std::invalid_argument("labels must be distinct single tokens");
                ids.push_back(t[0]);
            }
            if (ids.empty() || ids.size() > 255) throw std::invalid_argument("invalid candidate count");
            if (request.value("tokenize_only", false)) {
                std::cout << json({{"input_tokens", tokens.size()}, {"candidate_ids", ids}, {"input_ids", tokens}}).dump() << std::endl;
                continue;
            }
            if (vocab_only) throw std::runtime_error("vocabulary-only process cannot run inference");
            llama_memory_clear(llama_get_memory(ctx), true);
            auto start = std::chrono::steady_clock::now();
            for (size_t pos = 0; pos < tokens.size(); pos += cp.n_batch) {
                auto count = std::min<size_t>(cp.n_batch, tokens.size() - pos);
                auto batch = llama_batch_get_one(tokens.data() + pos, count);
                if (llama_decode(ctx, batch) != 0) throw std::runtime_error("llama_decode failed");
            }
            const float *all = llama_get_logits_ith(ctx, -1);
            if (!all) throw std::runtime_error("no final logits");
            std::vector<double> logits;
            for (auto id : ids) {
                if (!std::isfinite(all[id])) throw std::runtime_error("non-finite candidate logit");
                logits.push_back(all[id]);
            }
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            std::cout << json({{"logits", logits}, {"candidate_ids", ids},
                              {"input_tokens", tokens.size()}, {"prefill_ms", ms}}).dump() << std::endl;
        } catch (const std::invalid_argument &e) {
            std::cout << json({{"error", e.what()}, {"error_kind", "input"}}).dump() << std::endl;
        } catch (const json::exception &e) {
            std::cout << json({{"error", e.what()}, {"error_kind", "input"}}).dump() << std::endl;
        } catch (const std::exception &e) {
            std::cout << json({{"error", e.what()}, {"error_kind", "runtime"}}).dump() << std::endl;
        }
    }
    if (ctx) llama_free(ctx);
    if (adapter) llama_adapter_lora_free(adapter);
    llama_model_free(model);
    llama_backend_free();
}
