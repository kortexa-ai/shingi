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
    const char *gpu = std::getenv("CUDA_VISIBLE_DEVICES");
    const char *allow_4090 = std::getenv("SHINGI_ALLOW_4090");
    bool is_4090 = gpu && std::string(gpu) == "GPU-afb49bc6-cd89-6584-99cc-a0f03592a010";
    bool permitted = gpu && (std::string(gpu) == "GPU-a71210ca-e14a-755a-88bb-77f53a2102f6" ||
                            (is_4090 && allow_4090 && std::string(allow_4090) == "1"));
    if (!permitted) {
        std::cerr << "Refusing to load: pin an authorized GPU UUID; 4090 requires explicit opt-in\n";
        return 2;
    }
    int context = std::stoi(argv[2]);
    if (context < 512 || context > 65536) return 2;
    if (is_4090 && context > 16384) return 2;
    ggml_backend_load_all();
    llama_backend_init();
    if (!llama_supports_gpu_offload()) return 2;
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
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
    auto *ctx = llama_init_from_model(model, cp);
    if (!ctx) { llama_model_free(model); return 1; }
    llama_adapter_lora *adapter = nullptr;
    if (argc == 4) {
        adapter = llama_adapter_lora_init(model, argv[3]);
        float scale = 1.0f;
        if (!adapter || llama_set_adapters_lora(ctx, &adapter, 1, &scale) != 0) return 1;
    }
    const auto *vocab = llama_model_get_vocab(model);
    std::cout << json({{"ready", true}, {"context_tokens", llama_n_ctx(ctx)},
                       {"vocab_size", llama_vocab_n_tokens(vocab)}}).dump() << std::endl;
    std::string line;
    while (std::getline(std::cin, line)) {
        try {
            auto request = json::parse(line);
            std::string prompt = request.at("prompt");
            auto tokens = tokenize(vocab, prompt, true);
            if (tokens.empty() || tokens.size() > llama_n_ctx(ctx))
                throw std::runtime_error("prompt exceeds context or is empty; never truncated");
            std::vector<llama_token> ids;
            std::set<llama_token> unique;
            for (const auto &label : request.at("labels")) {
                auto t = tokenize(vocab, label.get<std::string>(), false);
                if (t.size() != 1 || !unique.insert(t[0]).second)
                    throw std::runtime_error("labels must be distinct single tokens");
                ids.push_back(t[0]);
            }
            if (ids.empty() || ids.size() > 255) throw std::runtime_error("invalid candidate count");
            if (request.value("tokenize_only", false)) {
                std::cout << json({{"input_tokens", tokens.size()}, {"candidate_ids", ids}, {"input_ids", tokens}}).dump() << std::endl;
                continue;
            }
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
        } catch (const std::exception &e) {
            std::cout << json({{"error", e.what()}}).dump() << std::endl;
        }
    }
    llama_free(ctx);
    if (adapter) llama_adapter_lora_free(adapter);
    llama_model_free(model);
    llama_backend_free();
}
