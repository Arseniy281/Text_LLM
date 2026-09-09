#include "transformer.h"
#include <sys/stat.h>
#include <chrono>
#include <iostream>

namespace {
struct ScopedTimer {
    std::string label;
    std::chrono::steady_clock::time_point start;

    explicit ScopedTimer(std::string name)
        : label(std::move(name)),
          start(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        const auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
        // std::cout << "[PROFILE] " << label << " took " << seconds << " s\n";
    }
};
}

Transformer::Transformer(size_t n, size_t embed_dim, size_t num_heads, 
        size_t hidden_dim) : blocks_count_(n) {

    blocks_.reserve(n);

    for (size_t i = 0; i < n; i++) {
        blocks_.emplace_back(embed_dim, num_heads, hidden_dim);
    }
}

std::shared_ptr<Tensor> Transformer::forward(const std::shared_ptr<Tensor>& x) {
    ScopedTimer timer("Transformer::forward");

    std::shared_ptr<Tensor> input = x;
    for (size_t i = 0; i < blocks_count_; i++) {
        input = blocks_[i].forward(input);
    }

    return input;
}

void Transformer::Update(float lr) {
    for (auto& block : blocks_) {
        block.Update(lr);
    }
}

void Transformer::ClearGrad() {
    for (auto& block : blocks_) {
        block.ClearGrad();
    }
}

void Transformer::ScaleGrad(float factor) {
    for (auto& block : blocks_) {
        block.ScaleGrad(factor);
    }
}

void Transformer::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error(
            "Cannot create directory: " + folder
        );
    }

    for (size_t i = 0; i < blocks_count_; i++) {
        blocks_[i].Save(
            folder + "/block_" + std::to_string(i)
        );
    }
}

void Transformer::Load(const std::string& folder) {
    for (size_t i = 0; i < blocks_count_; i++) {
        blocks_[i].Load(folder + "/block_" + std::to_string(i));
    }
}

void Transformer::ResetCache() {
    for (auto& block : blocks_) {
        block.ResetCache();
    }
}

void Transformer::SetUseKVCache(bool value) {
    for (auto& block : blocks_) {
        block.SetUseKVCache(value);
    }
}

const TransformerBlock& Transformer::GetBlock(size_t index) const {
    return blocks_.at(index);
}