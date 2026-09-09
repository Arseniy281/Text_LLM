#include "transformer_block.h"
#include "../Tensor/tensor.h"
#include <sys/stat.h>
#include <errno.h> 
#include <vector>
#include "../Tensor/device.h"

#include <chrono>
#include <iostream>

TransformerBlock::TransformerBlock(size_t embed_dim, size_t num_heads, size_t hidden_dim, Device device)
    : rms_norm_1_(embed_dim, device),
      attention_(embed_dim, num_heads, device),
      rms_norm_2_(embed_dim, device),
      feed_forward_(embed_dim, hidden_dim, device) {}

std::shared_ptr<Tensor> TransformerBlock::forward(const std::shared_ptr<Tensor>& x) {
    auto start = std::chrono::steady_clock::now();

    auto residual = x;

    auto t1 = std::chrono::steady_clock::now();
    auto input = rms_norm_1_.forward(x);
    auto t2 = std::chrono::steady_clock::now();

    input = attention_.forward(input);
    auto t3 = std::chrono::steady_clock::now();

    auto add1 = std::make_shared<AddOp>();
    input = add1->forward({input, residual});
    auto t4 = std::chrono::steady_clock::now();

    residual = input;

    input = rms_norm_2_.forward(input);
    auto t5 = std::chrono::steady_clock::now();

    input = feed_forward_.forward(input);
    auto t6 = std::chrono::steady_clock::now();

    auto add2 = std::make_shared<AddOp>();
    input = add2->forward({input, residual});
    auto t7 = std::chrono::steady_clock::now();

    // std::cout << "[PROFILE] TransformerBlock:\n";
    // std::cout << "  RMSNorm 1:      "
    //           << std::chrono::duration<double, std::milli>(t2 - t1).count()
    //           << " ms\n";
    // std::cout << "  Attention:      "
    //           << std::chrono::duration<double, std::milli>(t3 - t2).count()
    //           << " ms\n";
    // std::cout << "  Residual add 1: "
    //           << std::chrono::duration<double, std::milli>(t4 - t3).count()
    //           << " ms\n";
    // std::cout << "  RMSNorm 2:      "
    //           << std::chrono::duration<double, std::milli>(t5 - t4).count()
    //           << " ms\n";
    // std::cout << "  FeedForward:    "
    //           << std::chrono::duration<double, std::milli>(t6 - t5).count()
    //           << " ms\n";
    // std::cout << "  Residual add 2: "
    //           << std::chrono::duration<double, std::milli>(t7 - t6).count()
    //           << " ms\n";
    // std::cout << "  TOTAL:          "
    //           << std::chrono::duration<double, std::milli>(t7 - start).count()
    //           << " ms\n";

    return input;
}

void TransformerBlock::Update(float lr) {
    rms_norm_1_.Update(lr);
    attention_.Update(lr);
    rms_norm_2_.Update(lr);
    feed_forward_.Update(lr);
}

void TransformerBlock::ClearGrad() {
    rms_norm_1_.ClearGrad();
    attention_.ClearGrad();
    rms_norm_2_.ClearGrad();
    feed_forward_.ClearGrad();
}

void TransformerBlock::ScaleGrad(float factor) {
    rms_norm_1_.ScaleGrad(factor);
    attention_.ScaleGrad(factor);
    rms_norm_2_.ScaleGrad(factor);
    feed_forward_.ScaleGrad(factor);
}

void TransformerBlock::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + folder);
    }

    rms_norm_1_.Save(folder + "/rmsnorm_1_gamma");
    attention_.Save(folder + "/attention");
    rms_norm_2_.Save(folder + "/rmsnorm_2_gamma");
    feed_forward_.Save(folder + "/feedforward");
}

void TransformerBlock::Load(const std::string& folder) {
    rms_norm_1_.Load(folder + "/rmsnorm_1_gamma");
    attention_.Load(folder + "/attention");
    rms_norm_2_.Load(folder + "/rmsnorm_2_gamma");
    feed_forward_.Load(folder + "/feedforward");
}

void TransformerBlock::ResetCache() {
    attention_.ResetCache();
}

void TransformerBlock::SetUseKVCache(bool value) {
    attention_.SetUseKVCache(value);
}

const RMSNorm& TransformerBlock::GetRMSNorm1() const {
    return rms_norm_1_;
}

const MultiHeadAttention& TransformerBlock::GetAttention() const {
    return attention_;
}

const RMSNorm& TransformerBlock::GetRMSNorm2() const {
    return rms_norm_2_;
}

const FeedForward& TransformerBlock::GetFeedForward() const {
    return feed_forward_;
}