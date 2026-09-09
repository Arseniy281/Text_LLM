#pragma once

#include "../Normalization/rms_norm.h"
#include "../Attention/attention.h"
#include "../Layers/feed_forward.h"
#include "../Tensor/device.h"

class TransformerBlock {
private:
    RMSNorm rms_norm_1_;
    MultiHeadAttention attention_;
    RMSNorm rms_norm_2_;
    FeedForward feed_forward_;

    Tensor saved_residual_1_;
    Tensor saved_residual_2_;
    Tensor saved_norm_1_input_;
    Tensor saved_attention_input_;
    Tensor saved_norm_2_input_;
    Tensor saved_ffn_input_;
public:
    TransformerBlock(size_t embed_dim, size_t num_heads, size_t hidden_dim, Device device = Device::CPU);
    
    // std::shared_ptr<Tensor> forward(const Tensor& x);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);
    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    void Save(const std::string& folder) const;
    void Load(const std::string& folder);

    void ResetCache();
    void SetUseKVCache(bool value);

    RMSNorm& GetRMSNorm1() {
        return rms_norm_1_;
    }

    MultiHeadAttention& GetAttention() {
        return attention_;
    }

    RMSNorm& GetRMSNorm2() {
        return rms_norm_2_;
    }

    FeedForward& GetFeedForward() {
        return feed_forward_;
    }

    const RMSNorm& GetRMSNorm1() const;
    const MultiHeadAttention& GetAttention() const;
    const RMSNorm& GetRMSNorm2() const;
    const FeedForward& GetFeedForward() const;
};