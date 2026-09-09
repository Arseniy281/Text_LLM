#pragma once
#include "../Transformer/transformer.h"
#include "embedding_layer.h"
#include "../Tensor/tensor.h"
#include "../Tensor/device.h"
#include "linear_layer.h"
#include <vector>

class LanguageModel {
private:
    EmbeddingLayer embedding_;
    Transformer transformer_;
    LinearLayer lm_head_;
    Softmax softmax_;
    size_t vocab_size_;

    std::mt19937 gen_;

    Device device_;

public:
    LanguageModel(size_t vocab_size, size_t embed_dim, size_t num_blocks,
        size_t num_heads, size_t hidden_dim, Device device = Device::CPU);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& tokens);
    std::vector<size_t> generate(const std::vector<size_t>& prompt,
        int max_len, float temperature, float top_p, int end_token_id);

    int Sample(const Tensor& probs);
    int SampleGreedy(const Tensor& probs);
    void TopP(Tensor& last_logits, float top_p = 0.9);

    void SaveModel(const std::string& folder);
    void LoadModel(const std::string& folder);

    const Tensor& GetEmbeddings() const;
    const Tensor& GetLMHeadWeights() const;
    const Tensor& GetLMHeadBias() const;

    void ResetCache();
    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);
    void SetUseKVCache(bool value);
    std::shared_ptr<Tensor> GetEmbeddingGrad() const;
};