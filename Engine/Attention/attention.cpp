#include "../Layers/linear_layer.h"
#include "../Tensor/tensor.h"
#include "../Autograd/transpose_op.h"
#include "../Autograd/mul_scalar_op.h"
#include "../Autograd/concatenate_op.h"
#include "../Tensor/device.h"
#include "rope.h"
#include "attention.h"
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include <sys/stat.h>
#include <errno.h> 
#include <chrono>

MultiHeadAttention::MultiHeadAttention(size_t embed_dim, size_t num_heads) 
    : embed_dim_(embed_dim), num_heads_(num_heads), 
      is_first_token_(true), use_kv_cache_(false) {
    head_dim_ = embed_dim_ / num_heads_;
    softmax_.resize(num_heads_);
    
    q_layers_.reserve(num_heads_);
    k_layers_.reserve(num_heads_);
    v_layers_.reserve(num_heads_);
    
    for (size_t i = 0; i < num_heads_; i++) {
        q_layers_.emplace_back(embed_dim_, head_dim_);
        k_layers_.emplace_back(embed_dim_, head_dim_);
        v_layers_.emplace_back(embed_dim_, head_dim_);
    }

    output_layer_ = LinearLayer(embed_dim_, embed_dim_);
}

Tensor MultiHeadAttention::CreateCausalMask(size_t query_len, size_t key_len, size_t query_start, Device device) {
    return GetBackend(device).CreateCausalMask(
        query_len,
        key_len,
        query_start
    );
}

Tensor MultiHeadAttention::GetLastToken(const Tensor& tensor) {
    std::vector<size_t> shape = tensor.GetShape();
    size_t rank = shape.size();
    size_t seq_len = shape[rank - 2];
    size_t head_dim = shape[rank - 1];
    
    std::vector<size_t> new_shape = shape;
    new_shape[rank - 2] = 1;
    
    Tensor result(new_shape);
    
    for (size_t i = 0; i < result.GetSize(); i++) {
        std::vector<size_t> coords = Tensor::IndexToCoord(i, new_shape);
        
        std::vector<size_t> src_coords = coords;
        src_coords[rank - 2] = seq_len - 1;
        
        result.at(coords) = tensor.at(src_coords);
    }
    
    return result;
}


std::shared_ptr<Tensor> ApplyMatMul(const std::shared_ptr<Tensor>& first,
        const std::shared_ptr<Tensor>& second) {

    auto operation = std::make_shared<MulOp>();
    return operation->forward({first,second});
}

std::shared_ptr<Tensor> ApplyTranspose(const std::shared_ptr<Tensor>& input) {
    auto operation = std::make_shared<TransposeOp>();
    return operation->forward({input});
}

std::shared_ptr<Tensor> ApplyMulScalar(const std::shared_ptr<Tensor>& input, float scalar) {
    auto operation = std::make_shared<MulScalarOp>(scalar);
    return operation->forward({input});
}

std::shared_ptr<Tensor> ApplyConcatenate(
        const std::vector<std::shared_ptr<Tensor>>& inputs, size_t axis) {
    auto operation = std::make_shared<ConcatenateOp>(axis);
    return operation->forward(inputs);
}

std::shared_ptr<Tensor> ApplyAdd(const std::shared_ptr<Tensor>& first,
        const std::shared_ptr<Tensor>& second) {
    auto operation = std::make_shared<AddOp>();
    return operation->forward({first, second});
}

std::shared_ptr<Tensor> MultiHeadAttention::forward_no_cache(
        const std::shared_ptr<Tensor>& x) {

    auto total_start = std::chrono::steady_clock::now();

    std::vector<std::shared_ptr<Tensor>> Q;
    std::vector<std::shared_ptr<Tensor>> K;
    std::vector<std::shared_ptr<Tensor>> V;

    Q.reserve(num_heads_);
    K.reserve(num_heads_);
    V.reserve(num_heads_);

    // =========================
    // QKV projections
    // =========================
    auto qkv_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < num_heads_; i++) {
        Q.push_back(q_layers_[i].forward(x));
        K.push_back(k_layers_[i].forward(x));
        V.push_back(v_layers_[i].forward(x));
    }

    auto qkv_end = std::chrono::steady_clock::now();

    saved_Q_ = Q;
    saved_K_ = K;
    saved_V_ = V;

    rope_start_pos_ = 0;

    // =========================
    // RoPE
    // =========================
    auto rope_start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < num_heads_; i++) {
        Q[i] = ApplyRoPE(Q[i], 0);
        K[i] = ApplyRoPE(K[i], 0);
    }

    auto rope_end = std::chrono::steady_clock::now();

    saved_Q_rot_ = Q;
    saved_K_rot_ = K;

    size_t seq_len = x->GetShape()[1];

    // =========================
    // Causal mask
    // =========================
    auto mask_start = std::chrono::steady_clock::now();

    Tensor mask = CreateCausalMask(seq_len, seq_len, 0, x->GetDevice());
    mask = mask.Reshape({ 1, seq_len, seq_len});

    auto mask_ptr = std::make_shared<Tensor>( std::move(mask));
    auto mask_end = std::chrono::steady_clock::now();

    std::vector<std::shared_ptr<Tensor>> head_outputs;
    std::vector<Tensor> attention_weights;

    head_outputs.reserve(num_heads_);
    attention_weights.reserve(num_heads_);

    // =========================
    // Attention
    // =========================
    double qk_time = 0.0;
    double softmax_time = 0.0;
    double av_time = 0.0;

    for (size_t i = 0; i < num_heads_; i++) {

        // QK^T
        auto qk_start = std::chrono::steady_clock::now();

        auto K_transposed = ApplyTranspose(K[i]);
        auto scores = ApplyMatMul(Q[i], K_transposed);

        float scale =
            1.0f / std::sqrt(static_cast<float>(head_dim_));

        scores = ApplyMulScalar(scores, scale);
        scores = ApplyAdd(scores, mask_ptr);

        auto qk_end = std::chrono::steady_clock::now();

        qk_time += std::chrono::duration<double, std::milli>(
            qk_end - qk_start
        ).count();

        // Softmax
        auto softmax_start = std::chrono::steady_clock::now();

        auto weights = softmax_[i].forward(scores);

        auto softmax_end = std::chrono::steady_clock::now();

        softmax_time += std::chrono::duration<double, std::milli>(
            softmax_end - softmax_start
        ).count();

        attention_weights.push_back(*weights);

        // Attention × V
        auto av_start = std::chrono::steady_clock::now();

        auto head_output = ApplyMatMul(weights, V[i]);

        auto av_end = std::chrono::steady_clock::now();

        av_time += std::chrono::duration<double, std::milli>(
            av_end - av_start
        ).count();

        head_outputs.push_back(head_output);
    }

    saved_attention_weights_ = attention_weights;

    // =========================
    // Concatenate
    // =========================
    auto concat_start = std::chrono::steady_clock::now();

    auto concatenated = ApplyConcatenate(head_outputs, 2);

    auto concat_end = std::chrono::steady_clock::now();

    saved_concatenated_ = *concatenated;

    // =========================
    // Output projection
    // =========================
    auto output_start = std::chrono::steady_clock::now();

    auto output = output_layer_.forward(concatenated);

    auto output_end = std::chrono::steady_clock::now();

    saved_output_ = *output;

    // =========================
    // PRINT
    // =========================
    // std::cout << "[PROFILE] MultiHeadAttention:\n";

    // std::cout << "  QKV projections: "
    //           << std::chrono::duration<double, std::milli>(
    //                  qkv_end - qkv_start).count()
    //           << " ms\n";

    // std::cout << "  RoPE:            "
    //           << std::chrono::duration<double, std::milli>(
    //                  rope_end - rope_start).count()
    //           << " ms\n";

    // std::cout << "  Causal mask:     "
    //           << std::chrono::duration<double, std::milli>(
    //                  mask_end - mask_start).count()
    //           << " ms\n";

    // std::cout << "  QK^T + scale:    "
    //           << qk_time
    //           << " ms\n";

    // std::cout << "  Softmax:         "
    //           << softmax_time
    //           << " ms\n";

    // std::cout << "  Attention × V:   "
    //           << av_time
    //           << " ms\n";

    // std::cout << "  Concatenate:     "
    //           << std::chrono::duration<double, std::milli>(
    //                  concat_end - concat_start).count()
    //           << " ms\n";

    // std::cout << "  Output projection: "
    //           << std::chrono::duration<double, std::milli>(
    //                  output_end - output_start).count()
    //           << " ms\n";

    // std::cout << "  TOTAL:           "
    //           << std::chrono::duration<double, std::milli>(
    //                  output_end - total_start).count()
    //           << " ms\n";

    return output;
}

std::shared_ptr<Tensor> MultiHeadAttention::forward_with_cache(
        const std::shared_ptr<Tensor>& x) {
    std::vector<std::shared_ptr<Tensor>> Q;
    std::vector<std::shared_ptr<Tensor>> K;
    std::vector<std::shared_ptr<Tensor>> V;

    Q.reserve(num_heads_);
    K.reserve(num_heads_);
    V.reserve(num_heads_);

    for (size_t i = 0; i < num_heads_; i++) {
        Q.push_back(q_layers_[i].forward(x));
        K.push_back(k_layers_[i].forward(x));
        V.push_back(v_layers_[i].forward(x));
    }

    saved_Q_ = Q;
    saved_K_ = K;
    saved_V_ = V;

    size_t rope_start_pos = 0;

    if (!kv_cache_K_.empty()) {
        rope_start_pos = kv_cache_K_[0].GetShape()[1];
    }

    rope_start_pos_ = rope_start_pos;

    for (size_t i = 0; i < num_heads_; i++) {
        Q[i] = ApplyRoPE(Q[i], rope_start_pos);
        K[i] = ApplyRoPE(K[i], rope_start_pos);
    }

    saved_Q_rot_ = Q;
    saved_K_rot_ = K;
    size_t current_seq_len = x->GetShape()[1];

    if (is_first_token_) {
        is_first_token_ = false;

        kv_cache_K_.clear();
        kv_cache_V_.clear();

        kv_cache_K_.reserve(num_heads_);
        kv_cache_V_.reserve(num_heads_);

        for (size_t i = 0; i < num_heads_; i++) {
            kv_cache_K_.push_back(*K[i]);
            kv_cache_V_.push_back(*V[i]);
        }
    }
    else {
        for (size_t i = 0; i < num_heads_; i++) {
            kv_cache_K_[i] = Tensor::Concatenate({kv_cache_K_[i], *K[i]}, 1);
            kv_cache_V_[i] = Tensor::Concatenate({kv_cache_V_[i], *V[i]},1);
        }
    }
    size_t total_len = kv_cache_K_[0].GetShape()[1];
    size_t query_start = total_len - current_seq_len;

    Tensor mask = CreateCausalMask( current_seq_len, total_len, query_start, x->GetDevice());

    mask = mask.Reshape({1, current_seq_len, total_len});
    auto mask_ptr = std::make_shared<Tensor>( std::move(mask));
    std::vector<Tensor> attention_weights;
    attention_weights.reserve(num_heads_);
    std::vector<std::shared_ptr<Tensor>> head_outputs;
    head_outputs.reserve(num_heads_);

    for (size_t i = 0; i < num_heads_; i++) {
        auto K_cache = std::make_shared<Tensor>(kv_cache_K_[i]);
        auto V_cache = std::make_shared<Tensor>(kv_cache_V_[i]);
        auto K_transposed = ApplyTranspose(K_cache);

        auto scores = ApplyMatMul(Q[i], K_transposed);
        float scale = 1.0f / std::sqrt((float)(head_dim_));
        scores = ApplyMulScalar(scores, scale);
        scores = ApplyAdd(scores, mask_ptr);
        auto weights = softmax_[i].forward(scores);
        attention_weights.push_back(*weights);
        auto head_output = ApplyMatMul(weights, V_cache);
        head_outputs.push_back(head_output);
    }

    saved_attention_weights_ = attention_weights;
    auto concatenated = ApplyConcatenate(head_outputs, 2);
    saved_concatenated_ = *concatenated;
    auto output = output_layer_.forward(concatenated);

    saved_output_ = *output;
    return output;
}

std::shared_ptr<Tensor> MultiHeadAttention::forward(const std::shared_ptr<Tensor>& x) {
    if (!use_kv_cache_) {
        return forward_no_cache(x);
    }

    return forward_with_cache(x);
}

void MultiHeadAttention::Update(float lr) {
    for (auto& layer : q_layers_) layer.Update(lr);
    for (auto& layer : k_layers_) layer.Update(lr);
    for (auto& layer : v_layers_) layer.Update(lr);
    output_layer_.Update(lr);
}

void MultiHeadAttention::ClearGrad() {
    for (auto& layer : q_layers_) layer.ClearGrad();
    for (auto& layer : k_layers_) layer.ClearGrad();
    for (auto& layer : v_layers_) layer.ClearGrad();
    output_layer_.ClearGrad();
}

void MultiHeadAttention::ScaleGrad(float factor) {
    for (auto& layer : q_layers_) layer.ScaleGrad(factor);
    for (auto& layer : k_layers_) layer.ScaleGrad(factor);
    for (auto& layer : v_layers_) layer.ScaleGrad(factor);
    output_layer_.ScaleGrad(factor);
}

void MultiHeadAttention::Save(const std::string& folder) const {
    if (mkdir(folder.c_str(), 0777) != 0 && errno != EEXIST) {
        throw std::runtime_error("Cannot create directory: " + folder);
    }

    for (size_t i = 0; i < num_heads_; i++) {
        q_layers_[i].Save(folder, "q_" + std::to_string(i));
        k_layers_[i].Save(folder, "k_" + std::to_string(i));
        v_layers_[i].Save(folder, "v_" + std::to_string(i));
    }
    output_layer_.Save(folder, "output");
}

void MultiHeadAttention::Load(const std::string& folder) {
    for (size_t i = 0; i < num_heads_; i++) {
        q_layers_[i].Load(folder, "q_" + std::to_string(i));
        k_layers_[i].Load(folder, "k_" + std::to_string(i));
        v_layers_[i].Load(folder, "v_" + std::to_string(i));
    }
    output_layer_.Load(folder, "output");
}

void MultiHeadAttention::ResetCache() {
    is_first_token_ = true;
    kv_cache_K_.clear();
    kv_cache_V_.clear();
    rope_start_pos_ = 0;
}

size_t MultiHeadAttention::GetRopeStartPos() const {
    return rope_start_pos_;
}

void MultiHeadAttention::SetUseKVCache(bool value) {
    use_kv_cache_ = value;
}

const LinearLayer& MultiHeadAttention::GetQLayer(size_t head) const {
    return q_layers_.at(head);
}

const LinearLayer& MultiHeadAttention::GetKLayer(size_t head) const {
    return k_layers_.at(head);
}

const LinearLayer& MultiHeadAttention::GetVLayer(size_t head) const {
    return v_layers_.at(head);
}

LinearLayer& MultiHeadAttention::GetQLayer(size_t head) {
    return q_layers_.at(head);
}

LinearLayer& MultiHeadAttention::GetKLayer(size_t head) {
    return k_layers_.at(head);
}

LinearLayer& MultiHeadAttention::GetVLayer(size_t head) {
    return v_layers_.at(head);
}

const LinearLayer& MultiHeadAttention::GetOutputLayer() const {
    return output_layer_;
}