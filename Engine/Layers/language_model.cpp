#include "language_model.h"
#include "../Tensor/tensor.h"
#include "../Tensor/device.h"
#include "../Autograd/reshape_op.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <errno.h>
#include <chrono>
#include <iostream>
#include <filesystem>

#include <cuda_runtime.h>

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

LanguageModel::LanguageModel(size_t vocab_size, size_t embed_dim, size_t num_blocks,
    size_t num_heads, size_t hidden_dim, Device device) : vocab_size_(vocab_size),
      device_(device),
      embedding_(vocab_size, embed_dim, device),
      transformer_(num_blocks, embed_dim, num_heads, hidden_dim, device),
      lm_head_(embed_dim, vocab_size, device),
      gen_(std::random_device{}()) {}


void LanguageModel::UpdateAdamW(float lr, float beta1, float beta2, float eps, float weight_decay) {
    adam_step_++;
    embedding_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, adam_step_);
    transformer_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, adam_step_);
    lm_head_.UpdateAdamW(lr, beta1, beta2, eps, weight_decay, adam_step_);
}


std::shared_ptr<Tensor> LanguageModel::forward(const std::shared_ptr<Tensor>& tokens) {
    ScopedTimer timer("LanguageModel::forward");
    auto x = embedding_.forward(tokens);

    if (x->GetShape().size() == 2) {
        auto reshape_op = std::make_shared<ReshapeOp>(
            std::vector<size_t>{
                1,
                x->GetShape()[0],
                x->GetShape()[1]
            }
        );

        x = reshape_op->forward({x});
    }

    // if (x->GetShape().size() == 2) {
    //     x = std::make_shared<Tensor>(x->Reshape({
    //          1, x->GetShape()[0], x->GetShape()[1]}));
    // }

    x = transformer_.forward(x);
    return lm_head_.forward(x);
}

int LanguageModel::Sample(const Tensor& probs) {
    std::vector<float> probs_vec(probs.GetSize());
    for (size_t i = 0; i < probs.GetSize(); i++) {
        probs_vec[i] = probs.at(i);
    }
    std::discrete_distribution<int> dist(probs_vec.begin(), probs_vec.end());
    return dist(gen_);
}

int LanguageModel::SampleGreedy(const Tensor& probs) {
    return static_cast<int>(GetBackend(probs.GetDevice()).ArgMax(probs));
}


void LanguageModel::TopP(
    Tensor& probs,
    float top_p
) {
    if (top_p <= 0.0f || top_p >= 1.0f) {
        return;
    }

    std::vector<std::pair<float, int>> indexed_probs;

    indexed_probs.reserve(vocab_size_);

    for (size_t i = 0; i < vocab_size_; ++i) {
        indexed_probs.push_back({
            probs.at(i),
            static_cast<int>(i)
        });
    }

    std::sort(
        indexed_probs.begin(),
        indexed_probs.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;
        }
    );

    float cumulative = 0.0f;
    size_t border = 0;

    for (size_t i = 0; i < indexed_probs.size(); ++i) {

        cumulative += indexed_probs[i].first;

        border = i;

        if (cumulative >= top_p) {
            break;
        }
    }

    for (size_t i = border + 1;
         i < indexed_probs.size();
         ++i) {

        probs.at(
            static_cast<size_t>(indexed_probs[i].second)
        ) = 0.0f;
    }

    float sum = 0.0f;

    for (size_t i = 0; i <= border; ++i) {
        sum += indexed_probs[i].first;
    }

    if (sum <= 0.0f || !std::isfinite(sum)) {
        throw std::runtime_error(
            "LanguageModel::TopP: invalid probability sum"
        );
    }

    for (size_t i = 0; i <= border; ++i) {

        probs.at(
            static_cast<size_t>(indexed_probs[i].second)
        ) =
            indexed_probs[i].first / sum;
    }
}

std::vector<size_t> LanguageModel::generate(
    const std::vector<size_t>& prompt,
    int max_new_tokens,
    float temperature,
    float top_p,
    int end_token_id
) {
    if (prompt.empty() || max_new_tokens <= 0) {
        return {};
    }

    if (temperature <= 0.0f) {
        temperature = 1.0f;
    }

    if (device_ != Device::CUDA) {
        throw std::runtime_error(
            "LanguageModel::generate: "
            "CUDA generation currently requires CUDA device"
        );
    }

    transformer_.SetUseKVCache(true);
    transformer_.ResetCache();

    std::shared_ptr<Tensor> output;

    // ========================================================
    // PROMPT
    // ========================================================

    for (size_t pos = 0; pos < prompt.size(); ++pos) {

        size_t token = prompt[pos];

        // Токен сначала создаём на CPU.
        auto input_cpu = std::make_shared<Tensor>(
            std::vector<size_t>{1, 1}
        );

        input_cpu->at({0, 0}) =
            static_cast<float>(token);

        // Затем переносим его на CUDA.
        auto input = std::make_shared<Tensor>(
            std::vector<size_t>{1, 1},
            0.0f,
            Device::CUDA
        );

        input_cpu->CopyToCUDA(*input);

        output = forward(input);

        // Временно синхронизируемся после каждого prompt token.
        // Это позволяет точно определить место CUDA ошибки.
        cudaError_t error = cudaDeviceSynchronize();

        if (error != cudaSuccess) {
            transformer_.SetUseKVCache(false);
            transformer_.ResetCache();

            throw std::runtime_error(
                "LanguageModel::generate: "
                "CUDA error after prompt token " +
                std::to_string(pos) +
                ": " +
                cudaGetErrorString(error)
            );
        }
    }

    // ========================================================
    // GENERATION
    // ========================================================

    std::vector<size_t> generated;
    generated.reserve(max_new_tokens);

    for (int step = 0; step < max_new_tokens; ++step) {

        if (!output) {
            transformer_.SetUseKVCache(false);
            transformer_.ResetCache();

            throw std::runtime_error(
                "LanguageModel::generate: "
                "forward returned null output"
            );
        }

        const std::vector<size_t>& shape =
            output->GetShape();

        if (shape.size() != 3) {
            transformer_.SetUseKVCache(false);
            transformer_.ResetCache();

            throw std::runtime_error(
                "LanguageModel::generate: "
                "expected output rank 3"
            );
        }

        if (shape[0] != 1) {
            transformer_.SetUseKVCache(false);
            transformer_.ResetCache();

            throw std::runtime_error(
                "LanguageModel::generate: "
                "expected batch size 1"
            );
        }

        if (shape[2] != vocab_size_) {
            transformer_.SetUseKVCache(false);
            transformer_.ResetCache();

            throw std::runtime_error(
                "LanguageModel::generate: "
                "output vocabulary size does not match model"
            );
        }

        size_t last_position = shape[1] - 1;

        // ====================================================
        // CUDA -> CPU
        // ====================================================

        Tensor logits(
            {vocab_size_},
            0.0f,
            Device::CPU
        );

        const size_t offset =
            last_position * vocab_size_;

        cudaError_t error = cudaMemcpy(
            logits.Data(),
            output->Data() + offset,
            vocab_size_ * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        if (error != cudaSuccess) {
            transformer_.SetUseKVCache(false);
            transformer_.ResetCache();

            throw std::runtime_error(
                "LanguageModel::generate: "
                "failed to copy logits CUDA -> CPU: " +
                std::string(cudaGetErrorString(error))
            );
        }

        // ====================================================
        // SOFTMAX
        // ====================================================

        float max_logit = logits.at(0);

        for (size_t i = 1; i < vocab_size_; ++i) {

            float value = logits.at(i);

            if (value > max_logit) {
                max_logit = value;
            }
        }

        Tensor probs(
            {vocab_size_},
            0.0f,
            Device::CPU
        );

        float sum = 0.0f;

        for (size_t i = 0; i < vocab_size_; ++i) {

            float value = std::exp(
                (logits.at(i) - max_logit) /
                temperature
            );

            probs.at(i) = value;
            sum += value;
        }

        if (!std::isfinite(sum) || sum <= 0.0f) {
            transformer_.SetUseKVCache(false);
            transformer_.ResetCache();

            throw std::runtime_error(
                "LanguageModel::generate: "
                "invalid softmax sum"
            );
        }

        for (size_t i = 0; i < vocab_size_; ++i) {
            probs.at(i) /= sum;
        }

        // ====================================================
        // TOP-P
        // ====================================================

        if (top_p > 0.0f && top_p < 1.0f) {
            TopP(probs, top_p);
        }

        float probability_sum = 0.0f;

        for (size_t i = 0; i < vocab_size_; ++i) {
            probability_sum += probs.at(i);
        }

        if (!std::isfinite(probability_sum)) {
            throw std::runtime_error(
                "LanguageModel::generate: "
                "probabilities became NaN/Inf after TopP"
            );
        }

        // ====================================================
        // SAMPLING
        // ====================================================

        size_t next_token =
            static_cast<size_t>(Sample(probs));

        if (
            end_token_id >= 0 &&
            next_token == static_cast<size_t>(end_token_id)
        ) {
            break;
        }

        generated.push_back(next_token);

        // ====================================================
        // NEXT TOKEN -> CUDA
        // ====================================================

        auto next_input_cpu =
            std::make_shared<Tensor>(
                std::vector<size_t>{1, 1}
            );

        next_input_cpu->at({0, 0}) =
            static_cast<float>(next_token);

        auto next_input =
            std::make_shared<Tensor>(
                std::vector<size_t>{1, 1},
                0.0f,
                Device::CUDA
            );

        next_input_cpu->CopyToCUDA(*next_input);

        output = forward(next_input);

        error = cudaDeviceSynchronize();

        if (error != cudaSuccess) {
            transformer_.SetUseKVCache(false);
            transformer_.ResetCache();

            throw std::runtime_error(
                "LanguageModel::generate: "
                "CUDA error at generation step " +
                std::to_string(step) +
                ": " +
                cudaGetErrorString(error)
            );
        }
    }

    // ========================================================
    // CLEANUP
    // ========================================================

    transformer_.SetUseKVCache(false);
    transformer_.ResetCache();

    return generated;
}

void LanguageModel::SaveModel(const std::string& folder) {
    std::filesystem::create_directories(folder);

    embedding_.Save(folder + "/embedding");
    transformer_.Save(folder);
    lm_head_.Save(folder, "lm_head");

    std::ofstream file(folder + "/adam_step", std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "LanguageModel::SaveModel: "
            "failed to open adam_step"
        );
    }

    file.write(
        reinterpret_cast<const char*>(&adam_step_),
        sizeof(adam_step_)
    );

    if (!file) {
        throw std::runtime_error(
            "LanguageModel::SaveModel: "
            "failed to save adam_step"
        );
    }
}

void LanguageModel::LoadModel(const std::string& folder) {
    embedding_.Load(folder + "/embedding");
    transformer_.Load(folder);
    lm_head_.Load(folder, "lm_head");

    std::ifstream file(folder + "/adam_step", std::ios::binary);

    if (!file) {
        throw std::runtime_error(
            "LanguageModel::LoadModel: "
            "failed to open adam_step"
        );
    }

    file.read(
        reinterpret_cast<char*>(&adam_step_),
        sizeof(adam_step_)
    );

    if (!file) {
        throw std::runtime_error(
            "LanguageModel::LoadModel: "
            "failed to load adam_step"
        );
    }

    transformer_.ResetCache();
}

const Tensor& LanguageModel::GetEmbeddings() const {
    return embedding_.GetEmbeddings();
}
const Tensor& LanguageModel::GetLMHeadWeights() const {
    return lm_head_.GetWeights();
}
const Tensor& LanguageModel::GetLMHeadBias() const {
    return lm_head_.GetBias();
}

void LanguageModel::ResetCache() {
    transformer_.ResetCache();
}

void LanguageModel::Update(float lr) {
    embedding_.Update(lr);
    transformer_.Update(lr);
    lm_head_.Update(lr);
}

void LanguageModel::ClearGrad() {
    embedding_.ClearGrad();
    transformer_.ClearGrad();
    lm_head_.ClearGrad();
}

void LanguageModel::ScaleGrad(float factor) {
    embedding_.ScaleGrad(factor);
    transformer_.ScaleGrad(factor);
    lm_head_.ScaleGrad(factor);
}

void LanguageModel::SetUseKVCache(bool value) {
    transformer_.SetUseKVCache(value);
}

std::shared_ptr<Tensor> LanguageModel::GetEmbeddingGrad() const {
    return embedding_.GetGrad();
}