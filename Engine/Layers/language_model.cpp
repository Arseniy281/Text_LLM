#include "language_model.h"
#include "../Tensor/tensor.h"
#include "../Tensor/device.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <fstream>
#include <sys/stat.h>
#include <errno.h>
#include <chrono>
#include <iostream>
#include <filesystem>

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


std::shared_ptr<Tensor> LanguageModel::forward(const std::shared_ptr<Tensor>& tokens) {
    ScopedTimer timer("LanguageModel::forward");
    auto x = embedding_.forward(tokens);

    if (x->GetShape().size() == 2) {
        x = std::make_shared<Tensor>(x->Reshape({
             1, x->GetShape()[0], x->GetShape()[1]}));
    }

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


void LanguageModel::TopP(Tensor& last_logits, float top_p) {
    std::vector<std::pair<float, int>> indexed_probs;
    for (size_t i = 0; i < vocab_size_; i++) {
        indexed_probs.push_back({last_logits.at(i), i});
    }
    std::sort(indexed_probs.begin(), indexed_probs.end(), [](const auto& a, const auto& b){
            return a.first > b.first; 
        });
    float total_out = 0.0f;
    size_t border = vocab_size_ - 1;
    for (size_t i = 0; i < vocab_size_; i++) {
        total_out += indexed_probs[i].first;
        if (total_out >= top_p) { 
            border = i;
            break;
        }
    }
    for (size_t i = border + 1; i < vocab_size_; i++) {
        last_logits.at(indexed_probs[i].second) = 0.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i <= border; i++) {
        sum += indexed_probs[i].first;
    }
    if (sum > 0.0f) {
        for (size_t i = 0; i <= border; i++) {
            last_logits.at(indexed_probs[i].second) = indexed_probs[i].first / sum;
        }
    }
}

std::vector<size_t> LanguageModel::generate(const std::vector<size_t>& prompt,
        int max_new_tokens, float temperature, float top_p, int end_token_id) {

    if (prompt.empty() || max_new_tokens <= 0) {
        return {};
    }

    if (temperature <= 0.0f) {
        temperature = 1.0f;
    }

    transformer_.SetUseKVCache(true);
    transformer_.ResetCache();

    std::shared_ptr<Tensor> output;

    for (size_t token : prompt) {
        auto input = std::make_shared<Tensor>(
            std::vector<size_t>{1, 1});

        input->at({0, 0}) = static_cast<float>(token);
        output = forward(input);
    }

    std::vector<size_t> generated;
    generated.reserve(max_new_tokens);

    for (int step = 0; step < max_new_tokens; step++) {
        size_t last_position = output->GetShape()[1] - 1;
        Tensor probs({vocab_size_}, 0.0f, output->GetDevice());

        float max_logit = output->at({0, last_position, 0});

        for (size_t i = 1; i < vocab_size_; i++) {
            float value = output->at({0, last_position, i});

            if (value > max_logit) {
                max_logit = value;
            }
        }

        float sum = 0.0f;

        for (size_t i = 0; i < vocab_size_; i++) {
            float value =
                std::exp((output->at({
                    0, last_position, i}) - max_logit) / temperature);

            probs.at(i) = value;
            sum += value;
        }

        if (sum <= 0.0f) {
            throw std::runtime_error(
                "LanguageModel::generate: "
                "softmax sum is not positive"
            );
        }

        for (size_t i = 0; i < vocab_size_; i++) {
            probs.at(i) /= sum;
        }

        if (top_p > 0.0f && top_p < 1.0f) {
            TopP(probs, top_p);
        }

        size_t next_token = static_cast<size_t>(SampleGreedy(probs));

        if (end_token_id >= 0 && next_token == static_cast<size_t>(end_token_id)) {
            break;
        }

        generated.push_back(next_token);

        auto next_input = std::make_shared<Tensor>(
            std::vector<size_t>{1, 1}
        );

        next_input->at({0, 0}) = static_cast<float>(next_token);

        output = forward(next_input);
    }

    transformer_.SetUseKVCache(false);
    transformer_.ResetCache();

    return generated;
}

void LanguageModel::SaveModel(const std::string& folder) {
    std::filesystem::create_directories(folder);
    embedding_.Save(folder + "/embedding");
    transformer_.Save(folder);
    lm_head_.Save(folder, "lm_head");
}

void LanguageModel::LoadModel(const std::string& folder) {
    embedding_.Load(folder + "/embedding");
    transformer_.Load(folder);
    lm_head_.Load(folder, "lm_head");
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