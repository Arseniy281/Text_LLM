#include "trainer.h"
#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"
#include <vector>
#include <iostream>
#include <chrono>

Tensor MakeInput(const std::vector<size_t>& tokens) {
    Tensor input({1, tokens.size()});

    for (size_t i = 0; i < tokens.size(); i++) {
        input.at({0, i}) = static_cast<float>(tokens[i]);
    }

    return input;
}

void TrainModel(LanguageModel& model,
                const std::vector<std::vector<size_t>>& dataset,
                CrossEntropyLoss& loss_fn,
                size_t epochs,
                float learning_rate) {
    if (dataset.empty()) {
        return;
    }

    for (size_t epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0f;

        size_t correct = 0;
        size_t total = 0;

        for (const auto& chunk : dataset) {
            if (chunk.size() < 2) {
                continue;
            }

            // Input:  the cat eats
            // Target: cat eats fish
            std::vector<size_t> input_tokens(
                chunk.begin(),
                chunk.end() - 1
            );

            std::vector<size_t> target_tokens(
                chunk.begin() + 1,
                chunk.end()
            );

            std::shared_ptr<Tensor> input = std::make_shared<Tensor>(MakeInput(input_tokens));

            Tensor targets({1, target_tokens.size()});

            for (size_t i = 0; i < target_tokens.size(); i++) {
                targets.at({0, i}) = (float)target_tokens[i];
            }

            // =========================
            // FORWARD
            // =========================

            auto logits = model.forward(input);

            // =========================
            // LOSS
            // =========================

            Tensor loss = loss_fn.forward(*logits, targets);

            total_loss += loss.at(0);

            // =========================
            // ACCURACY
            // =========================

            const auto& shape = logits->GetShape();

            size_t seq_len = shape[1];
            size_t vocab_size = shape[2];

            for (size_t pos = 0; pos < seq_len; pos++) {

                size_t predicted_token = 0;
                float max_value = logits->at({0, pos, 0});

                for (size_t token = 1; token < vocab_size; token++) {
                    float value = logits->at({0, pos, token});

                    if (value > max_value) {
                        max_value = value;
                        predicted_token = token;
                    }
                }

                size_t target_token =
                    (size_t)targets.at({0, pos});

                if (predicted_token == target_token) {
                    correct++;
                }

                total++;
            }

            // =========================
            // BACKWARD
            // =========================

            Tensor grad = loss_fn.backward();

            logits->backward(grad);

            // =========================
            // UPDATE
            // =========================

            model.Update(learning_rate);
            model.ClearGrad();
        }

        float average_loss =
            total_loss / dataset.size();

        float accuracy = 0.0f;

        if (total > 0) {
            accuracy =
                100.0f * (float)correct / (float)total;
        }

        std::cout
            << "Epoch " << epoch
            << " | Loss: " << average_loss
            << " | Accuracy: " << accuracy
            << "%\n";
    }
}

void OverfitOneChunk(
    LanguageModel& model,
    const std::vector<size_t>& chunk,
    CrossEntropyLoss& loss_fn,
    size_t epochs,
    float learning_rate) {

    if (chunk.size() < 2) {
        return;
    }

    std::vector<size_t> input_tokens(
        chunk.begin(),
        chunk.end() - 1
    );

    std::vector<size_t> target_tokens(
        chunk.begin() + 1,
        chunk.end()
    );

    std::shared_ptr<Tensor> input = std::make_shared<Tensor>(MakeInput(input_tokens));

    Tensor targets({1, target_tokens.size()});

    for (size_t i = 0; i < target_tokens.size(); i++) {
        targets.at({0, i}) =
            static_cast<float>(target_tokens[i]);
    }

    for (size_t epoch = 0; epoch < epochs; epoch++) {

        model.ClearGrad();

        auto logits = model.forward(input);

        Tensor loss =
            loss_fn.forward(*logits, targets);

        Tensor grad =
            loss_fn.backward();

        logits->backward(grad);

        // Debug: print gradient norms to ensure gradients flow
        // LM head weights
        try {
            const Tensor& lm_w = model.GetLMHeadWeights();
            auto wgrad = lm_w.Grad();
            float w_norm = 0.0f;
            if (wgrad != nullptr) {
                for (size_t i = 0; i < wgrad->GetSize(); i++) {
                    float v = wgrad->RawData()[i];
                    w_norm += v * v;
                }
                w_norm = std::sqrt(w_norm);
            }

            auto egrad = model.GetEmbeddingGrad();
            float e_norm = 0.0f;
            if (egrad != nullptr) {
                for (size_t i = 0; i < egrad->GetSize(); i++) {
                    float v = egrad->RawData()[i];
                    e_norm += v * v;
                }
                e_norm = std::sqrt(e_norm);
            }

            // std::cout << "[DEBUG] loss=" << loss_fn.GetLoss()
            //           << " | grad_norm(lm_head)=" << w_norm
            //           << " | grad_norm(emb)=" << e_norm << "\n";
        } catch (...) {
            // ignore debug failures
        }

        model.Update(learning_rate);
        model.ClearGrad();

        // Accuracy
        size_t correct = 0;
        size_t total = target_tokens.size();

        for (size_t s = 0; s < total; s++) {

            size_t predicted = 0;
            float best = logits->at({0, s, 0});

            for (size_t v = 1;
                 v < logits->GetShape()[2];
                 v++) {

                float value =
                    logits->at({0, s, v});

                if (value > best) {
                    best = value;
                    predicted = v;
                }
            }

            if (predicted == target_tokens[s]) {
                correct++;
            }
        }

        if (epoch % 1 == 0 || epoch == epochs - 1) {

            float accuracy =
                100.0f * correct / total;

            std::cout
                << "Epoch " << epoch
                << " | Loss: "
                << loss_fn.GetLoss()
                << " | Accuracy: "
                << accuracy
                << "%\n";
        }
    }
}