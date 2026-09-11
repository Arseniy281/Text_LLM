#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// Настройки
// ============================================================

const std::string TOKENIZER_PATH =
    "../Models/MargaritaTokenizer";

const std::string MODEL_PATH =
    "../Models/MargaritaCUDA/step_20000";

const std::string CORPUS_PATH =
    "../Data/master_and_margarita.txt";

const size_t VOCAB_SIZE = 1000;
const size_t EMBED_DIM = 128;
const size_t BLOCKS = 4;
const size_t HEADS = 4;
const size_t HIDDEN = 512;

const size_t PROMPT_SIZE = 32;
const size_t TEST_STEPS = 5;

// ============================================================
// Read file
// ============================================================

std::string ReadFile(const std::string& path) {
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error(
            "Cannot open file: " + path
        );
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

// ============================================================
// CUDA tensor -> CPU vector
// ============================================================

std::vector<float> CopyLastLogitsToCPU(
    const std::shared_ptr<Tensor>& logits
) {
    const auto& shape = logits->GetShape();

    if (shape.size() != 3) {
        throw std::runtime_error(
            "Expected logits rank 3"
        );
    }

    if (shape[0] != 1) {
        throw std::runtime_error(
            "Expected batch size 1"
        );
    }

    size_t seq_len = shape[1];
    size_t vocab = shape[2];

    std::vector<float> result(vocab);

    size_t offset =
        (seq_len - 1) * vocab;

    cudaError_t error = cudaMemcpy(
        result.data(),
        logits->Data() + offset,
        vocab * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpy failed: ") +
            cudaGetErrorString(error)
        );
    }

    return result;
}

// ============================================================
// Create CUDA tensor from token vector
//
// Shape:
// [1, seq]
// ============================================================

std::shared_ptr<Tensor> CreateInput(
    const std::vector<size_t>& tokens
) {
    auto cpu = std::make_shared<Tensor>(
        std::vector<size_t>{
            1,
            tokens.size()
        }
    );

    for (size_t i = 0; i < tokens.size(); ++i) {
        cpu->at({0, i}) =
            static_cast<float>(tokens[i]);
    }

    auto gpu = std::make_shared<Tensor>(
        std::vector<size_t>{
            1,
            tokens.size()
        },
        0.0f,
        Device::CUDA
    );

    cpu->CopyToCUDA(*gpu);

    return gpu;
}

// ============================================================
// Max absolute difference
// ============================================================

float MaxDifference(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error(
            "Vectors have different sizes"
        );
    }

    float max_diff = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(
            max_diff,
            std::abs(a[i] - b[i])
        );
    }

    return max_diff;
}

// ============================================================
// ArgMax
// ============================================================

size_t ArgMax(
    const std::vector<float>& values
) {
    size_t best = 0;

    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }

    return best;
}

// ============================================================
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n";
        std::cout
            << "KV CACHE EQUIVALENCE TEST\n";
        std::cout
            << "========================================\n\n";

        // ----------------------------------------------------
        // CUDA
        // ----------------------------------------------------

        int device_count = 0;

        cudaError_t error =
            cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess ||
            device_count == 0) {

            throw std::runtime_error(
                "CUDA device not available"
            );
        }

        cudaDeviceProp prop{};

        cudaGetDeviceProperties(
            &prop,
            0
        );

        std::cout
            << "GPU: "
            << prop.name
            << "\n\n";

        // ----------------------------------------------------
        // Tokenizer
        // ----------------------------------------------------

        BPETokenizer tokenizer;

        tokenizer.Load(
            TOKENIZER_PATH
        );

        std::cout
            << "[OK] Tokenizer loaded.\n";

        // ----------------------------------------------------
        // Corpus
        // ----------------------------------------------------

        std::string corpus =
            ReadFile(CORPUS_PATH);

        std::vector<size_t> tokens =
            tokenizer.Encode(corpus);

        std::cout
            << "[OK] Corpus encoded.\n";

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n\n";

        // ----------------------------------------------------
        // Validation region
        // ----------------------------------------------------

        size_t train_size =
            static_cast<size_t>(
                tokens.size() * 0.9
            );

        size_t validation_start =
            train_size;

        size_t start =
            validation_start + 14136;

        if (start +
                PROMPT_SIZE +
                TEST_STEPS >=
            tokens.size()) {

            throw std::runtime_error(
                "Invalid test position"
            );
        }

        std::vector<size_t> prompt;

        for (size_t i = 0;
             i < PROMPT_SIZE;
             ++i) {

            prompt.push_back(
                tokens[start + i]
            );
        }

        std::cout
            << "Prompt token position: "
            << start
            << "\n";

        std::cout
            << "Prompt tokens: "
            << prompt.size()
            << "\n\n";

        // ----------------------------------------------------
        // Two identical models
        // ----------------------------------------------------

        LanguageModel full_model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        LanguageModel cache_model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        std::cout
            << "[OK] Models created.\n";

        // ----------------------------------------------------
        // Load same weights
        // ----------------------------------------------------

        full_model.LoadModel(
            MODEL_PATH
        );

        cache_model.LoadModel(
            MODEL_PATH
        );

        std::cout
            << "[OK] Models loaded.\n\n";

        // ----------------------------------------------------
        // FULL MODEL
        // ----------------------------------------------------

        full_model.SetUseKVCache(false);
        full_model.ResetCache();

        // ----------------------------------------------------
        // CACHE MODEL
        // ----------------------------------------------------

        cache_model.SetUseKVCache(true);
        cache_model.ResetCache();

        // ====================================================
        // PROMPT COMPARISON
        // ====================================================

        std::cout
            << "========================================\n";
        std::cout
            << "PROMPT COMPARISON\n";
        std::cout
            << "========================================\n\n";

        // FULL:
        // [prompt]
        auto full_input =
            CreateInput(prompt);

        auto full_output =
            full_model.forward(full_input);

        cudaDeviceSynchronize();

        auto full_logits =
            CopyLastLogitsToCPU(full_output);

        // CACHE:
        // prompt token-by-token
        std::vector<float> cache_logits;

        for (size_t i = 0;
             i < prompt.size();
             ++i) {

            auto input =
                CreateInput({prompt[i]});

            auto output =
                cache_model.forward(input);

            cudaDeviceSynchronize();

            cache_logits =
                CopyLastLogitsToCPU(output);
        }

        float prompt_diff =
            MaxDifference(
                full_logits,
                cache_logits
            );

        size_t full_prediction =
            ArgMax(full_logits);

        size_t cache_prediction =
            ArgMax(cache_logits);

        std::cout
            << "Full forward prediction:  "
            << full_prediction
            << "\n";

        std::cout
            << "KV cache prediction:      "
            << cache_prediction
            << "\n";

        std::cout
            << "Max logits difference:    "
            << prompt_diff
            << "\n";

        bool all_ok =
            prompt_diff < 1e-4f;

        if (all_ok) {

            std::cout
                << "[OK] Prompt KV equivalence.\n";

        } else {

            std::cout
                << "[ERROR] Prompt KV mismatch!\n";
        }

        // ====================================================
        // AUTOREGRESSIVE COMPARISON
        // ====================================================

        std::cout
            << "\n";
        std::cout
            << "========================================\n";
        std::cout
            << "AUTOREGRESSIVE KV COMPARISON\n";
        std::cout
            << "========================================\n\n";

        // ВАЖНО:
        //
        // После prompt обе модели находятся в состоянии:
        //
        // [prompt]
        //
        // На каждом шаге мы добавляем ОДИН И ТОТ ЖЕ
        // реальный токен.
        //
        // FULL получает всю последовательность.
        // CACHE получает только новый токен.

        std::vector<size_t> sequence =
            prompt;

        for (size_t step = 0;
             step < TEST_STEPS;
             ++step) {

            // ------------------------------------------------
            // Реальный следующий токен
            // ------------------------------------------------

            size_t token =
                tokens[
                    start +
                    PROMPT_SIZE +
                    step
                ];

            // ------------------------------------------------
            // Добавляем его в полный контекст
            //
            // Теперь sequence содержит:
            //
            // prompt + token
            // ------------------------------------------------

            sequence.push_back(token);

            // ------------------------------------------------
            // FULL
            //
            // Полностью пересчитываем:
            //
            // [prompt + token]
            // ------------------------------------------------

            auto full_prefix_input =
                CreateInput(sequence);

            auto full_prefix_output =
                full_model.forward(
                    full_prefix_input
                );

            cudaDeviceSynchronize();

            auto full_prefix_logits =
                CopyLastLogitsToCPU(
                    full_prefix_output
                );

            // ------------------------------------------------
            // CACHE
            //
            // Prompt уже был обработан выше.
            //
            // Теперь добавляем только token.
            // ------------------------------------------------

            auto next_input =
                CreateInput({token});

            auto cache_output =
                cache_model.forward(
                    next_input
                );

            cudaDeviceSynchronize();

            auto current_cache_logits =
                CopyLastLogitsToCPU(
                    cache_output
                );

            // ------------------------------------------------
            // Compare
            // ------------------------------------------------

            float diff =
                MaxDifference(
                    full_prefix_logits,
                    current_cache_logits
                );

            size_t full_prediction =
                ArgMax(full_prefix_logits);

            size_t cache_prediction =
                ArgMax(current_cache_logits);

            std::cout
                << "Step "
                << step
                << ":\n";

            std::cout
                << "  Input token:      "
                << token
                << "\n";

            std::cout
                << "  Full prediction:  "
                << full_prediction
                << "\n";

            std::cout
                << "  Cache prediction: "
                << cache_prediction
                << "\n";

            std::cout
                << "  Max difference:   "
                << diff
                << "\n";

            if (diff < 1e-4f) {

                std::cout
                    << "  [OK]\n";

            } else {

                std::cout
                    << "  [ERROR]\n";

                all_ok = false;
            }
        }

        // ====================================================
        // RESULT
        // ====================================================

        std::cout
            << "\n";
        std::cout
            << "========================================\n";

        if (all_ok) {

            std::cout
                << "[OK] KV CACHE EQUIVALENCE PASSED\n";

        } else {

            std::cout
                << "[ERROR] KV CACHE EQUIVALENCE FAILED\n";
        }

        std::cout
            << "========================================\n";
    }
    catch (const std::exception& e) {

        std::cerr
            << "\n[ERROR] "
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}