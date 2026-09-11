#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
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
        // Берём тот же validation region
        // ----------------------------------------------------

        size_t train_size =
            static_cast<size_t>(
                tokens.size() * 0.9
            );

        size_t validation_start =
            train_size;

        // Используем тот же offset,
        // который был в предыдущем тесте.
        size_t start =
            validation_start + 14136;

        if (start + PROMPT_SIZE >=
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
        // Создаём ДВЕ одинаковые модели
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
        // Загружаем одинаковые веса
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

        // ----------------------------------------------------
        // Сначала сравниваем весь prompt
        // ----------------------------------------------------

        std::cout
            << "========================================\n";
        std::cout
            << "PROMPT COMPARISON\n";
        std::cout
            << "========================================\n\n";

        auto full_input =
            CreateInput(prompt);

        auto full_output =
            full_model.forward(full_input);

        cudaDeviceSynchronize();

        auto full_logits =
            CopyLastLogitsToCPU(full_output);

        std::vector<float> cache_logits;

        // Подаём prompt в cache model
        // по одному токену.

        for (size_t i = 0;
             i < prompt.size();
             ++i) {

            std::vector<size_t> one_token = {
                prompt[i]
            };

            auto input =
                CreateInput(one_token);

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

        if (prompt_diff < 1e-4f) {

            std::cout
                << "[OK] Prompt KV equivalence.\n";

        } else {

            std::cout
                << "[ERROR] Prompt KV mismatch!\n";
        }

        // ----------------------------------------------------
        // Теперь добавляем токены
        // ----------------------------------------------------

        std::cout
            << "\n";
        std::cout
            << "========================================\n";
        std::cout
            << "AUTOREGRESSIVE KV COMPARISON\n";
        std::cout
            << "========================================\n\n";

        std::vector<size_t> sequence =
            prompt;

        bool all_ok = true;

        for (size_t step = 0;
             step < TEST_STEPS;
             ++step) {

            // ------------------------------------------------
            // FULL PREFIX
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
            // Здесь cache_model уже содержит prompt.
            // Поэтому подаём только новый токен.
            // ------------------------------------------------

            size_t token =
                tokens[
                    start +
                    PROMPT_SIZE +
                    step
                ];

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

            // Добавляем тот же настоящий токен
            // в sequence для следующего full forward.

            sequence.push_back(token);
        }

        // ----------------------------------------------------
        // RESULT
        // ----------------------------------------------------

        std::cout
            << "\n";
        std::cout
            << "========================================\n";

        if (all_ok &&
            prompt_diff < 1e-4f) {

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