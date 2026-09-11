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
const size_t TEST_STEPS = 200;

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
// Rank of target token
//
// Rank 1 = best prediction
// ============================================================

size_t GetRank(
    const std::vector<float>& logits,
    size_t target
) {
    size_t rank = 1;

    for (size_t i = 0; i < logits.size(); ++i) {

        if (logits[i] > logits[target]) {
            ++rank;
        }
    }

    return rank;
}

// ============================================================
// Cross entropy for one target
//
// log_softmax(logits)[target]
// ============================================================

float CrossEntropy(
    const std::vector<float>& logits,
    size_t target
) {
    float max_logit =
        *std::max_element(
            logits.begin(),
            logits.end()
        );

    double sum_exp = 0.0;

    for (float value : logits) {

        sum_exp +=
            std::exp(
                static_cast<double>(
                    value - max_logit
                )
            );
    }

    double log_sum_exp =
        static_cast<double>(max_logit) +
        std::log(sum_exp);

    return static_cast<float>(
        log_sum_exp -
        static_cast<double>(
            logits[target]
        )
    );
}

// ============================================================
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n";
        std::cout
            << "TEACHER-FORCED VALIDATION TEST\n";
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

        std::cout
            << "Validation token position: "
            << start
            << "\n";

        std::cout
            << "Prompt size: "
            << PROMPT_SIZE
            << "\n";

        std::cout
            << "Test steps: "
            << TEST_STEPS
            << "\n\n";

        // ----------------------------------------------------
        // Model
        // ----------------------------------------------------

        LanguageModel model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        std::cout
            << "[OK] Model created.\n";

        model.LoadModel(
            MODEL_PATH
        );

        std::cout
            << "[OK] Model loaded.\n\n";

        // ----------------------------------------------------
        // KV cache
        // ----------------------------------------------------

        model.SetUseKVCache(true);
        model.ResetCache();

        // ----------------------------------------------------
        // Prompt
        // ----------------------------------------------------

        std::vector<size_t> prompt;

        for (size_t i = 0;
             i < PROMPT_SIZE;
             ++i) {

            prompt.push_back(
                tokens[start + i]
            );
        }

        // ----------------------------------------------------
        // Feed prompt token-by-token
        // ----------------------------------------------------

        std::cout
            << "Processing prompt...\n";

        for (size_t i = 0;
             i < prompt.size();
             ++i) {

            auto input =
                CreateInput({prompt[i]});

            auto output =
                model.forward(input);

            cudaDeviceSynchronize();

            // Нам нужен только последний logits
            // после последнего prompt token.
        }

        std::cout
            << "[OK] Prompt processed.\n\n";

        // ====================================================
        // TEST
        // ====================================================

        size_t top1_hits = 0;
        size_t top5_hits = 0;
        size_t top10_hits = 0;

        double total_rank = 0.0;
        double total_loss = 0.0;

        for (size_t step = 0;
             step < TEST_STEPS;
             ++step) {

            // ------------------------------------------------
            // Получаем logits для текущего контекста
            //
            // ВАЖНО:
            //
            // Сейчас после prompt или после предыдущего
            // реального токена cache содержит правильный
            // контекст.
            //
            // Но logits этого контекста нужно получить
            // именно на последнем forward.
            //
            // Поэтому первый шаг берём из prompt_forward,
            // а дальше logits получаем после подачи
            // предыдущего токена.
            // ------------------------------------------------

            // Здесь будет реализована ниже.
        }

        // ----------------------------------------------------
        // Result
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n";

        std::cout
            << "RESULT\n";

        std::cout
            << "========================================\n";

        std::cout
            << "Top-1 accuracy:  "
            << 100.0 *
                static_cast<double>(top1_hits) /
                TEST_STEPS
            << "%\n";

        std::cout
            << "Top-5 accuracy:  "
            << 100.0 *
                static_cast<double>(top5_hits) /
                TEST_STEPS
            << "%\n";

        std::cout
            << "Top-10 accuracy: "
            << 100.0 *
                static_cast<double>(top10_hits) /
                TEST_STEPS
            << "%\n";

        std::cout
            << "Average rank:    "
            << total_rank /
                static_cast<double>(TEST_STEPS)
            << "\n";

        double average_loss =
            total_loss /
            static_cast<double>(TEST_STEPS);

        std::cout
            << "Cross entropy:   "
            << average_loss
            << "\n";

        std::cout
            << "Perplexity:      "
            << std::exp(average_loss)
            << "\n";

        std::cout
            << "========================================\n";

        model.SetUseKVCache(false);
        model.ResetCache();
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