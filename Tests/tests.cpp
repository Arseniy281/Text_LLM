#include "../Engine/LanguageModel/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <stdexcept>

#include <cuda_runtime.h>

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

const size_t CONTEXT = 32;
const size_t GENERATE_TOKENS = 50;

// ============================================================
// Читаем весь файл
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
// Печать токенов
// ============================================================

void PrintTokens(
    const BPETokenizer& tokenizer,
    const std::vector<size_t>& tokens
) {
    std::string text = tokenizer.Decode(tokens);

    std::cout << text << "\n";
}

// ============================================================
// ArgMax
// ============================================================

size_t ArgMax(const std::vector<float>& values) {
    size_t best = 0;

    for (size_t i = 1; i < values.size(); ++i) {
        if (values[i] > values[best]) {
            best = i;
        }
    }

    return best;
}

// ============================================================
// Получить logits последней позиции
// ============================================================

std::vector<float> GetLastLogits(
    const std::shared_ptr<Tensor>& logits,
    size_t vocab_size
) {
    const std::vector<size_t>& shape =
        logits->GetShape();

    if (shape.size() != 3) {
        throw std::runtime_error(
            "Expected logits with rank 3"
        );
    }

    if (shape[0] != 1) {
        throw std::runtime_error(
            "Expected batch size 1"
        );
    }

    if (shape[2] != vocab_size) {
        throw std::runtime_error(
            "Unexpected vocabulary size"
        );
    }

    size_t last_position = shape[1] - 1;

    std::vector<float> result(vocab_size);

    cudaError_t error = cudaMemcpy(
        result.data(),
        logits->Data() +
            last_position * vocab_size,
        vocab_size * sizeof(float),
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
// Greedy token-by-token generation
//
// ВАЖНО:
// Здесь используется именно LanguageModel::generate(),
// потому что мы хотим проверить текущий production inference.
// ============================================================

std::vector<size_t> Generate(
    LanguageModel& model,
    const std::vector<size_t>& prompt,
    size_t count
) {
    return model.generate(
        prompt,
        static_cast<int>(count),
        1.0f,
        1.0f,
        -1
    );
}

// ============================================================
// Main
// ============================================================

int main() {

    try {

        std::cout << "========================================\n";
        std::cout << "AUTOREGRESSIVE VALIDATION TEST\n";
        std::cout << "========================================\n\n";

        // ----------------------------------------------------
        // CUDA
        // ----------------------------------------------------

        int device_count = 0;

        cudaError_t error =
            cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDeviceCount failed: ") +
                cudaGetErrorString(error)
            );
        }

        std::cout << "CUDA devices: "
                  << device_count
                  << "\n";

        if (device_count == 0) {
            throw std::runtime_error(
                "No CUDA devices found"
            );
        }

        cudaDeviceProp prop{};

        cudaGetDeviceProperties(
            &prop,
            0
        );

        std::cout << "GPU: "
                  << prop.name
                  << "\n\n";

        // ----------------------------------------------------
        // Tokenizer
        // ----------------------------------------------------

        std::cout << "Loading tokenizer...\n";

        BPETokenizer tokenizer;

        tokenizer.Load(
            TOKENIZER_PATH
        );

        std::cout << "[OK] Tokenizer loaded.\n";

        std::cout << "Vocabulary size: "
                  << tokenizer.GetVocabSize()
                  << "\n\n";

        if (tokenizer.GetVocabSize() != VOCAB_SIZE) {
            throw std::runtime_error(
                "Unexpected tokenizer vocabulary size"
            );
        }

        // ----------------------------------------------------
        // Corpus
        // ----------------------------------------------------

        std::cout << "Reading corpus...\n";

        std::string corpus =
            ReadFile(CORPUS_PATH);

        std::cout << "[OK] Corpus loaded.\n";
        std::cout << "Corpus chars: "
                  << corpus.size()
                  << "\n\n";

        // ----------------------------------------------------
        // Encode
        // ----------------------------------------------------

        std::cout << "Encoding corpus...\n";

        std::vector<size_t> tokens =
            tokenizer.Encode(corpus);

        std::cout << "[OK] Corpus encoded.\n";

        std::cout << "Total tokens: "
                  << tokens.size()
                  << "\n\n";

        if (tokens.size() <
            CONTEXT + GENERATE_TOKENS + 1) {

            throw std::runtime_error(
                "Corpus is too small"
            );
        }

        // ----------------------------------------------------
        // Validation split
        //
        // ВАЖНО:
        // Это тот же 90/10 split, который использовался
        // в последнем validation test.
        // ----------------------------------------------------

        size_t train_size =
            static_cast<size_t>(
                tokens.size() * 0.9
            );

        size_t validation_start =
            train_size;

        size_t validation_size =
            tokens.size() - validation_start;

        std::cout << "Validation split:\n";
        std::cout << "  Train tokens: "
                  << train_size
                  << "\n";

        std::cout << "  Validation tokens: "
                  << validation_size
                  << "\n\n";

        if (validation_size <
            CONTEXT + GENERATE_TOKENS + 1) {

            throw std::runtime_error(
                "Validation split is too small"
            );
        }

        // ----------------------------------------------------
        // Выбираем случайное окно
        // ----------------------------------------------------

        const size_t usable =
            validation_size -
            CONTEXT -
            GENERATE_TOKENS;

        std::mt19937 rng(42);

        std::uniform_int_distribution<size_t>
            dist(0, usable);

        size_t local_start = dist(rng);

        size_t start =
            validation_start +
            local_start;

        // ----------------------------------------------------
        // Prompt
        // ----------------------------------------------------

        std::vector<size_t> prompt;

        prompt.reserve(CONTEXT);

        for (size_t i = 0; i < CONTEXT; ++i) {
            prompt.push_back(
                tokens[start + i]
            );
        }

        // ----------------------------------------------------
        // Реальное продолжение
        // ----------------------------------------------------

        std::vector<size_t> real_continuation;

        real_continuation.reserve(
            GENERATE_TOKENS
        );

        for (size_t i = 0;
             i < GENERATE_TOKENS;
             ++i) {

            real_continuation.push_back(
                tokens[
                    start +
                    CONTEXT +
                    i
                ]
            );
        }

        // ----------------------------------------------------
        // Информация
        // ----------------------------------------------------

        std::cout << "========================================\n";
        std::cout << "VALIDATION WINDOW\n";
        std::cout << "========================================\n\n";

        std::cout << "Token position: "
                  << start
                  << "\n\n";

        std::cout << "PROMPT (" << CONTEXT
                  << " tokens):\n";

        PrintTokens(
            tokenizer,
            prompt
        );

        std::cout << "\n";

        std::cout << "REAL CONTINUATION ("
                  << GENERATE_TOKENS
                  << " tokens):\n";

        PrintTokens(
            tokenizer,
            real_continuation
        );

        std::cout << "\n";

        // ----------------------------------------------------
        // Создаём модель
        // ----------------------------------------------------

        std::cout << "Creating model...\n";

        LanguageModel model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        std::cout << "[OK] Model created.\n";

        // ----------------------------------------------------
        // Load
        // ----------------------------------------------------

        std::cout << "Loading model...\n";

        model.LoadModel(
            MODEL_PATH
        );

        std::cout << "[OK] Model loaded.\n\n";

        // ----------------------------------------------------
        // Generation
        // ----------------------------------------------------

        std::cout << "Generating "
                  << GENERATE_TOKENS
                  << " tokens...\n";

        std::vector<size_t> generated =
            Generate(
                model,
                prompt,
                GENERATE_TOKENS
            );

        std::cout << "[OK] Generation finished.\n\n";

        // ----------------------------------------------------
        // Проверяем количество
        // ----------------------------------------------------

        std::cout << "Generated tokens: "
                  << generated.size()
                  << "\n";

        if (generated.size() !=
            GENERATE_TOKENS) {

            std::cout
                << "[WARNING] Generated token count differs.\n";
        }

        // ----------------------------------------------------
        // Проверяем IDs
        // ----------------------------------------------------

        bool ids_valid = true;

        for (size_t id : generated) {

            if (id >= VOCAB_SIZE) {
                ids_valid = false;

                std::cout
                    << "[ERROR] Invalid token id: "
                    << id
                    << "\n";
            }
        }

        if (ids_valid) {
            std::cout
                << "[OK] All generated token ids are valid.\n";
        }

        // ----------------------------------------------------
        // Печать generation
        // ----------------------------------------------------

        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "GENERATED CONTINUATION\n";
        std::cout << "========================================\n\n";

        PrintTokens(
            tokenizer,
            generated
        );

        std::cout << "\n";

        // ----------------------------------------------------
        // Token-by-token comparison
        // ----------------------------------------------------

        size_t compare_count =
            std::min(
                generated.size(),
                real_continuation.size()
            );

        size_t matches = 0;

        size_t first_mismatch =
            compare_count;

        for (size_t i = 0;
             i < compare_count;
             ++i) {

            if (generated[i] ==
                real_continuation[i]) {

                ++matches;

            } else if (
                first_mismatch ==
                compare_count) {

                first_mismatch = i;
            }
        }

        // ----------------------------------------------------
        // Результаты
        // ----------------------------------------------------

        std::cout
            << "========================================\n";
        std::cout
            << "AUTOREGRESSIVE RESULTS\n";
        std::cout
            << "========================================\n\n";

        std::cout
            << "Compared tokens: "
            << compare_count
            << "\n";

        std::cout
            << "Exact matches:   "
            << matches
            << "\n";

        if (compare_count > 0) {

            float accuracy =
                static_cast<float>(matches) /
                static_cast<float>(compare_count);

            std::cout
                << "Token accuracy:   "
                << accuracy * 100.0f
                << "%\n";
        }

        if (first_mismatch < compare_count) {

            std::cout
                << "First mismatch:   token "
                << first_mismatch
                << "\n";

            std::cout
                << "Expected token:  "
                << real_continuation[first_mismatch]
                << "\n";

            std::cout
                << "Generated token: "
                << generated[first_mismatch]
                << "\n";
        } else {

            std::cout
                << "No mismatches in compared tokens.\n";
        }

        // ----------------------------------------------------
        // Первые несколько токенов отдельно
        // ----------------------------------------------------

        std::cout << "\n";
        std::cout
            << "FIRST 10 TOKENS:\n";

        size_t first_count =
            std::min<size_t>(
                10,
                compare_count
            );

        for (size_t i = 0;
             i < first_count;
             ++i) {

            std::cout
                << i
                << ": expected="
                << real_continuation[i]
                << ", generated="
                << generated[i];

            if (real_continuation[i] ==
                generated[i]) {

                std::cout << "  [MATCH]";

            } else {

                std::cout << "  [MISS]";
            }

            std::cout << "\n";
        }

        // ----------------------------------------------------
        // Итог
        // ----------------------------------------------------

        std::cout << "\n";
        std::cout
            << "========================================\n";

        if (ids_valid) {

            std::cout
                << "[OK] AUTOREGRESSIVE TEST FINISHED\n";

        } else {

            std::cout
                << "[ERROR] AUTOREGRESSIVE TEST FAILED\n";
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