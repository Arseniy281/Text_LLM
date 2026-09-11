#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"
#include "../Engine/Tensor/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>

// ============================================================
// Настройки
// ============================================================

const size_t VOCAB_SIZE = 1000;

const size_t EMBED_DIM = 128;
const size_t BLOCKS = 4;
const size_t HEADS = 4;
const size_t HIDDEN = 512;

const size_t CONTEXT = 128;

const size_t PROMPT_SIZE = 32;
const size_t TEST_STEPS = 200;

const size_t VALIDATION_BATCHES = 20;

const size_t VALIDATION_OFFSET = 0;

const unsigned int SEED = 42;

const std::string TOKENIZER_PATH =
    "/content/Text_LLM/Models/MargaritaTokenizer";

const std::string MODEL_PATH =
    "/content/Text_LLM/Models/MargaritaConstantLR_best";

const std::string CORPUS_PATH =
    "/content/Text_LLM/Data/master_and_margarita.txt";

// ============================================================
// Load text
// ============================================================

std::string LoadText(const std::string& path) {

    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error(
            "Cannot open corpus: " + path
        );
    }

    return std::string(
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    );
}

// ============================================================
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n"
            << "       MARGARITA TEACHER RANK TEST\n"
            << "========================================\n\n";

        // ----------------------------------------------------
        // CUDA
        // ----------------------------------------------------

        int device_count = 0;

        cudaError_t error =
            cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess) {

            throw std::runtime_error(
                std::string(
                    "cudaGetDeviceCount failed: "
                ) +
                cudaGetErrorString(error)
            );
        }

        if (device_count == 0) {
            throw std::runtime_error(
                "No CUDA devices found"
            );
        }

        cudaDeviceProp properties{};

        cudaGetDeviceProperties(
            &properties,
            0
        );

        std::cout
            << "CUDA devices: "
            << device_count
            << "\n";

        std::cout
            << "GPU: "
            << properties.name
            << "\n\n";

        // ----------------------------------------------------
        // Load tokenizer
        // ----------------------------------------------------

        BPETokenizer tokenizer;

        tokenizer.Load(
            TOKENIZER_PATH
        );

        if (tokenizer.GetVocabSize() !=
            VOCAB_SIZE) {

            throw std::runtime_error(
                "Tokenizer vocabulary does not "
                "match model vocabulary"
            );
        }

        std::cout
            << "[OK] Tokenizer loaded.\n";

        // ----------------------------------------------------
        // Load corpus
        // ----------------------------------------------------

        std::string text =
            LoadText(CORPUS_PATH);

        std::vector<size_t> tokens =
            tokenizer.Encode(text);

        std::cout
            << "[OK] Corpus encoded.\n";

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n";

        // ----------------------------------------------------
        // Validation split
        // ----------------------------------------------------

        size_t train_size =
            static_cast<size_t>(
                tokens.size() * 0.9
            );

        size_t validation_start =
            train_size;

        size_t validation_size =
            tokens.size() -
            validation_start;

        std::cout
            << "Train tokens: "
            << train_size
            << "\n";

        std::cout
            << "Validation tokens: "
            << validation_size
            << "\n";

        // ----------------------------------------------------
        // Create model
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
            << "[OK] CUDA model created.\n";

        // ----------------------------------------------------
        // Load best checkpoint
        // ----------------------------------------------------

        std::cout
            << "\nLoading model:\n"
            << MODEL_PATH
            << "\n";

        model.LoadModel(
            MODEL_PATH
        );

        cudaDeviceSynchronize();

        std::cout
            << "[OK] Model loaded.\n";

        // ----------------------------------------------------
        // Teacher-forced evaluation
        // ----------------------------------------------------

        std::mt19937 rng(SEED);

        std::uniform_int_distribution<size_t>
            distribution(
                validation_start,
                tokens.size() - CONTEXT - 1
            );

        double total_loss = 0.0;

        size_t total_tokens = 0;

        size_t top1 = 0;
        size_t top5 = 0;
        size_t top10 = 0;

        double total_rank = 0.0;

        model.SetUseKVCache(false);

        for (size_t batch = 0;
             batch < VALIDATION_BATCHES;
             ++batch) {

            std::vector<size_t> input_data(
                CONTEXT
            );

            std::vector<size_t> target_data(
                CONTEXT
            );

            size_t start =
                distribution(rng);

            for (size_t i = 0;
                 i < CONTEXT;
                 ++i) {

                input_data[i] =
                    tokens[start + i];

                target_data[i] =
                    tokens[start + i + 1];
            }

            Tensor input(
                {1, CONTEXT},
                0.0f,
                Device::CUDA
            );

            Tensor target(
                {1, CONTEXT},
                0.0f,
                Device::CUDA
            );

            Tensor input_cpu(
                {1, CONTEXT},
                0.0f,
                Device::CPU
            );

            Tensor target_cpu(
                {1, CONTEXT},
                0.0f,
                Device::CPU
            );

            for (size_t i = 0;
                 i < CONTEXT;
                 ++i) {

                input_cpu.Data()[i] =
                    static_cast<float>(
                        input_data[i]
                    );

                target_cpu.Data()[i] =
                    static_cast<float>(
                        target_data[i]
                    );
            }

            input_cpu.CopyToCUDA(input);
            target_cpu.CopyToCUDA(target);

            model.ClearGrad();

            Tensor logits =
                model.forward(input);

            // ------------------------------------------------
            // Last-token predictions
            // ------------------------------------------------

            Tensor logits_cpu(
                logits->GetShape(),
                0.0f,
                Device::CPU
            );

            cudaMemcpy(
                logits_cpu.Data(),
                logits->Data(),
                logits->GetSize() * sizeof(float),
                cudaMemcpyDeviceToHost
            );

            const auto& shape =
                logits->GetShape();

            size_t sequence_length =
                shape[1];

            size_t vocab_size =
                shape[2];

            for (size_t pos = 0;
                 pos < sequence_length;
                 ++pos) {

                size_t target_token =
                    target_data[pos];

                std::vector<float> values(
                    vocab_size
                );

                for (size_t v = 0;
                     v < vocab_size;
                     ++v) {

                    values[v] =
                        logits_cpu.Data()[
                            pos * vocab_size + v
                        ];
                }

                // --------------------------------------------
                // Cross entropy
                // --------------------------------------------

                float max_logit =
                    *std::max_element(
                        values.begin(),
                        values.end()
                    );

                double sum_exp = 0.0;

                for (float value : values) {

                    sum_exp +=
                        std::exp(
                            static_cast<double>(
                                value - max_logit
                            )
                        );
                }

                double log_sum_exp =
                    static_cast<double>(
                        max_logit
                    ) +
                    std::log(sum_exp);

                double loss =
                    log_sum_exp -
                    static_cast<double>(
                        values[target_token]
                    );

                total_loss += loss;

                // --------------------------------------------
                // Rank
                // --------------------------------------------

                float target_logit =
                    values[target_token];

                size_t rank = 1;

                for (float value : values) {

                    if (value > target_logit) {
                        ++rank;
                    }
                }

                total_rank +=
                    static_cast<double>(rank);

                if (rank == 1) {
                    ++top1;
                }

                if (rank <= 5) {
                    ++top5;
                }

                if (rank <= 10) {
                    ++top10;
                }

                ++total_tokens;
            }

            if ((batch + 1) % 5 == 0) {

                double ce =
                    total_loss /
                    static_cast<double>(
                        total_tokens
                    );

                std::cout
                    << "Batch "
                    << (batch + 1)
                    << "/"
                    << VALIDATION_BATCHES
                    << " | CE: "
                    << ce
                    << "\n";
            }
        }

        // ----------------------------------------------------
        // Results
        // ----------------------------------------------------

        double cross_entropy =
            total_loss /
            static_cast<double>(
                total_tokens
            );

        double perplexity =
            std::exp(cross_entropy);

        double top1_accuracy =
            100.0 *
            static_cast<double>(top1) /
            static_cast<double>(total_tokens);

        double top5_accuracy =
            100.0 *
            static_cast<double>(top5) /
            static_cast<double>(total_tokens);

        double top10_accuracy =
            100.0 *
            static_cast<double>(top10) /
            static_cast<double>(total_tokens);

        double average_rank =
            total_rank /
            static_cast<double>(total_tokens);

        std::cout
            << "\n========================================\n"
            << "                RESULTS\n"
            << "========================================\n";

        std::cout
            << "Tokens tested: "
            << total_tokens
            << "\n";

        std::cout
            << "Top-1 accuracy: "
            << top1_accuracy
            << "%\n";

        std::cout
            << "Top-5 accuracy: "
            << top5_accuracy
            << "%\n";

        std::cout
            << "Top-10 accuracy: "
            << top10_accuracy
            << "%\n";

        std::cout
            << "Average rank: "
            << average_rank
            << "\n";

        std::cout
            << "Cross entropy: "
            << cross_entropy
            << "\n";

        std::cout
            << "Perplexity: "
            << perplexity
            << "\n";

        std::cout
            << "\n========================================\n"
            << "       TEACHER RANK TEST PASSED\n"
            << "========================================\n";
    }
    catch (const std::exception& exception) {

        std::cerr
            << "\n========================================\n"
            << "                ERROR\n"
            << "========================================\n";

        std::cerr
            << exception.what()
            << "\n";

        return 1;
    }

    return 0;
}