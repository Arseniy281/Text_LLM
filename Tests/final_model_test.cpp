#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"
#include "../Engine/Tensor/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <fstream>
#include <random>
#include <cmath>
#include <algorithm>
#include <memory>
#include <iomanip>

// ============================================================
// Настройки
// ============================================================

const size_t VOCAB_SIZE = 1000;

const size_t EMBED_DIM = 128;
const size_t BLOCKS = 4;
const size_t HEADS = 4;
const size_t HIDDEN = 512;

const size_t CONTEXT = 128;

// 20 окон по 128 токенов.
// Всего будет проверено 2560 предсказаний.
const size_t VALIDATION_BATCHES = 20;

const unsigned int SEED = 42;

const std::string TOKENIZER_PATH =
    "/content/Text_LLM/Models/MargaritaTokenizer";

const std::string MODEL_PATH =
    "/content/Text_LLM/Models/MargaritaSmaller_clean_best";

const std::string CORPUS_PATH =
    "/content/Text_LLM/Data/master_and_margarita.txt";

// ============================================================
// Load text
// ============================================================

std::string LoadText(
    const std::string& path
) {

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
// CUDA error helper
// ============================================================

void CheckCUDA(
    cudaError_t error,
    const std::string& operation
) {

    if (error != cudaSuccess) {

        throw std::runtime_error(
            operation + ": " +
            cudaGetErrorString(error)
        );
    }
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

        // ====================================================
        // CUDA
        // ====================================================

        int device_count = 0;

        CheckCUDA(
            cudaGetDeviceCount(&device_count),
            "cudaGetDeviceCount failed"
        );

        if (device_count == 0) {
            throw std::runtime_error(
                "No CUDA devices found"
            );
        }

        cudaDeviceProp properties{};

        CheckCUDA(
            cudaGetDeviceProperties(
                &properties,
                0
            ),
            "cudaGetDeviceProperties failed"
        );

        std::cout
            << "CUDA devices: "
            << device_count
            << "\n";

        std::cout
            << "GPU: "
            << properties.name
            << "\n\n";

        // ====================================================
        // Configuration
        // ====================================================

        std::cout
            << "========================================\n"
            << "             MODEL CONFIG\n"
            << "========================================\n";

        std::cout
            << "Vocabulary: "
            << VOCAB_SIZE
            << "\n";

        std::cout
            << "Embedding:  "
            << EMBED_DIM
            << "\n";

        std::cout
            << "Blocks:     "
            << BLOCKS
            << "\n";

        std::cout
            << "Heads:      "
            << HEADS
            << "\n";

        std::cout
            << "Hidden:     "
            << HIDDEN
            << "\n";

        std::cout
            << "Context:    "
            << CONTEXT
            << "\n";

        std::cout
            << "Validation batches: "
            << VALIDATION_BATCHES
            << "\n";

        std::cout
            << "Validation tokens:  "
            << VALIDATION_BATCHES * CONTEXT
            << "\n\n";

        // ====================================================
        // Tokenizer
        // ====================================================

        std::cout
            << "========================================\n"
            << "             TOKENIZER\n"
            << "========================================\n";

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

        std::cout
            << "Vocabulary size: "
            << tokenizer.GetVocabSize()
            << "\n\n";

        // ====================================================
        // Corpus
        // ====================================================

        std::cout
            << "========================================\n"
            << "              CORPUS\n"
            << "========================================\n";

        std::string text =
            LoadText(CORPUS_PATH);

        std::cout
            << "Characters: "
            << text.size()
            << "\n";

        std::vector<size_t> tokens =
            tokenizer.Encode(text);

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n\n";

        // ====================================================
        // Train / validation split
        // ====================================================

        size_t train_size =
            tokens.size() * 9 / 10;

        size_t validation_start =
            train_size;

        size_t validation_end =
            tokens.size();

        if (validation_end <=
            validation_start + CONTEXT) {

            throw std::runtime_error(
                "Validation set is too small"
            );
        }

        std::cout
            << "Train tokens: "
            << train_size
            << "\n";

        std::cout
            << "Validation tokens: "
            << validation_end -
               validation_start
            << "\n\n";

        // ====================================================
        // Model
        // ====================================================

        std::cout
            << "========================================\n"
            << "          LOADING BEST MODEL\n"
            << "========================================\n";

        LanguageModel model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        model.SetUseKVCache(false);

        std::cout
            << "Checkpoint:\n"
            << MODEL_PATH
            << "\n\n";

        model.LoadModel(
            MODEL_PATH
        );

        CheckCUDA(
            cudaDeviceSynchronize(),
            "cudaDeviceSynchronize after model load"
        );

        std::cout
            << "[OK] Model loaded.\n\n";

        // ====================================================
        // Validation windows
        // ====================================================

        std::mt19937 rng(SEED);

        size_t max_start =
            validation_end -
            CONTEXT -
            1;

        std::uniform_int_distribution<size_t>
            distribution(
                validation_start,
                max_start
            );

        // ====================================================
        // Statistics
        // ====================================================

        double total_loss = 0.0;

        double total_rank = 0.0;

        size_t total_tokens = 0;

        size_t top1 = 0;
        size_t top5 = 0;
        size_t top10 = 0;

        // ====================================================
        // Evaluation
        // ====================================================

        std::cout
            << "========================================\n"
            << "          TEACHER-FORCED EVALUATION\n"
            << "========================================\n\n";

        for (size_t batch = 0;
             batch < VALIDATION_BATCHES;
             ++batch) {

            // ------------------------------------------------
            // Select validation window
            // ------------------------------------------------

            size_t start =
                distribution(rng);

            std::vector<float> input_data(
                CONTEXT
            );

            std::vector<float> target_data(
                CONTEXT
            );

            for (size_t i = 0;
                 i < CONTEXT;
                 ++i) {

                input_data[i] =
                    static_cast<float>(
                        tokens[start + i]
                    );

                target_data[i] =
                    static_cast<float>(
                        tokens[start + i + 1]
                    );
            }

            // ------------------------------------------------
            // CPU tensors
            // ------------------------------------------------

            Tensor input_cpu(
                {1, CONTEXT},
                std::move(input_data)
            );

            Tensor target_cpu(
                {1, CONTEXT},
                std::move(target_data)
            );

            // ------------------------------------------------
            // CUDA tensors
            // ------------------------------------------------

            Tensor input(
                {1, CONTEXT},
                Device::CUDA
            );

            Tensor target(
                {1, CONTEXT},
                Device::CUDA
            );

            input_cpu.CopyToCUDA(input);
            target_cpu.CopyToCUDA(target);

            // ------------------------------------------------
            // Forward
            // ------------------------------------------------

            auto input_ptr =
                std::make_shared<Tensor>(
                    std::move(input)
                );

            auto logits =
                model.forward(input_ptr);

            if (logits->GetDevice() !=
                Device::CUDA) {

                throw std::runtime_error(
                    "Model returned non-CUDA logits"
                );
            }

            // ------------------------------------------------
            // CUDA -> CPU
            // ------------------------------------------------

            Tensor logits_cpu(
                logits->GetShape(),
                Device::CPU
            );

            CheckCUDA(
                cudaMemcpy(
                    logits_cpu.Data(),
                    logits->Data(),
                    logits->GetSize() * sizeof(float),
                    cudaMemcpyDeviceToHost
                ),
                "cudaMemcpy logits D2H failed"
            );

            // ------------------------------------------------
            // Shape
            // ------------------------------------------------

            const auto& shape =
                logits->GetShape();

            size_t sequence_length =
                shape[1];

            size_t vocab_size =
                shape[2];

            if (sequence_length != CONTEXT) {
                throw std::runtime_error(
                    "Unexpected logits sequence length"
                );
            }

            if (vocab_size != VOCAB_SIZE) {
                throw std::runtime_error(
                    "Unexpected logits vocabulary size"
                );
            }

            // =================================================
            // Every token in the window
            // =================================================

            for (size_t pos = 0;
                 pos < sequence_length;
                 ++pos) {

                size_t target_token =
                    static_cast<size_t>(
                        target_cpu.Data()[pos]
                    );

                // ------------------------------------------------
                // Copy logits for this position
                // ------------------------------------------------

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

                // =================================================
                // Cross entropy
                // =================================================

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

                double token_loss =
                    log_sum_exp -
                    static_cast<double>(
                        values[target_token]
                    );

                total_loss +=
                    token_loss;

                // =================================================
                // Rank
                // =================================================

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

                // ------------------------------------------------
                // Accuracy
                // ------------------------------------------------

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

            // ------------------------------------------------
            // Progress
            // ------------------------------------------------

            if ((batch + 1) % 5 == 0) {

                double current_ce =
                    total_loss /
                    static_cast<double>(
                        total_tokens
                    );

                std::cout
                    << "Batch "
                    << std::setw(2)
                    << batch + 1
                    << "/"
                    << VALIDATION_BATCHES
                    << " | Tokens: "
                    << total_tokens
                    << " | CE: "
                    << std::fixed
                    << std::setprecision(6)
                    << current_ce
                    << "\n";
            }
        }

        // ====================================================
        // Results
        // ====================================================

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

        // ====================================================
        // Results
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "                RESULTS\n"
            << "========================================\n";

        std::cout
            << std::fixed
            << std::setprecision(4);

        std::cout
            << "Model: "
            << MODEL_PATH
            << "\n";

        std::cout
            << "Tokens tested: "
            << total_tokens
            << "\n\n";

        std::cout
            << "Top-1 accuracy:  "
            << top1_accuracy
            << "%\n";

        std::cout
            << "Top-5 accuracy:  "
            << top5_accuracy
            << "%\n";

        std::cout
            << "Top-10 accuracy: "
            << top10_accuracy
            << "%\n";

        std::cout
            << "Average rank:    "
            << average_rank
            << "\n";

        std::cout
            << "Cross entropy:   "
            << cross_entropy
            << "\n";

        std::cout
            << "Perplexity:      "
            << perplexity
            << "\n";

        // ====================================================
        // Comparison
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "          PREVIOUS MODEL\n"
            << "========================================\n";

        std::cout
            << "Top-1:        23.3984%\n"
            << "Top-5:        44.1797%\n"
            << "Top-10:       53.4375%\n"
            << "Average rank: 53.6102\n"
            << "CE:           4.06908\n"
            << "PPL:          58.5029\n";

        std::cout
            << "\n========================================\n"
            << "       TEACHER RANK TEST FINISHED\n"
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