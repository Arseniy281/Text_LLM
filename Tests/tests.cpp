#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
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
const size_t TEST_STEPS = 50;

// Та же позиция, что и в предыдущем тесте.
const size_t VALIDATION_OFFSET = 14136;

// ============================================================
// Read file
// ============================================================

std::string ReadFile(
    const std::string& path
) {
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
// Create CUDA input
//
// Shape:
// [1, seq]
// ============================================================

std::shared_ptr<Tensor> CreateInput(
    const std::vector<size_t>& tokens
) {
    auto cpu =
        std::make_shared<Tensor>(
            std::vector<size_t>{
                1,
                tokens.size()
            }
        );

    for (size_t i = 0;
         i < tokens.size();
         ++i) {

        cpu->at({0, i}) =
            static_cast<float>(
                tokens[i]
            );
    }

    auto gpu =
        std::make_shared<Tensor>(
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
// CUDA logits -> CPU
// ============================================================

std::vector<float> CopyLastLogitsToCPU(
    const std::shared_ptr<Tensor>& logits
) {
    const auto& shape =
        logits->GetShape();

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
    size_t vocab_size = shape[2];

    if (vocab_size != VOCAB_SIZE) {
        throw std::runtime_error(
            "Unexpected vocabulary size"
        );
    }

    std::vector<float> result(
        vocab_size
    );

    size_t offset =
        (seq_len - 1) * vocab_size;

    cudaError_t error =
        cudaMemcpy(
            result.data(),
            logits->Data() + offset,
            vocab_size * sizeof(float),
            cudaMemcpyDeviceToHost
        );

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string(
                "cudaMemcpy failed: "
            ) +
            cudaGetErrorString(error)
        );
    }

    return result;
}

// ============================================================
// ArgMax
// ============================================================

size_t ArgMax(
    const std::vector<float>& values
) {
    size_t best = 0;

    for (size_t i = 1;
         i < values.size();
         ++i) {

        if (values[i] > values[best]) {
            best = i;
        }
    }

    return best;
}

// ============================================================
// Softmax
//
// Делается на CPU только для анализа.
// ============================================================

std::vector<float> Softmax(
    const std::vector<float>& logits
) {
    std::vector<float> probs(
        logits.size()
    );

    float max_logit =
        *std::max_element(
            logits.begin(),
            logits.end()
        );

    double sum = 0.0;

    for (size_t i = 0;
         i < logits.size();
         ++i) {

        probs[i] =
            std::exp(
                logits[i] - max_logit
            );

        sum += probs[i];
    }

    for (size_t i = 0;
         i < probs.size();
         ++i) {

        probs[i] =
            static_cast<float>(
                probs[i] / sum
            );
    }

    return probs;
}

// ============================================================
// Rank настоящего токена
//
// Rank 1 = лучший токен.
// Rank 2 = второй лучший и т.д.
// ============================================================

size_t GetRank(
    const std::vector<float>& logits,
    size_t target
) {
    float target_value =
        logits[target];

    size_t rank = 1;

    for (size_t i = 0;
         i < logits.size();
         ++i) {

        if (logits[i] > target_value) {
            ++rank;
        }
    }

    return rank;
}

// ============================================================
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n";
        std::cout
            << "TEACHER-FORCED TOKEN RANK TEST\n";
        std::cout
            << "========================================\n\n";

        // ----------------------------------------------------
        // CUDA
        // ----------------------------------------------------

        int device_count = 0;

        cudaError_t error =
            cudaGetDeviceCount(
                &device_count
            );

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
        // Validation position
        // ----------------------------------------------------

        size_t train_size =
            static_cast<size_t>(
                tokens.size() * 0.9
            );

        size_t validation_start =
            train_size;

        size_t start =
            validation_start +
            VALIDATION_OFFSET;

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
            << "[OK] Model loaded.\n";

        // ----------------------------------------------------
        // KV cache
        //
        // Используем тот же режим, что и при генерации.
        // ----------------------------------------------------

        model.SetUseKVCache(true);
        model.ResetCache();

        // ====================================================
        // Feed prompt
        // ====================================================

        std::cout
            << "\nFeeding prompt...\n";

        for (size_t i = 0;
             i < PROMPT_SIZE;
             ++i) {

            size_t token =
                tokens[start + i];

            auto input =
                CreateInput({
                    token
                });

            model.forward(input);

            cudaDeviceSynchronize();
        }

        std::cout
            << "[OK] Prompt processed.\n\n";

        // ====================================================
        // Statistics
        // ====================================================

        size_t top1 = 0;
        size_t top5 = 0;
        size_t top10 = 0;

        double rank_sum = 0.0;
        double cross_entropy = 0.0;

        // ====================================================
        // Header
        // ====================================================

        std::cout
            << "==============================================================\n";

        std::cout
            << "Step | Real | Pred | Rank | Top-1 | Top-5 | Top-10"
            << " | P(real) | P(pred)\n";

        std::cout
            << "--------------------------------------------------------------\n";

        // ====================================================
        // Teacher-forced test
        // ====================================================

        for (size_t step = 0;
             step < TEST_STEPS;
             ++step) {

            // ------------------------------------------------
            // Real next token
            // ------------------------------------------------

            size_t real_token =
                tokens[
                    start +
                    PROMPT_SIZE +
                    step
                ];

            // ------------------------------------------------
            // IMPORTANT:
            //
            // We need logits produced from the CURRENT
            // correct context BEFORE feeding real_token.
            //
            // Therefore we have to get the output from
            // the previous forward.
            //
            // To keep the logic clean, for the first step
            // we need to process the last prompt token again
            // and use its logits.
            // ------------------------------------------------

            std::vector<float> logits;

            if (step == 0) {

                // The prompt loop above already processed
                // the last prompt token, but did not save logits.
                // Rebuild the model state from scratch so that
                // we can obtain the exact logits.

                model.SetUseKVCache(false);

                model.ResetCache();

                auto prompt_input =
                    CreateInput(
                        std::vector<size_t>(
                            tokens.begin() + start,
                            tokens.begin() +
                            start +
                            PROMPT_SIZE
                        )
                    );

                auto output =
                    model.forward(
                        prompt_input
                    );

                cudaDeviceSynchronize();

                logits =
                    CopyLastLogitsToCPU(
                        output
                    );

                model.SetUseKVCache(true);
                model.ResetCache();

                // Re-feed prompt token-by-token
                // to initialize KV cache again.

                for (size_t i = 0;
                     i < PROMPT_SIZE;
                     ++i) {

                    auto input =
                        CreateInput({
                            tokens[start + i]
                        });

                    model.forward(input);

                    cudaDeviceSynchronize();
                }
            }
            else {

                // For subsequent steps logits were saved
                // after feeding the previous real token.
                //
                // They will be stored below.
            }

            // ------------------------------------------------
            // For step > 0, logits are already available
            // from previous iteration.
            // ------------------------------------------------

            static std::vector<float> saved_logits;

            if (step == 0) {
                saved_logits = logits;
            }

            logits = saved_logits;

            // ------------------------------------------------
            // Prediction
            // ------------------------------------------------

            size_t prediction =
                ArgMax(logits);

            // ------------------------------------------------
            // Rank
            // ------------------------------------------------

            size_t rank =
                GetRank(
                    logits,
                    real_token
                );

            // ------------------------------------------------
            // Probability
            // ------------------------------------------------

            std::vector<float> probs =
                Softmax(logits);

            float real_probability =
                probs[real_token];

            float prediction_probability =
                probs[prediction];

            // ------------------------------------------------
            // Statistics
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

            rank_sum +=
                static_cast<double>(rank);

            // Cross entropy:
            //
            // -log(P(real))
            // ------------------------------------------------

            cross_entropy -=
                std::log(
                    std::max(
                        static_cast<double>(
                            real_probability
                        ),
                        1e-12
                    )
                );

            // ------------------------------------------------
            // Print
            // ------------------------------------------------

            std::cout
                << std::setw(4)
                << step
                << " | "
                << std::setw(4)
                << real_token
                << " | "
                << std::setw(4)
                << prediction
                << " | "
                << std::setw(4)
                << rank
                << " | ";

            if (rank == 1) {
                std::cout
                    << " YES ";
            } else {
                std::cout
                    << "  NO ";
            }

            std::cout
                << " | ";

            if (rank <= 5) {
                std::cout
                    << " YES ";
            } else {
                std::cout
                    << "  NO ";
            }

            std::cout
                << " | ";

            if (rank <= 10) {
                std::cout
                    << " YES ";
            } else {
                std::cout
                    << "  NO ";
            }

            std::cout
                << " | "
                << std::fixed
                << std::setprecision(4)
                << real_probability
                << " | "
                << prediction_probability
                << "\n";

            // ------------------------------------------------
            // Feed the REAL token.
            //
            // This is the important teacher-forcing part.
            // ------------------------------------------------

            auto input =
                CreateInput({
                    real_token
                });

            auto output =
                model.forward(
                    input
                );

            cudaDeviceSynchronize();

            saved_logits =
                CopyLastLogitsToCPU(
                    output
                );
        }

        // ====================================================
        // Final statistics
        // ====================================================

        double top1_accuracy =
            100.0 *
            static_cast<double>(top1) /
            static_cast<double>(TEST_STEPS);

        double top5_accuracy =
            100.0 *
            static_cast<double>(top5) /
            static_cast<double>(TEST_STEPS);

        double top10_accuracy =
            100.0 *
            static_cast<double>(top10) /
            static_cast<double>(TEST_STEPS);

        double average_rank =
            rank_sum /
            static_cast<double>(TEST_STEPS);

        cross_entropy /=
            static_cast<double>(TEST_STEPS);

        double perplexity =
            std::exp(
                cross_entropy
            );

        // ====================================================
        // Result
        // ====================================================

        std::cout
            << "\n";
        std::cout
            << "========================================\n";
        std::cout
            << "RESULT\n";
        std::cout
            << "========================================\n";

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

        std::cout
            << "\n";

        // ====================================================
        // Interpretation helpers
        // ====================================================

        if (top1_accuracy >= 60.0) {

            std::cout
                << "[OK] Top-1 prediction is strong.\n";

        } else {

            std::cout
                << "[INFO] Top-1 prediction still has room "
                << "for improvement.\n";
        }

        if (top10_accuracy >= 85.0) {

            std::cout
                << "[OK] Most real tokens are inside Top-10.\n";

        } else {

            std::cout
                << "[INFO] Many real tokens are outside Top-10.\n";
        }

        // ====================================================
        // Cleanup
        // ====================================================

        model.SetUseKVCache(false);
        model.ResetCache();

        std::cout
            << "\n========================================\n";

        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "\n[ERROR] "
            << e.what()
            << "\n";

        return 1;
    }
}