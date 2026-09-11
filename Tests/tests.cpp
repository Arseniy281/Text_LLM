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
// Rank of target token
//
// Rank 1 = best prediction
// ============================================================

size_t GetRank(
    const std::vector<float>& logits,
    size_t target
) {
    if (target >= logits.size()) {
        throw std::runtime_error(
            "Target token is outside vocabulary"
        );
    }

    size_t rank = 1;

    for (size_t i = 0;
         i < logits.size();
         ++i) {

        if (logits[i] > logits[target]) {
            ++rank;
        }
    }

    return rank;
}

// ============================================================
// Cross entropy for one target
//
// CE = log(sum(exp(logits))) - logits[target]
// ============================================================

float CrossEntropy(
    const std::vector<float>& logits,
    size_t target
) {
    if (target >= logits.size()) {
        throw std::runtime_error(
            "Target token is outside vocabulary"
        );
    }

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
        // Process prompt
        // ----------------------------------------------------

        std::cout
            << "Processing prompt...\n";

        std::vector<float> current_logits;

        for (size_t i = 0;
             i < prompt.size();
             ++i) {

            auto input =
                CreateInput({
                    prompt[i]
                });

            auto output =
                model.forward(input);

            cudaDeviceSynchronize();

            if (i + 1 == prompt.size()) {

                current_logits =
                    CopyLastLogitsToCPU(
                        output
                    );
            }
        }

        std::cout
            << "[OK] Prompt processed.\n\n";

        // ====================================================
        // Teacher-forced validation
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
            // Real next token
            //
            // We predict this token using the current
            // real context.
            // ------------------------------------------------

            size_t target =
                tokens[
                    start +
                    PROMPT_SIZE +
                    step
                ];

            // ------------------------------------------------
            // Prediction
            // ------------------------------------------------

            size_t prediction =
                ArgMax(
                    current_logits
                );

            size_t rank =
                GetRank(
                    current_logits,
                    target
                );

            float loss =
                CrossEntropy(
                    current_logits,
                    target
                );

            // ------------------------------------------------
            // Metrics
            // ------------------------------------------------

            if (prediction == target) {
                ++top1_hits;
            }

            if (rank <= 5) {
                ++top5_hits;
            }

            if (rank <= 10) {
                ++top10_hits;
            }

            total_rank +=
                static_cast<double>(
                    rank
                );

            total_loss +=
                static_cast<double>(
                    loss
                );

            // ------------------------------------------------
            // First 10 steps
            // ------------------------------------------------

            if (step < 10) {

                std::cout
                    << "Step "
                    << step
                    << ":\n";

                std::cout
                    << "  Expected: "
                    << target
                    << "\n";

                std::cout
                    << "  Greedy:   "
                    << prediction
                    << "\n";

                std::cout
                    << "  Rank:     "
                    << rank
                    << "\n";

                std::cout
                    << "  Loss:     "
                    << loss
                    << "\n";

                if (prediction == target) {

                    std::cout
                        << "  [TOP-1]\n";

                } else if (rank <= 5) {

                    std::cout
                        << "  [TOP-5]\n";

                } else if (rank <= 10) {

                    std::cout
                        << "  [TOP-10]\n";

                } else {

                    std::cout
                        << "  [MISS]\n";
                }

                std::cout
                    << "\n";
            }

            // ------------------------------------------------
            // Teacher forcing
            //
            // IMPORTANT:
            //
            // We feed the REAL target token, not the
            // model prediction.
            // ------------------------------------------------

            auto input =
                CreateInput({
                    target
                });

            auto output =
                model.forward(input);

            cudaDeviceSynchronize();

            current_logits =
                CopyLastLogitsToCPU(
                    output
                );
        }

        // ====================================================
        // Result
        // ====================================================

        double average_loss =
            total_loss /
            static_cast<double>(
                TEST_STEPS
            );

        double average_rank =
            total_rank /
            static_cast<double>(
                TEST_STEPS
            );

        double perplexity =
            std::exp(
                average_loss
            );

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
            << 100.0 *
                static_cast<double>(
                    top1_hits
                ) /
                static_cast<double>(
                    TEST_STEPS
                )
            << "%\n";

        std::cout
            << "Top-5 accuracy:  "
            << 100.0 *
                static_cast<double>(
                    top5_hits
                ) /
                static_cast<double>(
                    TEST_STEPS
                )
            << "%\n";

        std::cout
            << "Top-10 accuracy: "
            << 100.0 *
                static_cast<double>(
                    top10_hits
                ) /
                static_cast<double>(
                    TEST_STEPS
                )
            << "%\n";

        std::cout
            << "Average rank:    "
            << average_rank
            << "\n";

        std::cout
            << "Cross entropy:   "
            << average_loss
            << "\n";

        std::cout
            << "Perplexity:      "
            << perplexity
            << "\n";

        std::cout
            << "========================================\n";

        // ----------------------------------------------------
        // Cleanup
        // ----------------------------------------------------

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