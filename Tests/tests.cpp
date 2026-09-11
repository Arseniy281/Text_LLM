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
const size_t TEST_STEPS = 50;

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
// Create CUDA input
//
// Shape:
// [1, seq]
//
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
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n";
        std::cout
            << "GREEDY VS TEACHER-FORCED TEST\n";
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

        // ====================================================
        // Create TWO identical models
        //
        // Teacher model:
        // uses real previous tokens.
        //
        // Greedy model:
        // uses its own predictions.
        //
        // They must have identical weights.
        // ====================================================

        LanguageModel teacher_model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        LanguageModel greedy_model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        std::cout
            << "[OK] Models created.\n";

        teacher_model.LoadModel(
            MODEL_PATH
        );

        greedy_model.LoadModel(
            MODEL_PATH
        );

        std::cout
            << "[OK] Models loaded.\n\n";

        // ----------------------------------------------------
        // Enable KV cache
        // ----------------------------------------------------

        teacher_model.SetUseKVCache(true);
        greedy_model.SetUseKVCache(true);

        teacher_model.ResetCache();
        greedy_model.ResetCache();

        // ====================================================
        // Prompt
        // ====================================================

        std::vector<size_t> prompt;

        for (size_t i = 0;
             i < PROMPT_SIZE;
             ++i) {

            prompt.push_back(
                tokens[start + i]
            );
        }

        // ====================================================
        // Initialize both models with the same prompt
        // ====================================================

        std::vector<float> teacher_logits;
        std::vector<float> greedy_logits;

        for (size_t i = 0;
             i < prompt.size();
             ++i) {

            auto teacher_input =
                CreateInput({
                    prompt[i]
                });

            auto greedy_input =
                CreateInput({
                    prompt[i]
                });

            auto teacher_output =
                teacher_model.forward(
                    teacher_input
                );

            auto greedy_output =
                greedy_model.forward(
                    greedy_input
                );

            cudaDeviceSynchronize();

            if (i + 1 == prompt.size()) {

                teacher_logits =
                    CopyLastLogitsToCPU(
                        teacher_output
                    );

                greedy_logits =
                    CopyLastLogitsToCPU(
                        greedy_output
                    );
            }
        }

        // ----------------------------------------------------
        // Check that both models agree after prompt
        // ----------------------------------------------------

        size_t teacher_first =
            ArgMax(teacher_logits);

        size_t greedy_first =
            ArgMax(greedy_logits);

        float max_difference = 0.0f;

        for (size_t i = 0;
             i < teacher_logits.size();
             ++i) {

            max_difference =
                std::max(
                    max_difference,
                    std::abs(
                        teacher_logits[i] -
                        greedy_logits[i]
                    )
                );
        }

        std::cout
            << "========================================\n";
        std::cout
            << "PROMPT COMPARISON\n";
        std::cout
            << "========================================\n\n";

        std::cout
            << "Teacher prediction: "
            << teacher_first
            << "\n";

        std::cout
            << "Greedy prediction:  "
            << greedy_first
            << "\n";

        std::cout
            << "Max logits difference: "
            << max_difference
            << "\n\n";

        if (teacher_first != greedy_first) {

            throw std::runtime_error(
                "Models disagree after identical prompt"
            );
        }

        if (max_difference > 1e-4f) {

            throw std::runtime_error(
                "Model outputs differ after identical prompt"
            );
        }

        std::cout
            << "[OK] Models are identical.\n\n";

        // ====================================================
        // Autoregressive comparison
        // ====================================================

        size_t teacher_correct = 0;
        size_t greedy_correct = 0;

        size_t first_greedy_mismatch =
            TEST_STEPS;

        std::vector<size_t> greedy_tokens;

        greedy_tokens.reserve(
            TEST_STEPS
        );

        std::cout
            << "========================================\n";
        std::cout
            << "AUTOREGRESSIVE COMPARISON\n";
        std::cout
            << "========================================\n\n";

        std::cout
            << "Step | Real | Teacher | Greedy | Status\n";
        std::cout
            << "----------------------------------------\n";

        for (size_t step = 0;
             step < TEST_STEPS;
             ++step) {

            // ------------------------------------------------
            // The real next token
            // ------------------------------------------------

            size_t real_token =
                tokens[
                    start +
                    PROMPT_SIZE +
                    step
                ];

            // ------------------------------------------------
            // Teacher prediction
            //
            // Uses the REAL previous token.
            // ------------------------------------------------

            size_t teacher_prediction =
                ArgMax(
                    teacher_logits
                );

            // ------------------------------------------------
            // Greedy prediction
            //
            // Uses its OWN previous prediction.
            // ------------------------------------------------

            size_t greedy_prediction =
                ArgMax(
                    greedy_logits
                );

            bool teacher_ok =
                teacher_prediction ==
                real_token;

            bool greedy_ok =
                greedy_prediction ==
                real_token;

            if (teacher_ok) {
                ++teacher_correct;
            }

            if (greedy_ok) {
                ++greedy_correct;
            }

            if (!greedy_ok &&
                first_greedy_mismatch ==
                    TEST_STEPS) {

                first_greedy_mismatch =
                    step;
            }

            greedy_tokens.push_back(
                greedy_prediction
            );

            std::cout
                << step
                << " | "
                << real_token
                << " | "
                << teacher_prediction
                << " | "
                << greedy_prediction
                << " | ";

            if (greedy_ok) {

                std::cout
                    << "OK";

            } else {

                std::cout
                    << "MISMATCH";
            }

            std::cout
                << "\n";

            // ------------------------------------------------
            // Teacher forcing:
            //
            // Feed REAL token.
            // ------------------------------------------------

            auto teacher_input =
                CreateInput({
                    real_token
                });

            auto teacher_output =
                teacher_model.forward(
                    teacher_input
                );

            // ------------------------------------------------
            // Greedy:
            //
            // Feed MODEL prediction.
            // ------------------------------------------------

            auto greedy_input =
                CreateInput({
                    greedy_prediction
                });

            auto greedy_output =
                greedy_model.forward(
                    greedy_input
                );

            cudaDeviceSynchronize();

            teacher_logits =
                CopyLastLogitsToCPU(
                    teacher_output
                );

            greedy_logits =
                CopyLastLogitsToCPU(
                    greedy_output
                );
        }

        // ====================================================
        // Results
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
            << "Teacher Top-1: "
            << 100.0 *
                static_cast<double>(
                    teacher_correct
                ) /
                static_cast<double>(
                    TEST_STEPS
                )
            << "%\n";

        std::cout
            << "Greedy accuracy: "
            << 100.0 *
                static_cast<double>(
                    greedy_correct
                ) /
                static_cast<double>(
                    TEST_STEPS
                )
            << "%\n";

        if (first_greedy_mismatch <
            TEST_STEPS) {

            std::cout
                << "First greedy mismatch: step "
                << first_greedy_mismatch
                << "\n";

        } else {

            std::cout
                << "First greedy mismatch: none\n";
        }

        std::cout
            << "\nGenerated token IDs:\n";

        for (size_t token :
             greedy_tokens) {

            std::cout
                << token
                << " ";
        }

        std::cout
            << "\n\n";

        // ====================================================
        // Decode greedy output
        // ====================================================

        std::string generated =
            tokenizer.Decode(
                greedy_tokens
            );

        std::cout
            << "Greedy generated text:\n";
        std::cout
            << generated
            << "\n";

        std::cout
            << "\n========================================\n";

        teacher_model.SetUseKVCache(false);
        greedy_model.SetUseKVCache(false);

        teacher_model.ResetCache();
        greedy_model.ResetCache();

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