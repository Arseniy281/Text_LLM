#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// Настройки
// ============================================================

const std::string TOKENIZER_PATH =
    "../Models/MargaritaTokenizer";

const std::string CORPUS_PATH =
    "../Data/master_and_margarita.txt";

const std::string MODEL_PATH =
    "../Models/MargaritaLR2000_final";

const size_t VOCAB_SIZE = 1000;

const size_t EMBED_DIM = 128;
const size_t BLOCKS = 4;
const size_t HEADS = 4;
const size_t HIDDEN = 512;

const size_t CONTEXT = 128;
const size_t BATCH_SIZE = 8;

const size_t STEPS = 2000;

// ============================================================
// Learning rate
// ============================================================

const float MAX_LR = 0.001f;
const float MIN_LR = 0.0001f;

const size_t WARMUP_STEPS = 200;

// ============================================================
// Validation
// ============================================================

const size_t VALIDATION_EVERY = 100;
const size_t VALIDATION_BATCHES = 20;

// ============================================================
// Random
// ============================================================

const unsigned int SEED = 42;

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
// Learning rate schedule
//
// Warmup:
//     0 -> MAX_LR
//
// Cosine decay:
//     MAX_LR -> MIN_LR
// ============================================================

float GetLearningRate(
    size_t step
) {
    if (step < WARMUP_STEPS) {

        return MAX_LR *
            static_cast<float>(step + 1) /
            static_cast<float>(WARMUP_STEPS);
    }

    float progress =
        static_cast<float>(
            step - WARMUP_STEPS
        ) /
        static_cast<float>(
            STEPS - WARMUP_STEPS
        );

    progress =
        std::clamp(
            progress,
            0.0f,
            1.0f
        );

    return MIN_LR +
        0.5f *
        (MAX_LR - MIN_LR) *
        (1.0f +
            std::cos(
                static_cast<float>(M_PI) *
                progress
            ));
}

// ============================================================
// Create CUDA tensor from token IDs
//
// Shape:
// [BATCH, CONTEXT]
// ============================================================

Tensor CreateInput(
    const std::vector<size_t>& tokens
) {
    if (tokens.size() !=
        BATCH_SIZE * CONTEXT) {

        throw std::runtime_error(
            "Invalid input token count"
        );
    }

    Tensor cpu(
        {BATCH_SIZE, CONTEXT},
        0.0f
    );

    for (size_t i = 0;
         i < tokens.size();
         ++i) {

        size_t b =
            i / CONTEXT;

        size_t t =
            i % CONTEXT;

        cpu.at({b, t}) =
            static_cast<float>(
                tokens[i]
            );
    }

    Tensor gpu(
        {BATCH_SIZE, CONTEXT},
        0.0f,
        Device::CUDA
    );

    cpu.CopyToCUDA(gpu);

    return gpu;
}

// ============================================================
// Create target tensor
//
// Shape:
// [BATCH, CONTEXT]
// ============================================================

Tensor CreateTarget(
    const std::vector<size_t>& tokens
) {
    if (tokens.size() !=
        BATCH_SIZE * CONTEXT) {

        throw std::runtime_error(
            "Invalid target token count"
        );
    }

    Tensor cpu(
        {BATCH_SIZE, CONTEXT},
        0.0f
    );

    for (size_t i = 0;
         i < tokens.size();
         ++i) {

        size_t b =
            i / CONTEXT;

        size_t t =
            i % CONTEXT;

        cpu.at({b, t}) =
            static_cast<float>(
                tokens[i]
            );
    }

    Tensor gpu(
        {BATCH_SIZE, CONTEXT},
        0.0f,
        Device::CUDA
    );

    cpu.CopyToCUDA(gpu);

    return gpu;
}

// ============================================================
// Create random batch
//
// Each sample:
// input  = tokens[start ... start + CONTEXT - 1]
// target = tokens[start + 1 ... start + CONTEXT]
// ============================================================

void CreateRandomBatch(
    const std::vector<size_t>& tokens,
    size_t begin,
    size_t end,
    std::mt19937& rng,
    Tensor& input,
    Tensor& target
) {
    if (end <= begin) {
        throw std::runtime_error(
            "Invalid token range"
        );
    }

    if (end - begin <= CONTEXT) {
        throw std::runtime_error(
            "Token range is too small"
        );
    }

    std::uniform_int_distribution<size_t> dist(
        begin,
        end - CONTEXT - 1
    );

    std::vector<size_t> input_tokens;
    std::vector<size_t> target_tokens;

    input_tokens.reserve(
        BATCH_SIZE * CONTEXT
    );

    target_tokens.reserve(
        BATCH_SIZE * CONTEXT
    );

    for (size_t b = 0;
         b < BATCH_SIZE;
         ++b) {

        size_t start =
            dist(rng);

        for (size_t i = 0;
             i < CONTEXT;
             ++i) {

            input_tokens.push_back(
                tokens[start + i]
            );

            target_tokens.push_back(
                tokens[start + i + 1]
            );
        }
    }

    input =
        CreateInput(input_tokens);

    target =
        CreateTarget(target_tokens);
}

// ============================================================
// Get scalar CUDA tensor
// ============================================================

float GetScalar(
    const Tensor& tensor
) {
    if (tensor.GetSize() != 1) {
        throw std::runtime_error(
            "Expected scalar tensor"
        );
    }

    float value = 0.0f;

    cudaError_t error =
        cudaMemcpy(
            &value,
            tensor.Data(),
            sizeof(float),
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

    return value;
}

// ============================================================
// Calculate validation loss
//
// No backward.
// No optimizer update.
// ============================================================

float EvaluateValidation(
    LanguageModel& model,
    CrossEntropyLoss& loss,
    const std::vector<size_t>& tokens,
    size_t validation_start,
    size_t validation_end,
    std::mt19937& rng
) {
    double total_loss = 0.0;

    for (size_t batch = 0;
         batch < VALIDATION_BATCHES;
         ++batch) {

        Tensor input;
        Tensor target;

        CreateRandomBatch(
            tokens,
            validation_start,
            validation_end,
            rng,
            input,
            target
        );

        auto input_ptr =
            std::make_shared<Tensor>(
                std::move(input)
            );

        auto logits =
            model.forward(
                input_ptr
            );

        Tensor current_loss =
            loss.forward(
                *logits,
                target
            );

        float loss_value =
            GetScalar(
                current_loss
            );

        total_loss +=
            static_cast<double>(
                loss_value
            );
    }

    return static_cast<float>(
        total_loss /
        static_cast<double>(
            VALIDATION_BATCHES
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
            << "LR SCHEDULE TRAINING TEST\n";
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
        // Configuration
        // ----------------------------------------------------

        std::cout
            << "Vocabulary: "
            << VOCAB_SIZE
            << "\n";

        std::cout
            << "Embedding: "
            << EMBED_DIM
            << "\n";

        std::cout
            << "Blocks: "
            << BLOCKS
            << "\n";

        std::cout
            << "Heads: "
            << HEADS
            << "\n";

        std::cout
            << "Hidden: "
            << HIDDEN
            << "\n";

        std::cout
            << "Context: "
            << CONTEXT
            << "\n";

        std::cout
            << "Batch: "
            << BATCH_SIZE
            << "\n";

        std::cout
            << "Steps: "
            << STEPS
            << "\n";

        std::cout
            << "Max LR: "
            << MAX_LR
            << "\n";

        std::cout
            << "Min LR: "
            << MIN_LR
            << "\n";

        std::cout
            << "Warmup: "
            << WARMUP_STEPS
            << "\n";

        std::cout
            << "\n";

        // ----------------------------------------------------
        // Print LR schedule
        // ----------------------------------------------------

        std::cout
            << "Learning rate schedule:\n";

        std::cout
            << "  step 0:    "
            << GetLearningRate(0)
            << "\n";

        std::cout
            << "  step 100:  "
            << GetLearningRate(100)
            << "\n";

        std::cout
            << "  step 200:  "
            << GetLearningRate(200)
            << "\n";

        std::cout
            << "  step 500:  "
            << GetLearningRate(500)
            << "\n";

        std::cout
            << "  step 1000: "
            << GetLearningRate(1000)
            << "\n";

        std::cout
            << "  step 1500: "
            << GetLearningRate(1500)
            << "\n";

        std::cout
            << "  step 1999: "
            << GetLearningRate(1999)
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
            ReadFile(
                CORPUS_PATH
            );

        std::vector<size_t> tokens =
            tokenizer.Encode(
                corpus
            );

        std::cout
            << "[OK] Corpus encoded.\n";

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n\n";

        // ----------------------------------------------------
        // Train / validation split
        // ----------------------------------------------------

        size_t train_size =
            static_cast<size_t>(
                tokens.size() * 0.9
            );

        size_t validation_start =
            train_size;

        size_t validation_end =
            tokens.size();

        std::cout
            << "Train tokens: "
            << train_size
            << "\n";

        std::cout
            << "Validation tokens: "
            << validation_end -
               validation_start
            << "\n\n";

        // ----------------------------------------------------
        // Random generators
        //
        // Separate generators are used so that validation
        // sampling does not affect training randomness.
        // ----------------------------------------------------

        std::mt19937 train_rng(
            SEED
        );

        std::mt19937 validation_rng(
            SEED
        );

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

        // ----------------------------------------------------
        // Loss
        // ----------------------------------------------------

        CrossEntropyLoss loss;

        std::cout
            << "[OK] Loss created.\n\n";

        // ----------------------------------------------------
        // Initial validation
        // ----------------------------------------------------

        std::cout
            << "Initial validation...\n";

        float initial_val_loss =
            EvaluateValidation(
                model,
                loss,
                tokens,
                validation_start,
                validation_end,
                validation_rng
            );

        std::cout
            << "Initial validation loss: "
            << initial_val_loss
            << "\n\n";

        // Reset validation RNG so that every validation
        // evaluation uses the same sequence of windows.

        validation_rng.seed(
            SEED
        );

        // ----------------------------------------------------
        // Training statistics
        // ----------------------------------------------------

        double total_train_loss = 0.0;

        size_t train_loss_count = 0;

        float best_val_loss =
            initial_val_loss;

        size_t best_step = 0;

        float final_train_loss = 0.0f;
        float final_val_loss = 0.0f;

        // ----------------------------------------------------
        // Training
        // ----------------------------------------------------

        std::cout
            << "========================================\n";
        std::cout
            << "TRAINING\n";
        std::cout
            << "========================================\n\n";

        for (size_t step = 0;
             step < STEPS;
             ++step) {

            // ------------------------------------------------
            // Learning rate
            // ------------------------------------------------

            float lr =
                GetLearningRate(
                    step
                );

            // ------------------------------------------------
            // Random batch
            // ------------------------------------------------

            Tensor input;
            Tensor target;

            CreateRandomBatch(
                tokens,
                0,
                train_size,
                train_rng,
                input,
                target
            );

            // ------------------------------------------------
            // Forward
            // ------------------------------------------------

            model.ClearGrad();

            auto input_ptr =
                std::make_shared<Tensor>(
                    std::move(input)
                );

            auto logits =
                model.forward(
                    input_ptr
                );

            Tensor current_loss =
                loss.forward(
                    *logits,
                    target
                );

            float loss_value =
                GetScalar(
                    current_loss
                );

            // ------------------------------------------------
            // Backward
            // ------------------------------------------------

            Tensor loss_grad =
                loss.backward();

            logits->backward(
                loss_grad
            );

            // ------------------------------------------------
            // AdamW
            // ------------------------------------------------

            model.UpdateAdamW(
                lr
            );

            // ------------------------------------------------
            // Statistics
            // ------------------------------------------------

            total_train_loss +=
                static_cast<double>(
                    loss_value
                );

            ++train_loss_count;

            final_train_loss =
                loss_value;

            // ------------------------------------------------
            // Validation
            // ------------------------------------------------

            if (step % VALIDATION_EVERY == 0 ||
                step + 1 == STEPS) {

                cudaDeviceSynchronize();

                validation_rng.seed(
                    SEED
                );

                float val_loss =
                    EvaluateValidation(
                        model,
                        loss,
                        tokens,
                        validation_start,
                        validation_end,
                        validation_rng
                    );

                final_val_loss =
                    val_loss;

                double average_train_loss =
                    total_train_loss /
                    static_cast<double>(
                        train_loss_count
                    );

                std::cout
                    << "Step "
                    << std::setw(4)
                    << step
                    << " | LR: "
                    << std::fixed
                    << std::setprecision(7)
                    << lr
                    << " | Train: "
                    << std::setprecision(5)
                    << loss_value
                    << " | Avg: "
                    << average_train_loss
                    << " | Val: "
                    << val_loss;

                if (val_loss < best_val_loss) {

                    best_val_loss =
                        val_loss;

                    best_step =
                        step;

                    std::cout
                        << " | BEST";
                }

                std::cout
                    << "\n";

                // Reset running training average after
                // validation so we can see the current
                // training trend more clearly.

                total_train_loss = 0.0;
                train_loss_count = 0;
            }
        }

        // ----------------------------------------------------
        // Final synchronization
        // ----------------------------------------------------

        cudaDeviceSynchronize();

        // ----------------------------------------------------
        // Save final model
        // ----------------------------------------------------

        std::cout
            << "\nSaving final model...\n";

        model.Save(
            MODEL_PATH
        );

        std::cout
            << "[OK] Model saved to: "
            << MODEL_PATH
            << "\n";

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
            << "Initial validation loss: "
            << initial_val_loss
            << "\n";

        std::cout
            << "Best validation loss:    "
            << best_val_loss
            << "\n";

        std::cout
            << "Best step:               "
            << best_step
            << "\n";

        std::cout
            << "Final train loss:        "
            << final_train_loss
            << "\n";

        std::cout
            << "Final validation loss:   "
            << final_val_loss
            << "\n";

        std::cout
            << "Validation improvement:  "
            << initial_val_loss -
               best_val_loss
            << "\n";

        if (best_val_loss <
            initial_val_loss) {

            std::cout
                << "[OK] Validation loss decreased.\n";

        } else {

            std::cout
                << "[WARNING] Validation loss did not decrease.\n";
        }

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