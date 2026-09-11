#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"
#include "../Engine/Tensor/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <random>
#include <iomanip>
#include <stdexcept>
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
const size_t BATCH_SIZE = 16;

const size_t STEPS = 10000;

const float LR = 0.0003f;

const size_t VALIDATION_EVERY = 100;
const size_t VALIDATION_BATCHES = 20;

const unsigned int SEED = 42;

const std::string DATA_PATH =
    "../Data/master_and_margarita.txt";

const std::string TOKENIZER_PATH =
    "../Models/MargaritaTokenizer";

const std::string MODEL_PATH =
    "../Models/MargaritaConstantLR_10k_best";

// ============================================================
// CUDA scalar -> CPU
// ============================================================

float GetScalar(const Tensor& tensor) {

    if (tensor.GetSize() != 1) {
        throw std::runtime_error(
            "GetScalar: tensor must contain exactly one value"
        );
    }

    if (tensor.GetDevice() == Device::CPU) {
        return tensor.at(0);
    }

    float value = 0.0f;

    cudaError_t error = cudaMemcpy(
        &value,
        tensor.Data(),
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("GetScalar cudaMemcpy failed: ") +
            cudaGetErrorString(error)
        );
    }

    return value;
}

// ============================================================
// Validation
// ============================================================

float EvaluateValidation(
    LanguageModel& model,
    CrossEntropyLoss& loss,
    const std::vector<size_t>& tokens,
    size_t validation_begin,
    size_t validation_end,
    std::mt19937& generator
) {

    size_t max_start =
        validation_end - CONTEXT - 1;

    std::uniform_int_distribution<size_t> distribution(
        validation_begin,
        max_start
    );

    double loss_sum = 0.0;

    for (size_t batch = 0;
         batch < VALIDATION_BATCHES;
         ++batch) {

        std::vector<float> input_data(
            BATCH_SIZE * CONTEXT
        );

        std::vector<float> target_data(
            BATCH_SIZE * CONTEXT
        );

        for (size_t b = 0;
             b < BATCH_SIZE;
             ++b) {

            size_t start =
                distribution(generator);

            for (size_t i = 0;
                 i < CONTEXT;
                 ++i) {

                input_data[b * CONTEXT + i] =
                    static_cast<float>(
                        tokens[start + i]
                    );

                target_data[b * CONTEXT + i] =
                    static_cast<float>(
                        tokens[start + i + 1]
                    );
            }
        }

        Tensor input_cpu(
            {BATCH_SIZE, CONTEXT},
            std::move(input_data)
        );

        Tensor target_cpu(
            {BATCH_SIZE, CONTEXT},
            std::move(target_data)
        );

        Tensor input(
            {BATCH_SIZE, CONTEXT},
            Device::CUDA
        );

        Tensor target(
            {BATCH_SIZE, CONTEXT},
            Device::CUDA
        );

        input_cpu.CopyToCUDA(input);
        target_cpu.CopyToCUDA(target);

        model.ClearGrad();

        auto input_ptr =
            std::make_shared<Tensor>(
                std::move(input)
            );

        auto logits =
            model.forward(input_ptr);

        Tensor current_loss =
            loss.forward(
                *logits,
                target
            );

        float loss_value =
            GetScalar(current_loss);

        loss_sum += loss_value;
    }

    return static_cast<float>(
        loss_sum /
        static_cast<double>(VALIDATION_BATCHES)
    );
}

// ============================================================
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n"
            << " CONSTANT LR CUDA TRAINING TEST\n"
            << "========================================\n\n";

        // ----------------------------------------------------
        // CUDA
        // ----------------------------------------------------

        int device_count = 0;

        cudaGetDeviceCount(&device_count);

        std::cout
            << "CUDA devices: "
            << device_count
            << "\n";

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
            << "GPU: "
            << properties.name
            << "\n\n";

        // ----------------------------------------------------
        // Load book
        // ----------------------------------------------------

        std::cout
            << "========================================\n"
            << "          LOADING TEXT\n"
            << "========================================\n";

        std::ifstream file(DATA_PATH);

        if (!file) {
            throw std::runtime_error(
                "Cannot open: " + DATA_PATH
            );
        }

        std::string text(
            (std::istreambuf_iterator<char>(file)),
            std::istreambuf_iterator<char>()
        );

        std::cout
            << "Text size: "
            << text.size()
            << " characters\n\n";

        // ----------------------------------------------------
        // Tokenizer
        // ----------------------------------------------------

        std::cout
            << "========================================\n"
            << "           TOKENIZATION\n"
            << "========================================\n";

        BPETokenizer tokenizer;

        tokenizer.Load(TOKENIZER_PATH);

        std::vector<size_t> tokens =
            tokenizer.Encode(text);

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n";

        if (tokens.size() <= CONTEXT + 1) {
            throw std::runtime_error(
                "Not enough tokens for training"
            );
        }

        // ----------------------------------------------------
        // Train / validation split
        // ----------------------------------------------------

        size_t train_end =
            tokens.size() * 9 / 10;

        size_t validation_begin =
            train_end;

        std::cout
            << "Train tokens: "
            << train_end
            << "\n";

        std::cout
            << "Validation tokens: "
            << tokens.size() - validation_begin
            << "\n\n";

        // ----------------------------------------------------
        // Model
        // ----------------------------------------------------

        std::cout
            << "========================================\n"
            << "             MODEL\n"
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

        CrossEntropyLoss loss;

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
            << "Learning rate: "
            << LR
            << "\n\n";

        // ----------------------------------------------------
        // Random generators
        // ----------------------------------------------------

        std::mt19937 train_rng(SEED);
        std::mt19937 validation_rng(SEED);

        std::uniform_int_distribution<size_t> train_distribution(
            0,
            train_end - CONTEXT - 1
        );

        // ----------------------------------------------------
        // Initial validation
        // ----------------------------------------------------

        std::cout
            << "========================================\n"
            << "      INITIAL VALIDATION\n"
            << "========================================\n";

        cudaDeviceSynchronize();

        float initial_validation =
            EvaluateValidation(
                model,
                loss,
                tokens,
                validation_begin,
                tokens.size(),
                validation_rng
            );

        std::cout
            << "Initial validation loss: "
            << std::fixed
            << std::setprecision(6)
            << initial_validation
            << "\n\n";

        float best_validation =
            initial_validation;

        size_t best_step = 0;

        // ----------------------------------------------------
        // Training
        // ----------------------------------------------------

        std::cout
            << "========================================\n"
            << "             TRAINING\n"
            << "========================================\n\n";

        double loss_sum = 0.0;

        float initial_train_loss = -1.0f;
        float last_train_loss = -1.0f;

        for (size_t step = 0;
             step < STEPS;
             ++step) {

            // ------------------------------------------------
            // Batch
            // ------------------------------------------------

            std::vector<float> input_data(
                BATCH_SIZE * CONTEXT
            );

            std::vector<float> target_data(
                BATCH_SIZE * CONTEXT
            );

            for (size_t b = 0;
                 b < BATCH_SIZE;
                 ++b) {

                size_t start =
                    train_distribution(train_rng);

                for (size_t i = 0;
                     i < CONTEXT;
                     ++i) {

                    input_data[b * CONTEXT + i] =
                        static_cast<float>(
                            tokens[start + i]
                        );

                    target_data[b * CONTEXT + i] =
                        static_cast<float>(
                            tokens[start + i + 1]
                        );
                }
            }

            // ------------------------------------------------
            // CPU tensors
            // ------------------------------------------------

            Tensor input_cpu(
                {BATCH_SIZE, CONTEXT},
                std::move(input_data)
            );

            Tensor target_cpu(
                {BATCH_SIZE, CONTEXT},
                std::move(target_data)
            );

            // ------------------------------------------------
            // CUDA tensors
            // ------------------------------------------------

            Tensor input(
                {BATCH_SIZE, CONTEXT},
                Device::CUDA
            );

            Tensor target(
                {BATCH_SIZE, CONTEXT},
                Device::CUDA
            );

            input_cpu.CopyToCUDA(input);
            target_cpu.CopyToCUDA(target);

            // ------------------------------------------------
            // Forward
            // ------------------------------------------------

            model.ClearGrad();

            auto input_ptr =
                std::make_shared<Tensor>(
                    std::move(input)
                );

            auto logits =
                model.forward(input_ptr);

            Tensor current_loss =
                loss.forward(
                    *logits,
                    target
                );

            float loss_value =
                GetScalar(current_loss);

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
                LR,
                0.9f,
                0.999f,
                1e-8f,
                0.01f
            );

            // ------------------------------------------------
            // Statistics
            // ------------------------------------------------

            if (step == 0) {
                initial_train_loss =
                    loss_value;
            }

            last_train_loss =
                loss_value;

            loss_sum += loss_value;

            // ------------------------------------------------
            // Validation
            // ------------------------------------------------

            if (step % VALIDATION_EVERY == 0 ||
                step == STEPS - 1) {

                cudaDeviceSynchronize();

                // Одинаковые validation windows
                // на каждом шаге.
                validation_rng.seed(SEED);

                float validation_loss =
                    EvaluateValidation(
                        model,
                        loss,
                        tokens,
                        validation_begin,
                        tokens.size(),
                        validation_rng
                    );

                double average_train_loss =
                    loss_sum /
                    static_cast<double>(
                        step + 1
                    );

                std::cout
                    << "Step "
                    << std::setw(5)
                    << step
                    << " | Train: "
                    << std::fixed
                    << std::setprecision(6)
                    << loss_value
                    << " | Avg: "
                    << average_train_loss
                    << " | Val: "
                    << validation_loss;

                if (validation_loss < best_validation) {

                    best_validation =
                        validation_loss;

                    best_step =
                        step;

                    model.SaveModel(
                        MODEL_PATH
                    );

                    std::cout
                        << " | BEST";
                }

                std::cout
                    << "\n";

                loss_sum = 0.0;
            }
        }

        cudaDeviceSynchronize();

        // ----------------------------------------------------
        // Result
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "               RESULT\n"
            << "========================================\n";

        std::cout
            << "Initial train loss: "
            << initial_train_loss
            << "\n";

        std::cout
            << "Final train loss:   "
            << last_train_loss
            << "\n";

        std::cout
            << "Initial validation: "
            << initial_validation
            << "\n";

        std::cout
            << "Best validation:    "
            << best_validation
            << "\n";

        std::cout
            << "Best step:          "
            << best_step
            << "\n";

        std::cout
            << "Validation change:  "
            << best_validation -
               initial_validation
            << "\n";

        std::cout
            << "Best model saved to:\n"
            << MODEL_PATH
            << "\n";

        if (!std::isfinite(initial_train_loss) ||
            !std::isfinite(last_train_loss) ||
            !std::isfinite(best_validation)) {

            std::cout
                << "\n[FAIL] Loss contains NaN or Inf\n";

            return 1;
        }

        if (best_validation < initial_validation) {

            std::cout
                << "\n[OK] Validation loss decreased.\n";

        } else {

            std::cout
                << "\n[WARNING] Validation loss did not decrease.\n";
        }

        std::cout
            << "\n========================================\n"
            << " CONSTANT LR TRAINING FINISHED\n"
            << "========================================\n";

        return 0;
    }
    catch (const std::exception& exception) {

        std::cerr
            << "\n[ERROR] "
            << exception.what()
            << "\n";

        return 1;
    }
}