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

const size_t BATCH_SIZE = 8;
const size_t STEPS = 2000;

const float LR = 0.001f;

const size_t VALIDATION_INTERVAL = 500;
const size_t VALIDATION_BATCHES = 20;

const float TRAIN_RATIO = 0.9f;

const std::string DATA_PATH =
    "../Data/master_and_margarita.txt";

const std::string TOKENIZER_PATH =
    "../Models/MargaritaTokenizer";

// ============================================================
// Получить scalar из Tensor
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
// Создать CUDA tensor из vector<float>
// ============================================================

Tensor MakeCUDATensor(
    const std::vector<float>& data,
    size_t first_dim,
    size_t second_dim
) {
    Tensor cpu(
        {first_dim, second_dim},
        std::vector<float>(data)
    );

    Tensor cuda_tensor(
        {first_dim, second_dim},
        Device::CUDA
    );

    cpu.CopyToCUDA(cuda_tensor);

    return cuda_tensor;
}

// ============================================================
// Посчитать validation loss
// ============================================================

float CalculateValidationLoss(
    LanguageModel& model,
    CrossEntropyLoss& loss,
    const std::vector<size_t>& tokens,
    size_t validation_start,
    size_t validation_end,
    std::mt19937& generator
) {
    if (validation_end <= validation_start + CONTEXT) {
        throw std::runtime_error(
            "Validation dataset is too small"
        );
    }

    size_t max_start =
        validation_end - CONTEXT - 1;

    std::uniform_int_distribution<size_t> distribution(
        validation_start,
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

            size_t start = distribution(generator);

            for (size_t i = 0;
                 i < CONTEXT;
                 ++i) {

                input_data[
                    b * CONTEXT + i
                ] =
                    static_cast<float>(
                        tokens[start + i]
                    );

                target_data[
                    b * CONTEXT + i
                ] =
                    static_cast<float>(
                        tokens[start + i + 1]
                    );
            }
        }

        Tensor input =
            MakeCUDATensor(
                input_data,
                BATCH_SIZE,
                CONTEXT
            );

        Tensor target =
            MakeCUDATensor(
                target_data,
                BATCH_SIZE,
                CONTEXT
            );

        auto input_ptr =
            std::make_shared<Tensor>(
                std::move(input)
            );

        // ----------------------------------------------------
        // ВАЖНО:
        // validation не должен строить/накапливать градиенты.
        // Пока используем тот же forward, но после него
        // градиенты очищаем и ничего не обновляем.
        // ----------------------------------------------------

        model.ClearGrad();

        auto logits =
            model.forward(input_ptr);

        Tensor current_loss =
            loss.forward(*logits, target);

        float value =
            GetScalar(current_loss);

        loss_sum += value;

        model.ClearGrad();
    }

    return static_cast<float>(
        loss_sum /
        static_cast<double>(VALIDATION_BATCHES)
    );
}

// ============================================================
// MAIN
// ============================================================

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << " RANDOM WINDOW CUDA TRAIN/VALIDATION TEST\n";
        std::cout << "========================================\n\n";

        // ====================================================
        // CUDA
        // ====================================================

        int device_count = 0;

        cudaError_t cuda_error =
            cudaGetDeviceCount(&device_count);

        if (cuda_error != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "cudaGetDeviceCount failed: "
                ) +
                cudaGetErrorString(cuda_error)
            );
        }

        if (device_count == 0) {
            throw std::runtime_error(
                "No CUDA devices found"
            );
        }

        cudaDeviceProp properties;

        cudaGetDeviceProperties(
            &properties,
            0
        );

        std::cout
            << "CUDA devices: "
            << device_count
            << '\n';

        std::cout
            << "GPU: "
            << properties.name
            << "\n\n";

        // ====================================================
        // LOAD TEXT
        // ====================================================

        std::cout
            << "========================================\n"
            << "          LOADING TEXT\n"
            << "========================================\n";

        std::ifstream file(DATA_PATH);

        if (!file) {
            throw std::runtime_error(
                "Failed to open text file: " +
                DATA_PATH
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

        // ====================================================
        // TOKENIZER
        // ====================================================

        std::cout
            << "========================================\n"
            << "           TOKENIZATION\n"
            << "========================================\n";

        BPETokenizer tokenizer;

        tokenizer.Load(TOKENIZER_PATH);

        std::cout
            << "Tokenizer loaded.\n";

        std::cout
            << "Vocab size: "
            << tokenizer.GetVocabSize()
            << '\n';

        if (tokenizer.GetVocabSize() != VOCAB_SIZE) {
            throw std::runtime_error(
                "Tokenizer vocabulary size does not match model"
            );
        }

        std::vector<size_t> tokens =
            tokenizer.Encode(text);

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n\n";

        // ====================================================
        // TRAIN / VALIDATION SPLIT
        // ====================================================

        const size_t train_tokens =
            static_cast<size_t>(
                static_cast<double>(tokens.size())
                * TRAIN_RATIO
            );

        const size_t validation_start =
            train_tokens;

        const size_t validation_end =
            tokens.size();

        const size_t train_max_start =
            train_tokens - CONTEXT - 1;

        if (train_tokens <= CONTEXT + 1) {
            throw std::runtime_error(
                "Training dataset is too small"
            );
        }

        if (validation_end <=
            validation_start + CONTEXT + 1) {

            throw std::runtime_error(
                "Validation dataset is too small"
            );
        }

        std::cout
            << "Dataset split:\n"
            << "  Train tokens: "
            << train_tokens
            << '\n'
            << "  Validation tokens: "
            << validation_end - validation_start
            << '\n'
            << "  Train ratio: "
            << TRAIN_RATIO
            << "\n\n";

        // ====================================================
        // MODEL
        // ====================================================

        std::cout
            << "========================================\n"
            << "               MODEL\n"
            << "========================================\n";

        std::cout
            << "Vocabulary: "
            << VOCAB_SIZE
            << '\n';

        std::cout
            << "Embedding: "
            << EMBED_DIM
            << '\n';

        std::cout
            << "Blocks: "
            << BLOCKS
            << '\n';

        std::cout
            << "Heads: "
            << HEADS
            << '\n';

        std::cout
            << "Hidden: "
            << HIDDEN
            << '\n';

        std::cout
            << "Context: "
            << CONTEXT
            << '\n';

        std::cout
            << "Batch: "
            << BATCH_SIZE
            << '\n';

        std::cout
            << "Steps: "
            << STEPS
            << '\n';

        std::cout
            << "Learning rate: "
            << LR
            << "\n\n";

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

        // ====================================================
        // RANDOM GENERATORS
        // ====================================================

        std::mt19937 generator(42);

        std::uniform_int_distribution<size_t>
            train_distribution(
                0,
                train_max_start
            );

        std::mt19937 validation_generator(12345);

        // ====================================================
        // TRAINING
        // ====================================================

        std::cout
            << "========================================\n"
            << "             TRAINING\n"
            << "========================================\n\n";

        double train_loss_sum = 0.0;

        float initial_train_loss = -1.0f;
        float final_train_loss = -1.0f;

        for (size_t step = 0;
             step < STEPS;
             ++step) {

            // ------------------------------------------------
            // TRAINING BATCH
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
                    train_distribution(generator);

                for (size_t i = 0;
                     i < CONTEXT;
                     ++i) {

                    input_data[
                        b * CONTEXT + i
                    ] =
                        static_cast<float>(
                            tokens[start + i]
                        );

                    target_data[
                        b * CONTEXT + i
                    ] =
                        static_cast<float>(
                            tokens[start + i + 1]
                        );
                }
            }

            Tensor input =
                MakeCUDATensor(
                    input_data,
                    BATCH_SIZE,
                    CONTEXT
                );

            Tensor target =
                MakeCUDATensor(
                    target_data,
                    BATCH_SIZE,
                    CONTEXT
                );

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

            Tensor loss_grad =
                loss.backward();

            logits->backward(
                loss_grad
            );

            model.UpdateAdamW(
                LR,
                0.9f,
                0.999f,
                1e-8f,
                0.01f
            );

            model.ClearGrad();

            // ------------------------------------------------
            // STATISTICS
            // ------------------------------------------------

            if (step == 0) {
                initial_train_loss =
                    loss_value;
            }

            final_train_loss =
                loss_value;

            train_loss_sum += loss_value;

            // ------------------------------------------------
            // LOGGING
            // ------------------------------------------------

            if (step % 100 == 0 ||
                step == STEPS - 1) {

                double average_train_loss =
                    train_loss_sum /
                    static_cast<double>(
                        step + 1
                    );

                std::cout
                    << "Step "
                    << std::setw(4)
                    << step
                    << " | Loss: "
                    << std::fixed
                    << std::setprecision(6)
                    << loss_value
                    << " | Avg: "
                    << average_train_loss
                    << '\n';
            }

            // ------------------------------------------------
            // VALIDATION
            // ------------------------------------------------

            if (step == 0 ||
                (step + 1) %
                    VALIDATION_INTERVAL == 0 ||
                step == STEPS - 1) {

                float validation_loss =
                    CalculateValidationLoss(
                        model,
                        loss,
                        tokens,
                        validation_start,
                        validation_end,
                        validation_generator
                    );

                std::cout
                    << "           Validation loss: "
                    << std::fixed
                    << std::setprecision(6)
                    << validation_loss
                    << "\n";
            }
        }

        // ====================================================
        // RESULT
        // ====================================================

        double average_train_loss =
            train_loss_sum /
            static_cast<double>(STEPS);

        std::cout
            << "\n========================================\n"
            << "               RESULT\n"
            << "========================================\n";

        std::cout
            << "Initial train loss: "
            << initial_train_loss
            << '\n';

        std::cout
            << "Final train loss:   "
            << final_train_loss
            << '\n';

        std::cout
            << "Average train loss: "
            << average_train_loss
            << '\n';

        std::cout
            << "Loss change:        "
            << final_train_loss
               - initial_train_loss
            << '\n';

        if (final_train_loss <
            initial_train_loss) {

            std::cout
                << "\n[OK] Training loss decreased.\n";

        } else {

            std::cout
                << "\n[WARNING] Training loss did not decrease.\n";
        }

        std::cout
            << "\n========================================\n"
            << " TRAIN/VALIDATION TEST FINISHED\n"
            << "========================================\n";

        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "\n[ERROR] "
            << e.what()
            << '\n';

        return 1;
    }
}