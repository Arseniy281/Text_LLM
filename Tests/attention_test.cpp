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
const size_t STEPS = 5000;

const float LR = 0.001f;

const std::string DATA_PATH =
    "../Data/master_and_margarita.txt";

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
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n"
            << "     RANDOM WINDOW CUDA MODEL TEST\n"
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

        tokenizer.Train(
            text,
            VOCAB_SIZE
        );

        std::vector<size_t> tokens =
            tokenizer.Encode(text);

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n";

        if (tokens.size() <= CONTEXT) {
            throw std::runtime_error(
                "Not enough tokens for training"
            );
        }

        size_t max_start =
            tokens.size() - CONTEXT - 1;

        std::cout
            << "Possible windows: "
            << max_start + 1
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

        // KV cache во время обучения не нужен.
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
            << "Steps: "
            << STEPS
            << "\n";

        std::cout
            << "Learning rate: "
            << LR
            << "\n\n";

        // ----------------------------------------------------
        // Random generator
        // ----------------------------------------------------

        std::mt19937 generator(42);

        std::uniform_int_distribution<size_t> distribution(
            0,
            max_start
        );

        // ----------------------------------------------------
        // Training
        // ----------------------------------------------------

        std::cout
            << "========================================\n"
            << "             TRAINING\n"
            << "========================================\n\n";

        float initial_loss = -1.0f;
        float last_loss = -1.0f;

        double loss_sum = 0.0;

        for (size_t step = 0;
             step < STEPS;
             ++step) {

            // ------------------------------------------------
            // Выбираем случайное окно
            // ------------------------------------------------

            // ------------------------------------------------
            // Выбираем BATCH_SIZE случайных окон
            // ------------------------------------------------

            std::vector<float> input_data(
                BATCH_SIZE * CONTEXT
            );

            std::vector<float> target_data(
                BATCH_SIZE * CONTEXT
            );

            std::vector<size_t> starts(
                BATCH_SIZE
            );

            for (size_t b = 0;
                b < BATCH_SIZE;
                ++b) {

                starts[b] =
                    distribution(generator);

                for (size_t i = 0;
                    i < CONTEXT;
                    ++i) {

                    input_data[b * CONTEXT + i] =
                        static_cast<float>(
                            tokens[starts[b] + i]
                        );

                    target_data[b * CONTEXT + i] =
                        static_cast<float>(
                            tokens[starts[b] + i + 1]
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
            // Update
            // ------------------------------------------------

            model.Update(LR);

            // ------------------------------------------------
            // Statistics
            // ------------------------------------------------

            if (step == 0) {
                initial_loss =
                    loss_value;
            }

            last_loss =
                loss_value;

            loss_sum +=
                loss_value;

            // ------------------------------------------------
            // Logging
            // ------------------------------------------------

            if (step % 100 == 0 ||
                step == STEPS - 1) {

                double average_loss =
                    loss_sum /
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
                    << average_loss
                    << "\n";
            }
        }

        // ----------------------------------------------------
        // Result
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "               RESULT\n"
            << "========================================\n";

        std::cout
            << "Initial loss: "
            << initial_loss
            << "\n";

        std::cout
            << "Final loss:   "
            << last_loss
            << "\n";

        std::cout
            << "Loss change:  "
            << last_loss - initial_loss
            << "\n";

        if (!std::isfinite(initial_loss) ||
            !std::isfinite(last_loss)) {

            std::cout
                << "\n[FAIL] Loss contains NaN or Inf\n";

            return 1;
        }

        if (last_loss >= initial_loss) {

            std::cout
                << "\n[WARNING] Loss did not decrease.\n";

        } else {

            std::cout
                << "\n[OK] Loss decreased.\n";
        }

        std::cout
            << "\n========================================\n"
            << " RANDOM WINDOW TRAINING FINISHED\n"
            << "========================================\n";


        const std::string MODEL_PATH = "../Models/MargaritaCUDA/step_5000";

        std::cout

            << "\n========================================\n"

            << "          SAVING MODEL\n"

            << "========================================\n";

        model.SaveModel(MODEL_PATH);

        std::cout

            << "[OK] Model saved to: "

            << MODEL_PATH

            << "\n";
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