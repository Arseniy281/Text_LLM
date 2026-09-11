#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Layers/ce_loss.h"
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
#include <numeric>

// ============================================================
// Настройки
// ============================================================

const size_t VOCAB_SIZE = 1000;

const size_t EMBED_DIM = 128;
const size_t BLOCKS = 4;
const size_t HEADS = 4;
const size_t HIDDEN = 512;

const size_t CONTEXT = 128;

const size_t VALIDATION_WINDOWS = 200;

const std::string DATA_PATH =
    "/content/Text_LLM/Data/master_and_margarita.txt";

const std::string TOKENIZER_PATH =
    "/content/Text_LLM/Models/MargaritaTokenizer";

const std::string MODEL_PATH =
    "/content/Text_LLM/Models/MargaritaCUDA/step_20000";

// ============================================================
// Read corpus
// ============================================================

std::string ReadFile(
    const std::string& path
) {
    std::ifstream file(path);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open corpus: " + path
        );
    }

    std::string text(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    if (text.empty()) {
        throw std::runtime_error(
            "Corpus is empty"
        );
    }

    return text;
}

// ============================================================
// Get scalar from tensor
// ============================================================

float GetScalar(
    const Tensor& tensor
) {
    if (tensor.GetSize() != 1) {
        throw std::runtime_error(
            "Expected scalar tensor"
        );
    }

    if (tensor.GetDevice() == Device::CPU) {
        return tensor.Data()[0];
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
// Create CUDA input tensor
// ============================================================

Tensor CreateInput(
    const std::vector<size_t>& tokens,
    size_t start
) {
    Tensor cpu_input(
        {1, CONTEXT},
        0.0f,
        Device::CPU
    );

    for (size_t i = 0; i < CONTEXT; ++i) {

        cpu_input.Data()[i] =
            static_cast<float>(
                tokens[start + i]
            );
    }

    Tensor cuda_input(
        {1, CONTEXT},
        0.0f,
        Device::CUDA
    );

    cpu_input.CopyToCUDA(
        cuda_input
    );

    return cuda_input;
}

// ============================================================
// Create CUDA target tensor
// ============================================================

Tensor CreateTarget(
    const std::vector<size_t>& tokens,
    size_t start
) {
    Tensor cpu_target(
        {1, CONTEXT},
        0.0f,
        Device::CPU
    );

    for (size_t i = 0; i < CONTEXT; ++i) {

        cpu_target.Data()[i] =
            static_cast<float>(
                tokens[start + i + 1]
            );
    }

    Tensor cuda_target(
        {1, CONTEXT},
        0.0f,
        Device::CUDA
    );

    cpu_target.CopyToCUDA(
        cuda_target
    );

    return cuda_target;
}

// ============================================================
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n"
            << "       MARGARITA CUDA VALIDATION TEST\n"
            << "========================================\n\n";

        // ----------------------------------------------------
        // CUDA
        // ----------------------------------------------------

        int device_count = 0;

        cudaError_t error =
            cudaGetDeviceCount(
                &device_count
            );

        if (error != cudaSuccess) {

            throw std::runtime_error(
                std::string(
                    "cudaGetDeviceCount failed: "
                ) +
                cudaGetErrorString(error)
            );
        }

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
        // Check files
        // ----------------------------------------------------

        std::cout
            << "========================================\n"
            << "          CHECKING FILES\n"
            << "========================================\n";

        if (!std::filesystem::exists(
                DATA_PATH)) {

            throw std::runtime_error(
                "Corpus not found: " +
                DATA_PATH
            );
        }

        if (!std::filesystem::exists(
                TOKENIZER_PATH)) {

            throw std::runtime_error(
                "Tokenizer not found: " +
                TOKENIZER_PATH
            );
        }

        if (!std::filesystem::exists(
                MODEL_PATH)) {

            throw std::runtime_error(
                "Model not found: " +
                MODEL_PATH
            );
        }

        std::cout
            << "[OK] Corpus found.\n";

        std::cout
            << "[OK] Tokenizer found.\n";

        std::cout
            << "[OK] Model found.\n\n";

        // ----------------------------------------------------
        // Load corpus
        // ----------------------------------------------------

        std::cout
            << "========================================\n"
            << "            LOADING CORPUS\n"
            << "========================================\n";

        std::string corpus =
            ReadFile(
                DATA_PATH
            );

        std::cout
            << "Corpus characters: "
            << corpus.size()
            << "\n";

        // ----------------------------------------------------
        // Load tokenizer
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "        LOADING TOKENIZER\n"
            << "========================================\n";

        BPETokenizer tokenizer;

        tokenizer.Load(
            TOKENIZER_PATH
        );

        std::cout
            << "[OK] Tokenizer loaded.\n";

        std::cout
            << "Vocabulary size: "
            << tokenizer.GetVocabSize()
            << "\n";

        if (tokenizer.GetVocabSize() !=
            VOCAB_SIZE) {

            throw std::runtime_error(
                "Tokenizer vocabulary does not "
                "match model vocabulary"
            );
        }

        // ----------------------------------------------------
        // Encode corpus
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "           ENCODING CORPUS\n"
            << "========================================\n";

        std::vector<size_t> tokens =
            tokenizer.Encode(
                corpus
            );

        if (tokens.empty()) {

            throw std::runtime_error(
                "Corpus produced no tokens"
            );
        }

        std::cout
            << "[OK] Corpus encoded.\n";

        std::cout
            << "Total tokens: "
            << tokens.size()
            << "\n";

        // ----------------------------------------------------
        // Train / validation split
        // ----------------------------------------------------

        size_t train_size =
            tokens.size() * 9 / 10;

        size_t validation_start =
            train_size;

        size_t validation_tokens =
            tokens.size() - validation_start;

        std::cout
            << "\n========================================\n"
            << "          DATASET SPLIT\n"
            << "========================================\n";

        std::cout
            << "Train tokens: "
            << train_size
            << "\n";

        std::cout
            << "Validation tokens: "
            << validation_tokens
            << "\n";

        if (validation_tokens <=
            CONTEXT + 1) {

            throw std::runtime_error(
                "Validation set is too small"
            );
        }

        // ----------------------------------------------------
        // Create model
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "          CREATING MODEL\n"
            << "========================================\n";

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
        // Load model
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "           LOADING MODEL\n"
            << "========================================\n";

        std::cout
            << "Path:\n"
            << MODEL_PATH
            << "\n";

        model.LoadModel(
            MODEL_PATH
        );

        cudaDeviceSynchronize();

        std::cout
            << "[OK] Model loaded.\n";

        // ----------------------------------------------------
        // Validation
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "             VALIDATION\n"
            << "========================================\n";

        std::cout
            << "Validation windows: "
            << VALIDATION_WINDOWS
            << "\n";

        std::cout
            << "Context: "
            << CONTEXT
            << "\n";

        std::cout
            << "\n";

        CrossEntropyLoss loss;

        std::mt19937 generator(42);

        size_t max_start =
            tokens.size() - CONTEXT - 1;

        std::uniform_int_distribution<size_t>
            distribution(
                validation_start,
                max_start
            );

        double total_loss = 0.0;

        size_t successful_windows = 0;

        for (size_t step = 0;
             step < VALIDATION_WINDOWS;
             ++step) {

            size_t start =
                distribution(
                    generator
                );

            Tensor input =
                CreateInput(
                    tokens,
                    start
                );

            Tensor target =
                CreateTarget(
                    tokens,
                    start
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

            cudaDeviceSynchronize();

            float loss_value =
                GetScalar(
                    current_loss
                );

            if (!std::isfinite(
                    loss_value)) {

                throw std::runtime_error(
                    "Validation produced "
                    "non-finite loss"
                );
            }

            total_loss +=
                static_cast<double>(
                    loss_value
                );

            ++successful_windows;

            if (
                step == 0 ||
                (step + 1) % 25 == 0 ||
                step + 1 == VALIDATION_WINDOWS
            ) {

                double average =
                    total_loss /
                    static_cast<double>(
                        successful_windows
                    );

                std::cout
                    << "Window "
                    << (step + 1)
                    << " / "
                    << VALIDATION_WINDOWS
                    << " | Loss: "
                    << loss_value
                    << " | Avg: "
                    << average
                    << "\n";
            }
        }

        cudaDeviceSynchronize();

        // ----------------------------------------------------
        // Final result
        // ----------------------------------------------------

        double validation_loss =
            total_loss /
            static_cast<double>(
                successful_windows
            );

        std::cout
            << "\n========================================\n"
            << "         VALIDATION RESULT\n"
            << "========================================\n";

        std::cout
            << "Validation windows: "
            << successful_windows
            << "\n";

        std::cout
            << "Validation loss: "
            << validation_loss
            << "\n";

        if (!std::isfinite(
                validation_loss)) {

            throw std::runtime_error(
                "Validation loss is not finite"
            );
        }

        std::cout
            << "\n[OK] Validation loss is finite.\n";

        std::cout
            << "[OK] Validation completed.\n";

        std::cout
            << "\n========================================\n"
            << "       VALIDATION TEST PASSED\n"
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