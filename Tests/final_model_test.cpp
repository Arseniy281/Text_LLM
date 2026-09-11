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

// ============================================================
// Настройки
// ============================================================

const size_t VOCAB_SIZE = 1000;

const size_t EMBED_DIM = 128;
const size_t BLOCKS = 4;
const size_t HEADS = 4;
const size_t HIDDEN = 512;

const size_t GENERATION_LENGTH = 50;

const float TEMPERATURE = 0.8f;
const float TOP_P = 0.9f;

const std::string TOKENIZER_PATH =
    "/content/Text_LLM/Models/MargaritaTokenizer";

const std::string MODEL_PATH =
    "/content/Text_LLM/Models/MargaritaCUDA/step_20000";

const std::string PROMPT =
    "The Master and Margarita";

// ============================================================
// Print generated text
// ============================================================

void PrintText(
    const std::string& prompt,
    const std::vector<size_t>& generated,
    BPETokenizer& tokenizer
) {
    std::string generated_text =
        tokenizer.Decode(generated);

    std::cout
        << "\n========================================\n"
        << "PROMPT\n"
        << "========================================\n";

    std::cout
        << prompt
        << "\n";

    std::cout
        << "\n========================================\n"
        << "GENERATED\n"
        << "========================================\n";

    std::cout
        << generated_text
        << "\n";

    std::cout
        << "\nGenerated tokens: "
        << generated.size()
        << "\n";
}

// ============================================================
// Encode prompt
// ============================================================

std::vector<size_t> EncodePrompt(
    BPETokenizer& tokenizer,
    const std::string& prompt
) {
    std::vector<size_t> tokens =
        tokenizer.Encode(prompt);

    if (tokens.empty()) {
        throw std::runtime_error(
            "Prompt produced no tokens"
        );
    }

    for (size_t token : tokens) {
        if (token >= tokenizer.GetVocabSize()) {
            throw std::runtime_error(
                "Prompt contains invalid token id"
            );
        }
    }

    std::cout
        << "Prompt tokens: "
        << tokens.size()
        << "\n";

    return tokens;
}

// ============================================================
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n"
            << "       MARGARITA CUDA GENERATION TEST\n"
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
            << "[OK] Tokenizer found.\n";

        std::cout
            << "[OK] Model found.\n\n";

        // ----------------------------------------------------
        // Load tokenizer
        // ----------------------------------------------------

        std::cout
            << "========================================\n"
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
        // Encode prompt
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "          ENCODING PROMPT\n"
            << "========================================\n";

        std::cout
            << "Prompt:\n"
            << PROMPT
            << "\n";

        std::vector<size_t> tokens =
            EncodePrompt(
                tokenizer,
                PROMPT
            );

        // ----------------------------------------------------
        // Generation
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "            GENERATION\n"
            << "========================================\n";

        std::cout
            << "Tokens to generate: "
            << GENERATION_LENGTH
            << "\n";

        std::cout
            << "Temperature: "
            << TEMPERATURE
            << "\n";

        std::cout
            << "Top-p: "
            << TOP_P
            << "\n";

        std::vector<size_t> generated =
            model.generate(
                tokens,
                GENERATION_LENGTH,
                TEMPERATURE,
                TOP_P,
                -1
            );

        cudaDeviceSynchronize();

        // ----------------------------------------------------
        // Basic validation
        // ----------------------------------------------------

        if (generated.size() !=
            GENERATION_LENGTH) {

            throw std::runtime_error(
                "Generation returned unexpected "
                "number of tokens"
            );
        }

        for (size_t token : generated) {

            if (token >= VOCAB_SIZE) {

                throw std::runtime_error(
                    "Generated invalid token id"
                );
            }
        }

        std::cout
            << "[OK] Generated token count is correct.\n";

        std::cout
            << "[OK] All generated token ids are valid.\n";

        // ----------------------------------------------------
        // Print result
        // ----------------------------------------------------

        PrintText(
            PROMPT,
            generated,
            tokenizer
        );

        // ----------------------------------------------------
        // Result
        // ----------------------------------------------------

        std::cout
            << "\n========================================\n"
            << "       GENERATION TEST PASSED\n"
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