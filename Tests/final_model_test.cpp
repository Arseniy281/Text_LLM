#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem>

const size_t VOCAB_SIZE = 1000;
const size_t EMBED_DIM = 128;
const size_t BLOCKS = 4;
const size_t HEADS = 4;
const size_t HIDDEN = 512;

const std::string TOKENIZER_PATH =
    "/content/Text_LLM/Models/MargaritaTokenizer";

const std::string MODEL_PATH =
    "/content/Text_LLM/Models/MargaritaCUDA/best";


void PrintText(
    const std::string& prompt,
    const std::vector<size_t>& generated,
    BPETokenizer& tokenizer
) {
    std::string generated_text = tokenizer.Decode(generated);

    std::cout
        << "\n========================================\n"
        << "PROMPT\n"
        << "========================================\n";

    std::cout << prompt << "\n";

    std::cout
        << "\n========================================\n"
        << "GENERATED\n"
        << "========================================\n";

    std::cout << generated_text << "\n";

    std::cout
        << "\nGenerated tokens: "
        << generated.size()
        << "\n";
}


std::vector<size_t> EncodePrompt(
    BPETokenizer& tokenizer,
    const std::string& prompt
) {
    std::vector<size_t> tokens = tokenizer.Encode(prompt);

    if (tokens.empty()) {
        throw std::runtime_error(
            "Prompt produced no tokens: " + prompt
        );
    }

    std::cout
        << "Prompt tokens: "
        << tokens.size()
        << "\n";

    return tokens;
}


int main() {
    try {
        std::cout
            << "========================================\n"
            << "       MARGARITA CUDA GENERATION TEST\n"
            << "========================================\n\n";

        int device_count = 0;

        cudaError_t error =
            cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDeviceCount failed: ") +
                cudaGetErrorString(error)
            );
        }

        if (device_count == 0) {
            throw std::runtime_error(
                "No CUDA devices found"
            );
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        std::cout
            << "GPU: "
            << prop.name
            << "\n\n";


        if (!std::filesystem::exists(MODEL_PATH)) {
            throw std::runtime_error(
                "Model directory not found: " +
                MODEL_PATH
            );
        }

        if (!std::filesystem::exists(TOKENIZER_PATH)) {
            throw std::runtime_error(
                "Tokenizer file not found: " +
                TOKENIZER_PATH
            );
        }


        std::cout
            << "Loading tokenizer...\n";

        BPETokenizer tokenizer;

        tokenizer.Load(TOKENIZER_PATH);

        std::cout
            << "[OK] Tokenizer loaded.\n"
            << "Vocab size: "
            << tokenizer.GetVocabSize()
            << "\n\n";


        if (tokenizer.GetVocabSize() != VOCAB_SIZE) {
            throw std::runtime_error(
                "Tokenizer vocab size does not match model"
            );
        }


        std::cout
            << "Creating CUDA model...\n";

        LanguageModel model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        std::cout
            << "[OK] Model created.\n\n";


        std::cout
            << "Loading model:\n"
            << MODEL_PATH
            << "\n";

        model.LoadModel(MODEL_PATH);

        cudaDeviceSynchronize();

        std::cout
            << "[OK] Model loaded.\n\n";


        std::vector<std::string> prompts = {
            "The Master and Margarita",
            "Margarita",
            "Pontius Pilate",
            "The professor said"
        };


        for (size_t i = 0; i < prompts.size(); ++i) {

            std::cout
                << "\n########################################\n"
                << "TEST "
                << (i + 1)
                << " / "
                << prompts.size()
                << "\n"
                << "########################################\n";

            const std::string& prompt = prompts[i];

            std::cout
                << "\nEncoding prompt:\n"
                << prompt
                << "\n";

            std::vector<size_t> tokens =
                EncodePrompt(tokenizer, prompt);


            std::cout
                << "Generating 200 tokens...\n";

            std::vector<size_t> generated =
                model.generate(
                    tokens,
                    200,
                    1.0f,
                    1.0f,
                    -1
                );

            cudaDeviceSynchronize();

            PrintText(
                prompt,
                generated,
                tokenizer
            );
        }


        std::cout
            << "\n========================================\n"
            << "          GENERATION TEST PASSED\n"
            << "========================================\n";

    } catch (const std::exception& e) {

        std::cerr
            << "\n========================================\n"
            << "               ERROR\n"
            << "========================================\n"
            << e.what()
            << "\n";

        return 1;
    }

    return 0;
}