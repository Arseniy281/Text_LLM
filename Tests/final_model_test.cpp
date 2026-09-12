#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/LanguageModel/language_model.h"

#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

int main() {
    const std::string TOKENIZER_PATH =
        "../Models/EnglishTokenizer";

    const std::string MODEL_PATH =
        "../Models/EnglishSmaller_best";

    const int MAX_NEW_TOKENS = 150;

    try {
        // ========================================================
        // TOKENIZER
        // ========================================================

        std::cout << "Loading tokenizer...\n";

        BPETokenizer tokenizer;
        tokenizer.Load(TOKENIZER_PATH);

        std::cout << "Tokenizer loaded.\n";
        std::cout << "Vocabulary size: "
                  << tokenizer.GetVocabSize()
                  << "\n";


        // ========================================================
        // MODEL
        // ========================================================

        std::cout << "\nLoading model...\n";

        LanguageModel model(
            tokenizer.GetVocabSize(),
            128,   // EMBED_DIM
            4,     // BLOCKS
            4,     // HEADS
            512,   // HIDDEN
            Device::CUDA
        );

        model.LoadModel(MODEL_PATH);

        std::cout << "Model loaded.\n";


        // ========================================================
        // PROMPTS
        // ========================================================

        std::vector<std::string> prompts = {
            "The man walked into the room and",

            "I don't know what happened, but",

            "She looked at him and said,",

            "The police arrived at the house and",

            "It was late at night when they finally"
        };


        // ========================================================
        // GENERATION
        // ========================================================

        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "          GENERATION TEST\n";
        std::cout << "========================================\n";


        for (size_t i = 0; i < prompts.size(); ++i) {

            const std::string& prompt = prompts[i];

            std::cout << "\n";
            std::cout << "----------------------------------------\n";
            std::cout << "Prompt " << i + 1 << ":\n";
            std::cout << prompt << "\n";
            std::cout << "----------------------------------------\n";


            // ====================================================
            // ENCODE
            // ====================================================

            std::vector<size_t> prompt_tokens =
                tokenizer.Encode(prompt);

            std::cout << "Prompt tokens: "
                      << prompt_tokens.size()
                      << "\n";


            if (prompt_tokens.empty()) {
                std::cout << "[SKIP] Empty token sequence.\n";
                continue;
            }


            // ====================================================
            // GENERATE
            // ====================================================

            std::vector<size_t> generated_tokens =
                model.generate(
                    prompt_tokens,
                    MAX_NEW_TOKENS,
                    1.0f,    // temperature
                    0.9f,    // top_p
                    -1       // no EOS
                );


            // ====================================================
            // DECODE
            // ====================================================

            std::string generated_text =
                tokenizer.Decode(generated_tokens);


            std::cout << "\nGenerated tokens: "
                      << generated_tokens.size()
                      << "\n\n";

            std::cout << prompt;
            std::cout << generated_text;
            std::cout << "\n";
        }


        // ========================================================
        // DONE
        // ========================================================

        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "[OK] Generation test finished.\n";
        std::cout << "========================================\n";

    } catch (const std::exception& e) {

        std::cerr << "\n[ERROR] "
                  << e.what()
                  << "\n";

        return 1;
    }

    return 0;
}