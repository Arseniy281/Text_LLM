#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Layers/language_model.h"
#include "chat_interface.h"

// void ChatWithModel(LanguageModel& model, BPETokenizer& tokenizer) {
//     std::string input;
//     std::cout << "\n=== Chat with Character ===\n";
//     std::cout << "Type 'exit' to quit\n\n";

//     while (true) {
//         std::cout << "You: ";
//         std::getline(std::cin, input);
//         if (input == "exit") break;

//         auto tokens = tokenizer.Encode(input);
//         auto response = model.generate(tokens, 100, 0.8f, 0.9f, -1);
//         std::vector<size_t> response_size_t(response.begin(), response.end());
//         std::string text = tokenizer.Decode(response_size_t);

//         std::cout << "AI: " << text << "\n\n";
//     }
// }

void ChatWithModel(
    LanguageModel& model,
    BPETokenizer& tokenizer) {

    std::string input;

    std::cout << "\n=== My C++ Language Model ===\n";
    std::cout << "Type 'exit' to quit\n\n";

    while (true) {

        std::cout << "You: ";

        std::getline(std::cin, input);

        if (input == "exit") {
            break;
        }

        if (input.empty()) {
            continue;
        }

        auto prompt =
            tokenizer.Encode(input);

        auto generated =
            model.generate(
                prompt,
                30,
                1.0f,
                1.0f,
                -1
            );

        std::cout << "AI: ";

        std::cout <<
            tokenizer.Decode(generated);

        std::cout << "\n\n";
    }
}