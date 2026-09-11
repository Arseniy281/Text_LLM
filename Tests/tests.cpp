#include "../Engine/Tokenizer/bpe_tokenizer.h"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stdexcept>

const std::string TOKENIZER_PATH =
    "../Models/MargaritaTokenizer";

const std::string CORPUS_PATH =
    "../Data/master_and_margarita.txt";

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

void TestString(
    BPETokenizer& tokenizer,
    const std::string& text
) {
    std::cout << "----------------------------------------\n";
    std::cout << "Original:\n";
    std::cout << text << "\n\n";

    auto tokens =
        tokenizer.Encode(text);

    std::cout
        << "Tokens: "
        << tokens.size()
        << "\n";

    std::cout
        << "IDs: ";

    for (size_t id : tokens) {
        std::cout << id << " ";
    }

    std::cout << "\n\n";

    std::string decoded =
        tokenizer.Decode(tokens);

    std::cout << "Decoded:\n";
    std::cout << decoded << "\n\n";

    if (decoded == text) {
        std::cout
            << "[OK] Round-trip is exact.\n";
    } else {
        std::cout
            << "[FAIL] Round-trip differs!\n";

        std::cout
            << "Original size: "
            << text.size()
            << "\n";

        std::cout
            << "Decoded size:  "
            << decoded.size()
            << "\n";

        size_t min_size =
            std::min(
                text.size(),
                decoded.size()
            );

        for (size_t i = 0;
             i < min_size;
             ++i) {

            if (text[i] != decoded[i]) {

                std::cout
                    << "First difference at byte "
                    << i
                    << "\n";

                std::cout
                    << "Original byte: "
                    << static_cast<int>(
                        static_cast<unsigned char>(
                            text[i]
                        )
                    )
                    << "\n";

                std::cout
                    << "Decoded byte:  "
                    << static_cast<int>(
                        static_cast<unsigned char>(
                            decoded[i]
                        )
                    )
                    << "\n";

                break;
            }
        }
    }

    std::cout << "\n";
}

int main() {

    try {

        std::cout
            << "========================================\n";
        std::cout
            << "BPE ROUND-TRIP TEST\n";
        std::cout
            << "========================================\n\n";

        BPETokenizer tokenizer;

        tokenizer.Load(
            TOKENIZER_PATH
        );

        std::cout
            << "[OK] Tokenizer loaded.\n";

        std::cout
            << "Vocabulary size: "
            << tokenizer.GetVocabSize()
            << "\n\n";

        // ----------------------------------------------------
        // 1. Простые строки
        // ----------------------------------------------------

        TestString(
            tokenizer,
            "Hello world!"
        );

        TestString(
            tokenizer,
            "The Master and Margarita"
        );

        TestString(
            tokenizer,
            "Hello, world!\nThis is a test."
        );

        // ----------------------------------------------------
        // 2. Русский / UTF-8
        // ----------------------------------------------------

        TestString(
            tokenizer,
            "Привет, мир!"
        );

        TestString(
            tokenizer,
            "Мастер и Маргарита"
        );

        // ----------------------------------------------------
        // 3. Фрагмент реальной книги
        // ----------------------------------------------------

        std::string corpus =
            ReadFile(CORPUS_PATH);

        if (corpus.empty()) {
            throw std::runtime_error(
                "Corpus is empty"
            );
        }

        size_t test_size =
            std::min<size_t>(
                2000,
                corpus.size()
            );

        std::string fragment =
            corpus.substr(
                0,
                test_size
            );

        std::cout
            << "========================================\n";
        std::cout
            << "REAL CORPUS TEST\n";
        std::cout
            << "========================================\n\n";

        auto tokens =
            tokenizer.Encode(fragment);

        std::string decoded =
            tokenizer.Decode(tokens);

        std::cout
            << "Original bytes: "
            << fragment.size()
            << "\n";

        std::cout
            << "Token count:    "
            << tokens.size()
            << "\n";

        std::cout
            << "Decoded bytes:  "
            << decoded.size()
            << "\n\n";

        if (decoded == fragment) {

            std::cout
                << "[OK] Real corpus round-trip is exact.\n";

        } else {

            std::cout
                << "[FAIL] Real corpus round-trip differs!\n";

            size_t min_size =
                std::min(
                    fragment.size(),
                    decoded.size()
                );

            for (size_t i = 0;
                 i < min_size;
                 ++i) {

                if (fragment[i] != decoded[i]) {

                    std::cout
                        << "First difference at byte "
                        << i
                        << "\n";

                    std::cout
                        << "Original byte: "
                        << static_cast<int>(
                            static_cast<unsigned char>(
                                fragment[i]
                            )
                        )
                        << "\n";

                    std::cout
                        << "Decoded byte:  "
                        << static_cast<int>(
                            static_cast<unsigned char>(
                                decoded[i]
                            )
                        )
                        << "\n";

                    size_t from =
                        i > 30 ? i - 30 : 0;

                    size_t to =
                        std::min(
                            i + 30,
                            fragment.size()
                        );

                    std::cout
                        << "\nOriginal context:\n";

                    std::cout
                        << fragment.substr(
                            from,
                            to - from
                        )
                        << "\n";

                    to =
                        std::min(
                            i + 30,
                            decoded.size()
                        );

                    std::cout
                        << "\nDecoded context:\n";

                    std::cout
                        << decoded.substr(
                            from,
                            to > from
                                ? to - from
                                : 0
                        )
                        << "\n";

                    break;
                }
            }
        }

        std::cout
            << "\n========================================\n";

        return decoded == fragment ? 0 : 1;

    }
    catch (const std::exception& e) {

        std::cerr
            << "[ERROR] "
            << e.what()
            << "\n";

        return 1;
    }
}