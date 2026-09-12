#include "bpe_tokenizer.h"

#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <iostream>

// ============================================================
// Вспомогательные структуры
// ============================================================

namespace {

struct Word {
    std::vector<int> tokens;
    size_t frequency = 0;
};

uint64_t MakePairKey(int a, int b) {
    return (static_cast<uint64_t>(
                static_cast<uint32_t>(a)
            ) << 32)
           |
           static_cast<uint32_t>(b);
}

}

// ============================================================
// SplitInVector
// ============================================================

std::vector<std::vector<std::string>>
BPETokenizer::SplitInVector(const std::string& str) {
    std::vector<std::vector<std::string>> tokens;

    std::vector<std::string> current_word;

    for (char c : str) {
        if (c == ' ') {
            if (!current_word.empty()) {
                current_word.push_back(" ");
                tokens.push_back(current_word);
                current_word.clear();
            } else {
                tokens.push_back({" "});
            }
        }
        else if (c == '\n') {
            if (!current_word.empty()) {
                tokens.push_back(current_word);
                current_word.clear();
            }

            tokens.push_back({"\n"});
        }
        else {
            current_word.push_back(
                std::string(1, c)
            );
        }
    }

    if (!current_word.empty()) {
        tokens.push_back(current_word);
    }

    return tokens;
}

// ============================================================
// Split
// ============================================================

std::vector<std::string>
BPETokenizer::Split(const std::string& str) {
    std::vector<std::string> tokens;

    std::istringstream iss(str);
    std::string token;

    while (iss >> token) {
        tokens.push_back(token);
    }

    return tokens;
}

// ============================================================
// Constructor
// ============================================================

BPETokenizer::BPETokenizer() {
    for (int i = 0; i < 256; ++i) {
        AddToken(
            std::string(
                1,
                static_cast<char>(i)
            )
        );
    }
}

// ============================================================
// AddToken
// ============================================================

void BPETokenizer::AddToken(
    const std::string& token
) {
    token_to_id_[token] = vocab_.size();
    vocab_.push_back(token);

    trie.Insert(token);
}

// ============================================================
// GetBestPair
// ============================================================

std::string BPETokenizer::GetBestPair(
    const std::unordered_map<std::string, int>& pair_counts
) {
    std::string best_pair;
    int best_freq = 0;

    for (const auto& [pair, freq] : pair_counts) {
        if (freq > best_freq) {
            best_freq = freq;
            best_pair = pair;
        }
        else if (freq == best_freq &&
                 !pair.empty() &&
                 (best_pair.empty() || pair < best_pair)) {
            best_pair = pair;
        }
    }

    return best_pair;
}

// ============================================================
// UpdateWords
// ============================================================

void BPETokenizer::UpdateWords(
    std::vector<std::vector<std::string>>& words,
    const std::string& best_pair
) {
    for (auto& word : words) {
        size_t i = 0;

        while (i + 1 < word.size()) {
            if (word[i] + word[i + 1] == best_pair) {
                word[i] = best_pair;

                word.erase(
                    word.begin() + i + 1
                );
            }
            else {
                ++i;
            }
        }
    }
}

// ============================================================
// Train
// ============================================================
//
// Главное отличие:
//
// Старый вариант:
//
//   68 млн символов
//          ↓
//   vector<vector<string>>
//          ↓
//   744 раза пройти практически весь корпус
//
// Новый вариант:
//
//   корпус
//      ↓
//   одинаковые последовательности объединяются
//      ↓
//   храним последовательность + frequency
//      ↓
//   merge работает по уникальным последовательностям
//
// ============================================================

void BPETokenizer::Train(
    const std::string& corpus,
    size_t vocab_size
) {
    if (vocab_size < 256) {
        throw std::runtime_error(
            "BPETokenizer::Train: vocab_size must be >= 256"
        );
    }

    if (corpus.empty()) {
        throw std::runtime_error(
            "BPETokenizer::Train: corpus is empty"
        );
    }

    // --------------------------------------------------------
    // Если tokenizer уже содержит больше токенов,
    // обучение продолжаем с текущего состояния.
    // --------------------------------------------------------

    if (vocab_.size() >= vocab_size) {
        return;
    }

    // --------------------------------------------------------
    // 1. Сначала собираем одинаковые последовательности.
    //
    // Например:
    //
    // "hello "
    // "hello "
    // "hello "
    //
    // хранится как:
    //
    // "hello " -> 3
    //
    // --------------------------------------------------------

    std::unordered_map<std::string, size_t> word_counts;

    std::string current;

    current.reserve(64);

    auto flush_current = [&]() {
        if (!current.empty()) {
            ++word_counts[current];
            current.clear();
        }
    };

    for (char c : corpus) {
        if (c == '\n') {
            flush_current();

            ++word_counts["\n"];
        }
        else if (c == ' ') {
            if (!current.empty()) {
                current.push_back(' ');

                ++word_counts[current];

                current.clear();
            }
            else {
                ++word_counts[" "];
            }
        }
        else {
            current.push_back(c);
        }
    }

    flush_current();

    std::cout
        << "Unique sequences: "
        << word_counts.size()
        << "\n";

    // --------------------------------------------------------
    // 2. Переводим строки в ID токенов.
    //
    // Вместо:
    //
    // vector<string>
    //
    // получаем:
    //
    // vector<int>
    //
    // --------------------------------------------------------

    std::vector<Word> words;

    words.reserve(word_counts.size());

    for (const auto& [text, frequency] : word_counts) {
        Word word;

        word.frequency = frequency;

        word.tokens.reserve(text.size());

        for (unsigned char c : text) {
            word.tokens.push_back(
                static_cast<int>(c)
            );
        }

        words.push_back(
            std::move(word)
        );
    }

    // word_counts больше не нужен.
    word_counts.clear();
    word_counts.rehash(0);

    // --------------------------------------------------------
    // 3. BPE merge
    // --------------------------------------------------------

    size_t merge_number = 0;

    while (vocab_.size() < vocab_size) {
        // ----------------------------------------------------
        // pair -> frequency
        // ----------------------------------------------------

        std::unordered_map<
            uint64_t,
            size_t
        > pair_counts;

        // ----------------------------------------------------
        // Считаем пары.
        //
        // Важно:
        // frequency слова учитывается сразу.
        // ----------------------------------------------------

        for (const Word& word : words) {
            if (word.tokens.size() < 2) {
                continue;
            }

            for (size_t i = 0;
                 i + 1 < word.tokens.size();
                 ++i) {

                uint64_t key =
                    MakePairKey(
                        word.tokens[i],
                        word.tokens[i + 1]
                    );

                pair_counts[key] +=
                    word.frequency;
            }
        }

        if (pair_counts.empty()) {
            break;
        }

        // ----------------------------------------------------
        // Ищем самую частую пару.
        //
        // При одинаковой частоте выбираем меньший key.
        // Это делает обучение детерминированным.
        // ----------------------------------------------------

        uint64_t best_key = 0;
        size_t best_frequency = 0;

        bool found = false;

        for (const auto& [key, frequency]
             : pair_counts) {

            if (!found ||
                frequency > best_frequency ||
                (frequency == best_frequency &&
                 key < best_key)) {

                best_key = key;
                best_frequency = frequency;
                found = true;
            }
        }

        if (!found) {
            break;
        }

        int left =
            static_cast<int>(
                best_key >> 32
            );

        int right =
            static_cast<int>(
                best_key & 0xffffffffULL
            );

        // ----------------------------------------------------
        // Получаем строковое представление пары.
        // ----------------------------------------------------

        if (left < 0 ||
            right < 0 ||
            static_cast<size_t>(left) >= vocab_.size() ||
            static_cast<size_t>(right) >= vocab_.size()) {

            throw std::runtime_error(
                "BPETokenizer::Train: invalid pair"
            );
        }

        std::string merged =
            vocab_[left] + vocab_[right];

        // ----------------------------------------------------
        // Проверяем, что такого токена ещё нет.
        // ----------------------------------------------------

        auto existing =
            token_to_id_.find(merged);

        if (existing != token_to_id_.end()) {
            // Такое практически не должно происходить,
            // но защищаемся от повторного токена.
            break;
        }

        int new_id =
            static_cast<int>(vocab_.size());

        AddToken(merged);

        // ----------------------------------------------------
        // Заменяем pair на новый token ID.
        // ----------------------------------------------------

        for (Word& word : words) {
            if (word.tokens.size() < 2) {
                continue;
            }

            size_t i = 0;

            while (i + 1 < word.tokens.size()) {
                if (word.tokens[i] == left &&
                    word.tokens[i + 1] == right) {

                    word.tokens[i] = new_id;

                    word.tokens.erase(
                        word.tokens.begin() + i + 1
                    );
                }
                else {
                    ++i;
                }
            }
        }

        ++merge_number;

        // ----------------------------------------------------
        // Прогресс
        // ----------------------------------------------------

        if (merge_number % 25 == 0 ||
            vocab_.size() == vocab_size) {

            std::cout
                << "BPE merge "
                << merge_number
                << " | vocab: "
                << vocab_.size()
                << " | pair frequency: "
                << best_frequency
                << "\n";
        }
    }

    std::cout
        << "BPE training finished.\n";

    std::cout
        << "Final vocabulary: "
        << vocab_.size()
        << "\n";
}

// ============================================================
// Save
// ============================================================

void BPETokenizer::Save(
    const std::string& filename
) {
    std::ofstream file(
        filename,
        std::ios::binary
    );

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open file for writing: "
            + filename
        );
    }

    uint64_t vocab_size =
        vocab_.size();

    file.write(
        reinterpret_cast<const char*>(
            &vocab_size
        ),
        sizeof(vocab_size)
    );

    for (const auto& token : vocab_) {
        uint64_t length =
            token.size();

        file.write(
            reinterpret_cast<const char*>(
                &length
            ),
            sizeof(length)
        );

        if (length > 0) {
            file.write(
                token.data(),
                static_cast<std::streamsize>(
                    token.size()
                )
            );
        }
    }

    if (!file) {
        throw std::runtime_error(
            "Error while writing tokenizer: "
            + filename
        );
    }

    file.close();
}

// ============================================================
// Load
// ============================================================

void BPETokenizer::Load(
    const std::string& filename
) {
    std::ifstream file(
        filename,
        std::ios::binary
    );

    if (!file.is_open()) {
        throw std::runtime_error(
            "Cannot open file for reading: "
            + filename
        );
    }

    vocab_.clear();
    token_to_id_.clear();
    trie.Clear();

    uint64_t vocab_size = 0;

    file.read(
        reinterpret_cast<char*>(
            &vocab_size
        ),
        sizeof(vocab_size)
    );

    if (!file) {
        throw std::runtime_error(
            "Invalid tokenizer file: "
            + filename
        );
    }

    for (uint64_t i = 0;
         i < vocab_size;
         ++i) {

        uint64_t length = 0;

        file.read(
            reinterpret_cast<char*>(
                &length
            ),
            sizeof(length)
        );

        if (!file) {
            throw std::runtime_error(
                "Invalid tokenizer file: "
                + filename
            );
        }

        std::string token(
            length,
            '\0'
        );

        if (length > 0) {
            file.read(
                token.data(),
                static_cast<std::streamsize>(
                    length
                )
            );
        }

        if (!file) {
            throw std::runtime_error(
                "Invalid tokenizer file: "
                + filename
            );
        }

        AddToken(token);
    }

    file.close();
}

// ============================================================
// Encode
// ============================================================

std::vector<size_t>
BPETokenizer::Encode(
    const std::string& str
) {
    std::vector<size_t> ids;

    ids.reserve(str.size());

    size_t i = 0;

    while (i < str.size()) {
        std::string prefix =
            trie.LongestToken(str, i);

        if (!prefix.empty()) {
            auto it =
                token_to_id_.find(prefix);

            if (it == token_to_id_.end()) {
                throw std::runtime_error(
                    "Tokenizer trie contains unknown token"
                );
            }

            ids.push_back(it->second);

            i += prefix.size();
        }
        else {
            // При наличии всех 256 byte-токенов
            // сюда попадать не должно.
            auto it =
                token_to_id_.find(
                    std::string(
                        1,
                        str[i]
                    )
                );

            if (it == token_to_id_.end()) {
                throw std::runtime_error(
                    "Cannot encode character"
                );
            }

            ids.push_back(it->second);

            ++i;
        }
    }

    return ids;
}

// ============================================================
// Decode
// ============================================================

std::string
BPETokenizer::Decode(
    std::vector<size_t> ids
) {
    std::string result;

    for (size_t id : ids) {
        if (id >= vocab_.size()) {
            throw std::runtime_error(
                "BPETokenizer::Decode: invalid token id"
            );
        }

        result += vocab_[id];
    }

    return result;
}

// ============================================================
// GetVocabSize
// ============================================================

size_t BPETokenizer::GetVocabSize() const {
    return vocab_.size();
}

// ============================================================
// GetTokenId
// ============================================================

size_t BPETokenizer::GetTokenId(
    const std::string& token
) const {
    return token_to_id_.at(token);
}