#include "bpe_tokenizer.h"
#include <vector>
#include <string>
#include <sstream>
#include <unordered_map>
#include <cstdint>

std::vector<std::vector<std::string>> BPETokenizer::SplitInVector(const std::string& str) {
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
        } else if (c == '\n') {
            if (!current_word.empty()) {
                tokens.push_back(current_word);
                current_word.clear();
            }
            tokens.push_back({"\n"});
        } else {
            current_word.push_back(std::string(1, c));
        }
    }

    if (!current_word.empty()) {
        tokens.push_back(current_word);
    }
    return tokens;
}

std::vector<std::string> BPETokenizer::Split(const std::string& str) {
    std::vector<std::string> tokens;
    std::istringstream iss(str);
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

BPETokenizer::BPETokenizer() {
    for (int i = 0; i < 256; ++i) {
        AddToken(std::string(1, static_cast<char>(i)));
    }
}

void BPETokenizer::AddToken(const std::string& token) {
    token_to_id_[token] = vocab_.size();
    vocab_.push_back(token);
    trie.Insert(token);
}

std::string BPETokenizer::GetBestPair(const std::unordered_map<std::string, int>& pair_counts) {
    std::string best_pair = "";
    int best_freq = 0;
    for (const auto& [pair, freq] : pair_counts) {
        if (freq > best_freq) {
            best_freq = freq;
            best_pair = pair;
        }
    }
    return best_pair;
}

void BPETokenizer::UpdateWords(std::vector<std::vector<std::string>>& words, const std::string& best_pair) {
    for (auto& word : words) {
        size_t i = 0;
        while (i + 1 < word.size()) {
            if (word[i] + word[i + 1] == best_pair) {
                word[i] = best_pair;
                word.erase(word.begin() + i + 1);
            } else {
                i++;
            }
        }
    }
}

void BPETokenizer::Train(const std::string& corpus, size_t vocab_size) {
    std::vector<std::vector<std::string>> words;
    std::vector<std::string> current;

    for (char c : corpus) {
        if (c == '\n') {
            if (!current.empty()) {
                words.push_back(current);
                current.clear();
            }

            words.push_back({"\n"});
        } else if (c == ' ') {
            if (!current.empty()) {
                current.push_back(" ");
                words.push_back(current);
                current.clear();
            } else {
                words.push_back({" "});
            }
        } else {
            current.push_back(std::string(1, c));
        }
    }

    if (!current.empty()) {
        words.push_back(current);
    }

    while (vocab_.size() < vocab_size) {
        std::unordered_map<std::string, int> pair_counts;
        for (const auto& word : words) {
            if (word.size() < 2) { continue; }
            for (size_t i = 0; i + 1 < word.size(); i++) {
                std::string pair = word[i] + word[i + 1];
                pair_counts[pair]++;
            }
        }

        std::string best_pair = GetBestPair(pair_counts);
        if (best_pair.empty()) { break; }
        AddToken(best_pair);
        UpdateWords(words, best_pair);
    }
}

void BPETokenizer::Save(const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    uint64_t vocab_size = vocab_.size();
    file.write(reinterpret_cast<const char*>(&vocab_size), sizeof(vocab_size));

    for (const auto& token : vocab_) {
        uint64_t length = token.size();
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        file.write(token.data(), static_cast<std::streamsize>(token.size()));
    }

    file.close();
}

void BPETokenizer::Load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    vocab_.clear();
    token_to_id_.clear();
    trie.Clear();
    uint64_t vocab_size = 0;
    file.read(reinterpret_cast<char*>(&vocab_size), sizeof(vocab_size));

    if (!file) {
        throw std::runtime_error("Invalid tokenizer file: " + filename);
    }

    for (uint64_t i = 0; i < vocab_size; i++) {
        uint64_t length = 0;
        file.read(reinterpret_cast<char*>(&length), sizeof(length));
        if (!file) {
            throw std::runtime_error("Invalid tokenizer file: " + filename);
        }

        std::string token(length, '\0');
        if (length > 0) {
            file.read(token.data(), static_cast<std::streamsize>(length));
        }

        if (!file) {
            throw std::runtime_error("Invalid tokenizer file: " + filename);
        }
        AddToken(token);
    }
    file.close();
}


std::vector<size_t> BPETokenizer::Encode(const std::string& str) {
    std::vector<size_t> ids;

    size_t i = 0;

    while (i < str.size()) {
        std::string prefix = trie.LongestToken(str, i);

        if (!prefix.empty()) {
            ids.push_back(token_to_id_[prefix]);
            i += prefix.size();
        } else {
            ids.push_back(token_to_id_["<UNK>"]);
            i++;
        }
    }

    return ids;
}

std::string BPETokenizer::Decode(std::vector<size_t> ids) {
    std::string result = "";
    for (const auto& id : ids) {
        result += vocab_[id];
    }
    return result;
}

size_t BPETokenizer::GetVocabSize() const {
    return vocab_.size();
}

size_t BPETokenizer::GetTokenId(const std::string& token) const {
    return token_to_id_.at(token);
}