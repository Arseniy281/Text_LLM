#include "book_loader.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include <fstream>
#include <string>
#include <vector>

std::vector<std::vector<size_t>> LoadBook(BPETokenizer& tokenizer, const std::string& book_path, size_t chunk_size) {
    std::ifstream file(book_path);
    if (!file.is_open()) {
        throw std::runtime_error("could not open file: " + book_path);
    }

    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::vector<size_t> tokens = tokenizer.Encode(text);

    std::vector<std::vector<size_t>> chunks;
    for (size_t i = 0; i + chunk_size <= tokens.size(); i += chunk_size) {
        chunks.push_back({tokens.begin() + i, tokens.begin() + i + chunk_size});
    }
    return chunks;
}