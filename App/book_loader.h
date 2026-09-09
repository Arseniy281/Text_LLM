#pragma once
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include <vector>
#include <string>

std::vector<std::vector<size_t>> LoadBook(BPETokenizer& tokenizer, const std::string& book_path, size_t chunk_size = 128);