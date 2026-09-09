#pragma once
#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"

void ChatWithModel(LanguageModel& model, BPETokenizer& tokenizer);