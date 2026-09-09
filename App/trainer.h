#pragma once
#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"
#include <vector>

void TrainModel(LanguageModel& model, const std::vector<std::vector<size_t>>& dataset, 
    CrossEntropyLoss& loss, size_t epochs = 10, float learning_rate = 0.01f);

void OverfitOneChunk(
    LanguageModel& model,
    const std::vector<size_t>& chunk,
    CrossEntropyLoss& loss_fn,
    size_t epochs,
    float learning_rate);