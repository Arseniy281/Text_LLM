#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"
#include "../Engine/Tensor/tensor.h"
#include "../Engine/Tensor/backend.h"
#include "../Engine/Tensor/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

// ============================================================
// CUDA helpers
// ============================================================

void CopyToCUDA(Tensor& tensor, const std::vector<float>& data) {
    if (tensor.GetDevice() != Device::CUDA) {
        throw std::runtime_error("CopyToCUDA: tensor is not CUDA");
    }

    if (tensor.GetSize() != data.size()) {
        throw std::runtime_error("CopyToCUDA: size mismatch");
    }

    cudaError_t error = cudaMemcpy(
        tensor.Data(),
        data.data(),
        data.size() * sizeof(float),
        cudaMemcpyHostToDevice
    );

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CopyToCUDA failed: ") +
            cudaGetErrorString(error)
        );
    }
}

std::vector<float> CopyFromCUDA(const Tensor& tensor) {
    std::vector<float> data(tensor.GetSize());

    cudaError_t error = cudaMemcpy(
        data.data(),
        tensor.Data(),
        tensor.GetSize() * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CopyFromCUDA failed: ") +
            cudaGetErrorString(error)
        );
    }

    return data;
}

// ============================================================
// Statistics
// ============================================================

void PrintTensorStats(
    const std::string& name,
    const Tensor& tensor
) {
    std::cout << "\n--- " << name << " ---\n";

    std::cout << "Shape: ";
    for (size_t x : tensor.GetShape()) {
        std::cout << x << " ";
    }
    std::cout << "\n";

    std::cout << "Device: "
              << (tensor.GetDevice() == Device::CUDA ? "CUDA" : "CPU")
              << "\n";

    std::vector<float> data;

    if (tensor.GetDevice() == Device::CUDA) {
        cudaDeviceSynchronize();
        data = CopyFromCUDA(tensor);
    } else {
        data.resize(tensor.GetSize());

        for (size_t i = 0; i < tensor.GetSize(); i++) {
            data[i] = tensor.at(i);
        }
    }

    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();

    double sum = 0.0;
    double sum_squared = 0.0;

    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t zero_count = 0;

    for (float x : data) {
        if (std::isnan(x)) {
            nan_count++;
            continue;
        }

        if (std::isinf(x)) {
            inf_count++;
            continue;
        }

        min_value = std::min(min_value, x);
        max_value = std::max(max_value, x);

        sum += x;
        sum_squared += static_cast<double>(x) * x;

        if (x == 0.0f) {
            zero_count++;
        }
    }

    double mean = 0.0;

    if (data.size() > 0) {
        mean = sum / static_cast<double>(data.size());
    }

    double norm = std::sqrt(sum_squared);

    std::cout << "Min:       " << min_value << "\n";
    std::cout << "Max:       " << max_value << "\n";
    std::cout << "Mean:      " << mean << "\n";
    std::cout << "Norm:      " << norm << "\n";
    std::cout << "NaN:       " << nan_count << "\n";
    std::cout << "Inf:       " << inf_count << "\n";
    std::cout << "Zeros:     " << zero_count << "\n";

    std::cout << "First values: ";

    size_t count = std::min<size_t>(10, data.size());

    for (size_t i = 0; i < count; i++) {
        std::cout << data[i] << " ";
    }

    std::cout << "\n";
}

// ============================================================
// Compare tensors
// ============================================================

float MaxDifference(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error("MaxDifference: size mismatch");
    }

    float max_diff = 0.0f;

    for (size_t i = 0; i < a.size(); i++) {
        max_diff = std::max(
            max_diff,
            std::abs(a[i] - b[i])
        );
    }

    return max_diff;
}

// ============================================================
// Main
// ============================================================

int main() {
    std::cout << "========================================\n";
    std::cout << "       CUDA TRAINING DIAGNOSTIC\n";
    std::cout << "========================================\n";

    // --------------------------------------------------------
    // Model
    // --------------------------------------------------------

    const size_t VOCAB_SIZE = 256;
    const size_t EMBED_DIM = 16;
    const size_t BLOCKS = 2;
    const size_t HEADS = 2;
    const size_t HIDDEN = 32;

    const float LR = 0.0001f;

    LanguageModel model(
        VOCAB_SIZE,
        EMBED_DIM,
        BLOCKS,
        HEADS,
        HIDDEN,
        Device::CUDA
    );

    CrossEntropyLoss loss;

    // --------------------------------------------------------
    // Tiny dataset
    // --------------------------------------------------------

    std::vector<size_t> input_tokens = {
        104, 101, 108, 108, 111,
        32,
        119, 111, 114, 108, 100
    };

    std::vector<size_t> target_tokens = {
        101, 108, 108, 111, 32,
        119,
        111, 114, 108, 100, 0
    };

    Tensor input(
        {input_tokens.size()},
        Device::CUDA
    );

    Tensor targets(
        {1, target_tokens.size()},
        Device::CUDA
    );

    std::vector<float> input_host(input_tokens.size());

    for (size_t i = 0; i < input_tokens.size(); i++) {
        input_host[i] = static_cast<float>(input_tokens[i]);
    }

    std::vector<float> target_host(target_tokens.size());

    for (size_t i = 0; i < target_tokens.size(); i++) {
        target_host[i] = static_cast<float>(target_tokens[i]);
    }

    CopyToCUDA(input, input_host);
    CopyToCUDA(targets, target_host);

    auto input_ptr =
        std::make_shared<Tensor>(std::move(input));

    // --------------------------------------------------------
    // Initial parameter statistics
    // --------------------------------------------------------

    std::cout << "\n========================================\n";
    std::cout << "       INITIAL PARAMETERS\n";
    std::cout << "========================================\n";

    PrintTensorStats(
        "LM head weights BEFORE",
        model.GetLMHeadWeights()
    );

    PrintTensorStats(
        "LM head bias BEFORE",
        model.GetLMHeadBias()
    );

    PrintTensorStats(
        "Embeddings BEFORE",
        model.GetEmbeddings()
    );

    // --------------------------------------------------------
    // STEP 0
    // --------------------------------------------------------

    std::cout << "\n========================================\n";
    std::cout << "              STEP 0\n";
    std::cout << "========================================\n";

    model.ClearGrad();

    cudaDeviceSynchronize();

    auto logits0 = model.forward(input_ptr);

    cudaDeviceSynchronize();

    PrintTensorStats(
        "Logits BEFORE",
        *logits0
    );

    Tensor loss0 =
        loss.forward(*logits0, targets);

    cudaDeviceSynchronize();

    PrintTensorStats(
        "Loss BEFORE",
        loss0
    );

    // --------------------------------------------------------
    // CE backward ONLY
    // --------------------------------------------------------

    Tensor loss_grad = loss.backward();

    cudaDeviceSynchronize();

    PrintTensorStats(
        "CE gradient",
        loss_grad
    );

    // --------------------------------------------------------
    // Full model backward
    // --------------------------------------------------------

    std::cout << "\nRunning model backward...\n";

    logits0->backward();

    cudaDeviceSynchronize();

    std::cout << "Backward finished.\n";

    // --------------------------------------------------------
    // Parameter gradients
    // --------------------------------------------------------

    std::cout << "\n========================================\n";
    std::cout << "       PARAMETER GRADIENTS\n";
    std::cout << "========================================\n";

    const Tensor& lm_weights =
        model.GetLMHeadWeights();

    const Tensor& lm_bias =
        model.GetLMHeadBias();

    PrintTensorStats(
        "LM head weights",
        lm_weights
    );

    if (lm_weights.Grad() != nullptr) {
        PrintTensorStats(
            "LM head WEIGHT GRAD",
            *lm_weights.Grad()
        );
    } else {
        std::cout << "\nLM head WEIGHT GRAD = nullptr\n";
    }

    if (lm_bias.Grad() != nullptr) {
        PrintTensorStats(
            "LM head BIAS GRAD",
            *lm_bias.Grad()
        );
    } else {
        std::cout << "\nLM head BIAS GRAD = nullptr\n";
    }

    auto embedding_grad = model.GetEmbeddingGrad();

    if (embedding_grad != nullptr) {
        PrintTensorStats(
            "EMBEDDING GRAD",
            *embedding_grad
        );
    } else {
        std::cout << "\nEMBEDDING GRAD = nullptr\n";
    }

    // --------------------------------------------------------
    // Save weights BEFORE update
    // --------------------------------------------------------

    std::vector<float> weights_before =
        CopyFromCUDA(model.GetLMHeadWeights());

    std::vector<float> bias_before =
        CopyFromCUDA(model.GetLMHeadBias());

    // --------------------------------------------------------
    // UPDATE
    // --------------------------------------------------------

    std::cout << "\n========================================\n";
    std::cout << "              UPDATE\n";
    std::cout << "========================================\n";

    model.Update(LR);

    cudaDeviceSynchronize();

    std::vector<float> weights_after =
        CopyFromCUDA(model.GetLMHeadWeights());

    std::vector<float> bias_after =
        CopyFromCUDA(model.GetLMHeadBias());

    float weight_diff =
        MaxDifference(
            weights_before,
            weights_after
        );

    float bias_diff =
        MaxDifference(
            bias_before,
            bias_after
        );

    std::cout << "Max LM weight change: "
              << weight_diff
              << "\n";

    std::cout << "Max LM bias change:   "
              << bias_diff
              << "\n";

    PrintTensorStats(
        "LM head weights AFTER",
        model.GetLMHeadWeights()
    );

    // --------------------------------------------------------
    // STEP 1
    // --------------------------------------------------------

    std::cout << "\n========================================\n";
    std::cout << "              STEP 1\n";
    std::cout << "========================================\n";

    model.ClearGrad();

    cudaDeviceSynchronize();

    auto logits1 = model.forward(input_ptr);

    cudaDeviceSynchronize();

    PrintTensorStats(
        "Logits AFTER UPDATE",
        *logits1
    );

    Tensor loss1 =
        loss.forward(*logits1, targets);

    cudaDeviceSynchronize();

    PrintTensorStats(
        "Loss AFTER UPDATE",
        loss1
    );

    // --------------------------------------------------------
    // RESULT
    // --------------------------------------------------------

    std::vector<float> loss_before_data =
        CopyFromCUDA(loss0);

    std::vector<float> loss_after_data =
        CopyFromCUDA(loss1);

    float loss_before_value =
        loss_before_data[0];

    float loss_after_value =
        loss_after_data[0];

    std::cout << "\n========================================\n";
    std::cout << "              RESULT\n";
    std::cout << "========================================\n";

    std::cout << "Loss before update: "
              << loss_before_value
              << "\n";

    std::cout << "Loss after update:  "
              << loss_after_value
              << "\n";

    std::cout << "Loss difference:    "
              << loss_after_value - loss_before_value
              << "\n";

    std::cout << "\n";

    if (std::isnan(loss_before_value) ||
        std::isnan(loss_after_value)) {

        std::cout << "[FAIL] Loss contains NaN\n";

    } else if (std::isinf(loss_before_value) ||
               std::isinf(loss_after_value)) {

        std::cout << "[FAIL] Loss contains infinity\n";

    } else if (weight_diff == 0.0f &&
               bias_diff == 0.0f) {

        std::cout << "[FAIL] Update did not change LM head weights\n";

    } else if (loss_after_value >= 27.63102f &&
               loss_before_value < 27.63102f) {

        std::cout << "[FAIL] One update pushed the model into "
                     "the probability clamp\n";

    } else {

        std::cout << "[OK] Parameters changed and loss is finite\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "          DIAGNOSTIC FINISHED\n";
    std::cout << "========================================\n";

    return 0;
}