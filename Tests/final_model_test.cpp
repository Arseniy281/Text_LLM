#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"
#include "../Engine/Tensor/tensor.h"
#include "../Engine/Tensor/device.h"
#include <iomanip>

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>
#include <memory>
#include <string>

// ============================================================
// CUDA helpers
// ============================================================

void CopyToCUDA(
    Tensor& tensor,
    const std::vector<float>& data
) {
    if (tensor.GetDevice() != Device::CUDA) {
        throw std::runtime_error(
            "CopyToCUDA: tensor is not CUDA"
        );
    }

    if (tensor.GetSize() != data.size()) {
        throw std::runtime_error(
            "CopyToCUDA: size mismatch"
        );
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

std::vector<float> CopyFromCUDA(
    const Tensor& tensor
) {
    std::vector<float> data(
        tensor.GetSize()
    );

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
// Tensor statistics
// ============================================================

struct TensorStats {
    float min = 0.0f;
    float max = 0.0f;

    double mean = 0.0;
    double norm = 0.0;

    size_t nan_count = 0;
    size_t inf_count = 0;
    size_t zero_count = 0;
};

TensorStats GetTensorStats(
    const Tensor& tensor
) {
    std::vector<float> data;

    if (tensor.GetDevice() == Device::CUDA) {
        cudaDeviceSynchronize();
        data = CopyFromCUDA(tensor);
    } else {
        data.resize(tensor.GetSize());

        for (size_t i = 0;
             i < tensor.GetSize();
             ++i) {

            data[i] = tensor.at(i);
        }
    }

    TensorStats stats;

    if (data.empty()) {
        return stats;
    }

    stats.min =
        std::numeric_limits<float>::infinity();

    stats.max =
        -std::numeric_limits<float>::infinity();

    double sum = 0.0;
    double squared = 0.0;

    for (float x : data) {

        if (std::isnan(x)) {
            stats.nan_count++;
            continue;
        }

        if (std::isinf(x)) {
            stats.inf_count++;
            continue;
        }

        stats.min =
            std::min(stats.min, x);

        stats.max =
            std::max(stats.max, x);

        sum += x;

        squared +=
            static_cast<double>(x) * x;

        if (x == 0.0f) {
            stats.zero_count++;
        }
    }

    stats.mean =
        sum / static_cast<double>(data.size());

    stats.norm =
        std::sqrt(squared);

    return stats;
}

void PrintTensorStats(
    const std::string& name,
    const Tensor& tensor
) {
    TensorStats stats =
        GetTensorStats(tensor);

    std::cout
        << name
        << ": ";

    std::cout
        << "shape ";

    for (size_t x : tensor.GetShape()) {
        std::cout << x << " ";
    }

    std::cout
        << " | "
        << (tensor.GetDevice() == Device::CUDA
                ? "CUDA"
                : "CPU");

    std::cout
        << " | min "
        << stats.min
        << " max "
        << stats.max
        << " mean "
        << stats.mean
        << " norm "
        << stats.norm
        << " | NaN "
        << stats.nan_count
        << " Inf "
        << stats.inf_count
        << " zeros "
        << stats.zero_count
        << "\n";
}

// ============================================================
// Loss extraction
// ============================================================

float GetScalar(
    const Tensor& tensor
) {
    if (tensor.GetSize() != 1) {
        throw std::runtime_error(
            "GetScalar: tensor is not scalar"
        );
    }

    if (tensor.GetDevice() == Device::CUDA) {
        std::vector<float> data =
            CopyFromCUDA(tensor);

        return data[0];
    }

    return tensor.at(0);
}

// ============================================================
// Difference
// ============================================================

float MaxDifference(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error(
            "MaxDifference: size mismatch"
        );
    }

    float result = 0.0f;

    for (size_t i = 0;
         i < a.size();
         ++i) {

        result = std::max(
            result,
            std::abs(a[i] - b[i])
        );
    }

    return result;
}

// ============================================================
// Main
// ============================================================

int main() {

    std::cout
        << "========================================\n"
        << "          CUDA MODEL TEST\n"
        << "========================================\n";

    // ========================================================
    // Model configuration
    // ========================================================

    const size_t VOCAB_SIZE = 1000;

    const size_t EMBED_DIM = 32;

    const size_t BLOCKS = 2;

    const size_t HEADS = 2;

    const size_t HIDDEN = 64;

    const size_t CONTEXT = 32;

    const size_t STEPS = 1000;

    const float LR = 0.001f;

    // ========================================================
    // Model
    // ========================================================

    LanguageModel model(
        VOCAB_SIZE,
        EMBED_DIM,
        BLOCKS,
        HEADS,
        HIDDEN,
        Device::CUDA
    );

    CrossEntropyLoss loss;

    // ========================================================
    // Dataset
    // ========================================================

    std::vector<size_t> input_tokens;

    std::vector<size_t> target_tokens;

    for (size_t i = 0;
         i < CONTEXT;
         ++i) {

        input_tokens.push_back(
            100 + (i % 20)
        );

        target_tokens.push_back(
            100 + ((i + 1) % 20)
        );
    }

    Tensor input(
        {input_tokens.size()},
        Device::CUDA
    );

    Tensor targets(
        {1, target_tokens.size()},
        Device::CUDA
    );

    std::vector<float> input_host(
        input_tokens.size()
    );

    std::vector<float> target_host(
        target_tokens.size()
    );

    for (size_t i = 0;
         i < input_tokens.size();
         ++i) {

        input_host[i] =
            static_cast<float>(
                input_tokens[i]
            );
    }

    for (size_t i = 0;
         i < target_tokens.size();
         ++i) {

        target_host[i] =
            static_cast<float>(
                target_tokens[i]
            );
    }

    CopyToCUDA(
        input,
        input_host
    );

    CopyToCUDA(
        targets,
        target_host
    );

    auto input_ptr =
        std::make_shared<Tensor>(
            std::move(input)
        );

    // ========================================================
    // Initial parameters
    // ========================================================

    std::cout
        << "\n========================================\n"
        << "       INITIAL PARAMETERS\n"
        << "========================================\n";

    PrintTensorStats(
        "LM head weights",
        model.GetLMHeadWeights()
    );

    PrintTensorStats(
        "LM head bias",
        model.GetLMHeadBias()
    );

    PrintTensorStats(
        "Embeddings",
        model.GetEmbeddings()
    );

    // ========================================================
    // Initial forward
    // ========================================================

    model.ClearGrad();

    cudaDeviceSynchronize();

    auto logits =
        model.forward(input_ptr);

    cudaDeviceSynchronize();

    PrintTensorStats(
        "Initial logits",
        *logits
    );

    Tensor initial_loss =
        loss.forward(
            *logits,
            targets
        );

    cudaDeviceSynchronize();

    float first_loss =
        GetScalar(initial_loss);

    std::cout
        << "\nInitial loss: "
        << first_loss
        << "\n";

    if (!std::isfinite(first_loss)) {

        std::cout
            << "[FAIL] Initial loss is not finite\n";

        return 1;
    }

    // ========================================================
    // Save initial LM head weights
    // ========================================================

    std::vector<float> initial_weights =
        CopyFromCUDA(
            model.GetLMHeadWeights()
        );

    std::vector<float> initial_bias =
        CopyFromCUDA(
            model.GetLMHeadBias()
        );

    // ========================================================
    // Training
    // ========================================================

    std::cout
        << "\n========================================\n"
        << "             TRAINING\n"
        << "========================================\n";

    float last_loss = first_loss;

    bool embedding_gradient_seen = false;

    bool transformer_gradient_seen = false;

    bool lm_gradient_seen = false;

    for (size_t step = 0;
         step < STEPS;
         ++step) {

        model.ClearGrad();

        cudaDeviceSynchronize();

        auto current_logits =
            model.forward(input_ptr);

        cudaDeviceSynchronize();

        Tensor current_loss =
            loss.forward(
                *current_logits,
                targets
            );

        cudaDeviceSynchronize();

        float loss_value =
            GetScalar(current_loss);

        if (!std::isfinite(loss_value)) {

            std::cout
                << "\n[FAIL] Loss became non-finite "
                << "at step "
                << step
                << "\n";

            return 1;
        }

        // ----------------------------------------------------
        // Backward
        // ----------------------------------------------------

        Tensor loss_grad =
            loss.backward();

        cudaDeviceSynchronize();

        current_logits->backward(
            loss_grad
        );

        cudaDeviceSynchronize();

        // ----------------------------------------------------
        // Check LM head gradient
        // ----------------------------------------------------

        const Tensor& lm_weights =
            model.GetLMHeadWeights();

        if (lm_weights.Grad() != nullptr) {

            TensorStats stats =
                GetTensorStats(
                    *lm_weights.Grad()
                );

            if (stats.norm > 0.0 &&
                stats.nan_count == 0 &&
                stats.inf_count == 0) {

                lm_gradient_seen = true;
            }
        }

        // ----------------------------------------------------
        // Check embedding gradient
        // ----------------------------------------------------

        auto embedding_grad =
            model.GetEmbeddingGrad();

        if (embedding_grad != nullptr) {

            TensorStats stats =
                GetTensorStats(
                    *embedding_grad
                );

            if (stats.norm > 0.0 &&
                stats.nan_count == 0 &&
                stats.inf_count == 0) {

                embedding_gradient_seen = true;
            }
        }

        // ----------------------------------------------------
        // Update
        // ----------------------------------------------------

        model.Update(LR);

        cudaDeviceSynchronize();

        // ----------------------------------------------------
        // Logging
        // ----------------------------------------------------

        if (step % 100 == 0 ||
            step == STEPS - 1) {

            std::cout
                << "Step "
                << std::setw(4)
                << step
                << " | Loss: "
                << std::fixed
                << std::setprecision(6)
                << loss_value
                << "\n";
        }

        last_loss = loss_value;
    }

    // ========================================================
    // Final parameters
    // ========================================================

    std::cout
        << "\n========================================\n"
        << "        FINAL PARAMETERS\n"
        << "========================================\n";

    PrintTensorStats(
        "LM head weights AFTER",
        model.GetLMHeadWeights()
    );

    PrintTensorStats(
        "LM head bias AFTER",
        model.GetLMHeadBias()
    );

    PrintTensorStats(
        "Embeddings AFTER",
        model.GetEmbeddings()
    );

    // ========================================================
    // Parameter change
    // ========================================================

    std::vector<float> final_weights =
        CopyFromCUDA(
            model.GetLMHeadWeights()
        );

    std::vector<float> final_bias =
        CopyFromCUDA(
            model.GetLMHeadBias()
        );

    float weight_change =
        MaxDifference(
            initial_weights,
            final_weights
        );

    float bias_change =
        MaxDifference(
            initial_bias,
            final_bias
        );

    // ========================================================
    // Final forward
    // ========================================================

    model.ClearGrad();

    auto final_logits =
        model.forward(input_ptr);

    cudaDeviceSynchronize();

    Tensor final_loss =
        loss.forward(
            *final_logits,
            targets
        );

    cudaDeviceSynchronize();

    float final_loss_value =
        GetScalar(final_loss);

    // ========================================================
    // Result
    // ========================================================

    std::cout
        << "\n========================================\n"
        << "               RESULT\n"
        << "========================================\n";

    std::cout
        << "Initial loss: "
        << first_loss
        << "\n";

    std::cout
        << "Final loss:   "
        << final_loss_value
        << "\n";

    std::cout
        << "Loss change:  "
        << final_loss_value - first_loss
        << "\n";

    std::cout
        << "LM weight change: "
        << weight_change
        << "\n";

    std::cout
        << "LM bias change:   "
        << bias_change
        << "\n";

    std::cout
        << "LM gradient:      "
        << (lm_gradient_seen
                ? "OK"
                : "ZERO")
        << "\n";

    std::cout
        << "Embedding gradient: "
        << (embedding_gradient_seen
                ? "OK"
                : "ZERO")
        << "\n";

    // ========================================================
    // Final validation
    // ========================================================

    bool success = true;

    if (!std::isfinite(final_loss_value)) {

        std::cout
            << "[FAIL] Final loss is not finite\n";

        success = false;
    }

    if (weight_change == 0.0f &&
        bias_change == 0.0f) {

        std::cout
            << "[FAIL] LM head parameters "
               "did not change\n";

        success = false;
    }

    if (!lm_gradient_seen) {

        std::cout
            << "[FAIL] LM head gradient is zero\n";

        success = false;
    }

    if (!embedding_gradient_seen) {

        std::cout
            << "[FAIL] Embedding gradient is zero\n";

        success = false;
    }

    if (final_loss_value >= first_loss) {

        std::cout
            << "[FAIL] Loss did not decrease\n";

        success = false;
    }

    if (success) {

        std::cout
            << "\n[OK] CUDA training test passed\n";

    } else {

        std::cout
            << "\n[FAIL] CUDA training test failed\n";
    }

    std::cout
        << "\n========================================\n"
        << "             TEST FINISHED\n"
        << "========================================\n";

    return success ? 0 : 1;
}