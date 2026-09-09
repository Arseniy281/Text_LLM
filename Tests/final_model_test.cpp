#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"
#include "../Engine/Tensor/device.h"

#include <cuda_runtime.h>

#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <algorithm>

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

// ============================================================
// CUDA synchronization
// ============================================================

void CheckCUDA() {
    cudaError_t error = cudaGetLastError();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error: ") +
            cudaGetErrorString(error)
        );
    }

    error = cudaDeviceSynchronize();

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA sync error: ") +
            cudaGetErrorString(error)
        );
    }
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
    std::vector<float> data(
        tensor.GetSize()
    );

    if (tensor.GetDevice() == Device::CUDA) {

        cudaError_t error = cudaMemcpy(
            data.data(),
            tensor.Data(),
            tensor.GetSize() * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("GetTensorStats CUDA copy failed: ") +
                cudaGetErrorString(error)
            );
        }

    } else {

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
        << ": shape ";

    for (size_t x : tensor.GetShape()) {
        std::cout << x << " ";
    }

    std::cout
        << " | "
        << (tensor.GetDevice() == Device::CUDA
                ? "CUDA"
                : "CPU")

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
// Scalar
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

        float value = 0.0f;

        cudaError_t error = cudaMemcpy(
            &value,
            tensor.Data(),
            sizeof(float),
            cudaMemcpyDeviceToHost
        );

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("GetScalar failed: ") +
                cudaGetErrorString(error)
            );
        }

        return value;
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
// Read text
// ============================================================

std::string ReadFile(
    const std::string& path
) {
    std::ifstream file(path);

    if (!file) {
        throw std::runtime_error(
            "Cannot open file: " + path
        );
    }

    std::string text(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    return text;
}

// ============================================================
// Main
// ============================================================

int main() {

    try {

        std::cout
            << "========================================\n"
            << "       REAL TEXT CUDA MODEL TEST\n"
            << "========================================\n";

        // ====================================================
        // Configuration
        // ====================================================

        const size_t VOCAB_SIZE = 1000;

        const size_t EMBED_DIM = 32;

        const size_t BLOCKS = 2;

        const size_t HEADS = 2;

        const size_t HIDDEN = 64;

        const size_t CONTEXT = 32;

        const size_t STEPS = 1000;

        const float LR = 0.001f;

        const std::string DATA_PATH =
            "../Data/master_and_margarita.txt";

        // ====================================================
        // CUDA device
        // ====================================================

        int device_count = 0;

        cudaGetDeviceCount(&device_count);

        if (device_count == 0) {

            std::cout
                << "[FAIL] CUDA device not found\n";

            return 1;
        }

        std::cout
            << "CUDA devices: "
            << device_count
            << "\n";

        cudaDeviceProp properties;

        cudaGetDeviceProperties(
            &properties,
            0
        );

        std::cout
            << "GPU: "
            << properties.name
            << "\n";

        // ====================================================
        // Load text
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "          LOADING TEXT\n"
            << "========================================\n";

        std::string text =
            ReadFile(DATA_PATH);

        std::cout
            << "Text size: "
            << text.size()
            << " characters\n";

        if (text.empty()) {

            std::cout
                << "[FAIL] Text is empty\n";

            return 1;
        }

        // ====================================================
        // Tokenizer
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "           TOKENIZATION\n"
            << "========================================\n";

        BPETokenizer tokenizer;

        tokenizer.Train(
            text,
            VOCAB_SIZE
        );

        std::vector<size_t> tokens =
            tokenizer.Encode(text);

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n";

        if (tokens.size() <= CONTEXT) {

            std::cout
                << "[FAIL] Not enough tokens\n";

            return 1;
        }

        // ====================================================
        // Check token range
        // ====================================================

        size_t min_token =
            std::numeric_limits<size_t>::max();

        size_t max_token = 0;

        for (size_t token : tokens) {

            min_token =
                std::min(
                    min_token,
                    token
                );

            max_token =
                std::max(
                    max_token,
                    token
                );
        }

        std::cout
            << "Token range: "
            << min_token
            << " - "
            << max_token
            << "\n";

        if (max_token >= VOCAB_SIZE) {

            std::cout
                << "[FAIL] Token exceeds vocabulary\n";

            return 1;
        }

        // ====================================================
        // Model
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "             MODEL\n"
            << "========================================\n";

        LanguageModel model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        // На обучении KV cache не нужен.
        model.SetUseKVCache(false);

        CrossEntropyLoss loss;

        // ====================================================
        // Dataset
        // ====================================================

        std::vector<size_t> input_tokens(
            CONTEXT
        );

        std::vector<size_t> target_tokens(
            CONTEXT
        );

        /*
         * Берём начало реального текста.
         *
         * x: token[i]
         * y: token[i + 1]
         */

        for (size_t i = 0;
             i < CONTEXT;
             ++i) {

            input_tokens[i] =
                tokens[i];

            target_tokens[i] =
                tokens[i + 1];
        }

        Tensor input(
            {CONTEXT},
            Device::CUDA
        );

        Tensor targets(
            {1, CONTEXT},
            Device::CUDA
        );

        std::vector<float> input_host(
            CONTEXT
        );

        std::vector<float> target_host(
            CONTEXT
        );

        for (size_t i = 0;
             i < CONTEXT;
             ++i) {

            input_host[i] =
                static_cast<float>(
                    input_tokens[i]
                );

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

        // ====================================================
        // Print sample
        // ====================================================

        std::cout
            << "\nTraining tokens:\n";

        for (size_t i = 0;
             i < CONTEXT;
             ++i) {

            std::cout
                << input_tokens[i]
                << " -> "
                << target_tokens[i]
                << "\n";
        }

        // ====================================================
        // Initial parameters
        // ====================================================

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

        // ====================================================
        // Initial forward
        // ====================================================

        model.ClearGrad();

        auto logits =
            model.forward(input_ptr);

        CheckCUDA();

        PrintTensorStats(
            "Initial logits",
            *logits
        );

        Tensor initial_loss =
            loss.forward(
                *logits,
                targets
            );

        CheckCUDA();

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

        // ====================================================
        // Save initial parameters
        // ====================================================

        std::vector<float> initial_weights =
            GetTensorStats(
                model.GetLMHeadWeights()
            ).norm > -1
                ? std::vector<float>()
                : std::vector<float>();

        /*
         * Копируем веса отдельно.
         */

        initial_weights.resize(
            model.GetLMHeadWeights().GetSize()
        );

        cudaMemcpy(
            initial_weights.data(),
            model.GetLMHeadWeights().Data(),
            initial_weights.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        std::vector<float> initial_bias(
            model.GetLMHeadBias().GetSize()
        );

        cudaMemcpy(
            initial_bias.data(),
            model.GetLMHeadBias().Data(),
            initial_bias.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        std::vector<float> initial_embeddings(
            model.GetEmbeddings().GetSize()
        );

        cudaMemcpy(
            initial_embeddings.data(),
            model.GetEmbeddings().Data(),
            initial_embeddings.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        // ====================================================
        // Training
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "             TRAINING\n"
            << "========================================\n";

        bool lm_gradient_seen = false;

        bool embedding_gradient_seen = false;

        float last_loss = first_loss;

        for (size_t step = 0;
             step < STEPS;
             ++step) {

            model.ClearGrad();

            auto current_logits =
                model.forward(input_ptr);

            CheckCUDA();

            Tensor current_loss =
                loss.forward(
                    *current_logits,
                    targets
                );

            CheckCUDA();

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

            // ------------------------------------------------
            // Backward
            // ------------------------------------------------

            Tensor loss_grad =
                loss.backward();

            CheckCUDA();

            current_logits->backward(
                loss_grad
            );

            CheckCUDA();

            // ------------------------------------------------
            // LM gradient
            // ------------------------------------------------

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

            // ------------------------------------------------
            // Embedding gradient
            // ------------------------------------------------

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

            // ------------------------------------------------
            // Update
            // ------------------------------------------------

            model.Update(LR);

            CheckCUDA();

            // ------------------------------------------------
            // Logging
            // ------------------------------------------------

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

            last_loss =
                loss_value;
        }

        // ====================================================
        // Final forward
        // ====================================================

        model.ClearGrad();

        auto final_logits =
            model.forward(input_ptr);

        CheckCUDA();

        Tensor final_loss =
            loss.forward(
                *final_logits,
                targets
            );

        CheckCUDA();

        float final_loss_value =
            GetScalar(final_loss);

        // ====================================================
        // Final parameters
        // ====================================================

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

        // ====================================================
        // Copy final parameters
        // ====================================================

        std::vector<float> final_weights(
            model.GetLMHeadWeights().GetSize()
        );

        cudaMemcpy(
            final_weights.data(),
            model.GetLMHeadWeights().Data(),
            final_weights.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        std::vector<float> final_bias(
            model.GetLMHeadBias().GetSize()
        );

        cudaMemcpy(
            final_bias.data(),
            model.GetLMHeadBias().Data(),
            final_bias.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        std::vector<float> final_embeddings(
            model.GetEmbeddings().GetSize()
        );

        cudaMemcpy(
            final_embeddings.data(),
            model.GetEmbeddings().Data(),
            final_embeddings.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        // ====================================================
        // Parameter changes
        // ====================================================

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

        float embedding_change =
            MaxDifference(
                initial_embeddings,
                final_embeddings
            );

        // ====================================================
        // Result
        // ====================================================

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
            << "Embedding change: "
            << embedding_change
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

        // ====================================================
        // Validation
        // ====================================================

        bool success = true;

        if (!std::isfinite(final_loss_value)) {

            std::cout
                << "[FAIL] Final loss is not finite\n";

            success = false;
        }

        if (!lm_gradient_seen) {

            std::cout
                << "[FAIL] LM gradient is zero\n";

            success = false;
        }

        if (!embedding_gradient_seen) {

            std::cout
                << "[FAIL] Embedding gradient is zero\n";

            success = false;
        }

        if (weight_change == 0.0f) {

            std::cout
                << "[FAIL] LM weights did not change\n";

            success = false;
        }

        if (embedding_change == 0.0f) {

            std::cout
                << "[FAIL] Embeddings did not change\n";

            success = false;
        }

        if (final_loss_value >= first_loss) {

            std::cout
                << "[FAIL] Loss did not decrease\n";

            success = false;
        }

        // ====================================================
        // Final
        // ====================================================

        if (success) {

            std::cout
                << "\n[OK] REAL TEXT CUDA TRAINING PASSED\n";

        } else {

            std::cout
                << "\n[FAIL] REAL TEXT CUDA TRAINING FAILED\n";
        }

        std::cout
            << "\n========================================\n"
            << "             TEST FINISHED\n"
            << "========================================\n";

        return success ? 0 : 1;

    } catch (const std::exception& error) {

        std::cerr
            << "\n========================================\n"
            << "                 ERROR\n"
            << "========================================\n"
            << error.what()
            << "\n";

        return 1;
    }
}