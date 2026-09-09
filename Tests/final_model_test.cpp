#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <memory>
#include <algorithm>

#include <cuda_runtime.h>

#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"
#include "../Engine/Tensor/device.h"


// ============================================================
// CUDA helpers
// ============================================================

void CheckCUDA(cudaError_t error, const std::string& message) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            message + ": " + cudaGetErrorString(error)
        );
    }
}


// ------------------------------------------------------------
// Создать CUDA Tensor из vector<float>
// ------------------------------------------------------------

Tensor MakeCudaTensor(
    const std::vector<size_t>& shape,
    const std::vector<float>& data
) {
    Tensor result(shape, Device::CUDA);

    if (result.GetSize() != data.size()) {
        throw std::runtime_error(
            "MakeCudaTensor: size mismatch"
        );
    }

    if (!data.empty()) {
        CheckCUDA(
            cudaMemcpy(
                result.Data(),
                data.data(),
                data.size() * sizeof(float),
                cudaMemcpyHostToDevice
            ),
            "cudaMemcpy HostToDevice"
        );
    }

    return result;
}


// ------------------------------------------------------------
// CUDA Tensor -> vector<float>
// ------------------------------------------------------------

std::vector<float> CopyToHost(const Tensor& tensor) {
    std::vector<float> result(tensor.GetSize());

    if (tensor.GetSize() == 0) {
        return result;
    }

    CheckCUDA(
        cudaMemcpy(
            result.data(),
            tensor.Data(),
            tensor.GetSize() * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "cudaMemcpy DeviceToHost"
    );

    return result;
}


// ------------------------------------------------------------
// CUDA scalar -> float
// ------------------------------------------------------------

float GetCudaScalar(const Tensor& tensor) {
    if (tensor.GetSize() != 1) {
        throw std::runtime_error(
            "GetCudaScalar: tensor is not scalar"
        );
    }

    std::vector<float> data = CopyToHost(tensor);
    return data[0];
}


// ------------------------------------------------------------
// Проверка device
// ------------------------------------------------------------

void CheckTensorCUDA(const Tensor& tensor, const std::string& name) {
    if (tensor.GetDevice() != Device::CUDA) {
        throw std::runtime_error(
            name + " is not a CUDA tensor"
        );
    }
}


// ============================================================
// Attention KV-cache test
// ============================================================

void TestAttentionKVCache() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " ATTENTION FULL vs KV-CACHE CUDA\n";
    std::cout << "========================================\n";

    const size_t embed_dim = 16;
    const size_t num_heads = 2;
    const size_t seq_len = 4;

    MultiHeadAttention attention(
        embed_dim,
        num_heads,
        Device::CUDA
    );

    // --------------------------------------------------------
    // Один и тот же вход
    // --------------------------------------------------------

    std::vector<float> host_x(
        seq_len * embed_dim
    );

    for (size_t i = 0; i < host_x.size(); i++) {
        host_x[i] =
            0.01f * static_cast<float>(i + 1);
    }

    Tensor x = MakeCudaTensor(
        {1, seq_len, embed_dim},
        host_x
    );

    CheckTensorCUDA(x, "x");

    // --------------------------------------------------------
    // FULL
    // --------------------------------------------------------

    std::cout << "\n[1] FULL FORWARD\n";

    attention.SetUseKVCache(false);

    std::vector<std::vector<float>> full_outputs;

    for (size_t len = 1; len <= seq_len; len++) {

        attention.ResetCache();

        std::vector<float> prefix(
            len * embed_dim
        );

        for (size_t i = 0; i < len; i++) {
            for (size_t j = 0; j < embed_dim; j++) {
                prefix[i * embed_dim + j] =
                    host_x[i * embed_dim + j];
            }
        }

        Tensor input = MakeCudaTensor(
            {1, len, embed_dim},
            prefix
        );

        auto input_ptr =
            std::make_shared<Tensor>(
                std::move(input)
            );

        auto output =
            attention.forward(input_ptr);

        CheckTensorCUDA(
            *output,
            "attention full output"
        );

        std::vector<float> host_output =
            CopyToHost(*output);

        std::vector<float> last(embed_dim);

        for (size_t j = 0; j < embed_dim; j++) {
            last[j] =
                host_output[
                    (len - 1) * embed_dim + j
                ];
        }

        full_outputs.push_back(last);

        std::cout
            << "FULL position "
            << (len - 1)
            << " completed\n";
    }

    // --------------------------------------------------------
    // KV CACHE
    // --------------------------------------------------------

    std::cout << "\n[2] KV-CACHE FORWARD\n";

    attention.SetUseKVCache(true);
    attention.ResetCache();

    std::vector<std::vector<float>> cache_outputs;

    for (size_t pos = 0; pos < seq_len; pos++) {

        std::vector<float> current_input(
            embed_dim
        );

        for (size_t j = 0; j < embed_dim; j++) {
            current_input[j] =
                host_x[pos * embed_dim + j];
        }

        Tensor input = MakeCudaTensor(
            {1, 1, embed_dim},
            current_input
        );

        auto input_ptr =
            std::make_shared<Tensor>(
                std::move(input)
            );

        auto output =
            attention.forward(input_ptr);

        CheckTensorCUDA(
            *output,
            "attention cache output"
        );

        std::vector<float> host_output =
            CopyToHost(*output);

        cache_outputs.push_back(
            host_output
        );

        std::cout
            << "CACHE position "
            << pos
            << " | rope_start_pos="
            << attention.GetRopeStartPos()
            << "\n";
    }

    // --------------------------------------------------------
    // COMPARE
    // --------------------------------------------------------

    std::cout << "\n[3] COMPARISON\n\n";

    float global_max_diff = 0.0f;

    for (size_t pos = 0; pos < seq_len; pos++) {

        float max_diff = 0.0f;
        size_t max_index = 0;

        for (size_t j = 0; j < embed_dim; j++) {

            float diff =
                std::abs(
                    full_outputs[pos][j] -
                    cache_outputs[pos][j]
                );

            if (diff > max_diff) {
                max_diff = diff;
                max_index = j;
            }
        }

        global_max_diff =
            std::max(
                global_max_diff,
                max_diff
            );

        std::cout
            << "Position "
            << pos
            << " | max_diff="
            << max_diff;

        if (max_diff < 1e-5f) {

            std::cout << "  OK";

        } else {

            std::cout << "  DIFFERENT";

            std::cout
                << " | dimension="
                << max_index;

            std::cout
                << " | FULL="
                << full_outputs[pos][max_index];

            std::cout
                << " | CACHE="
                << cache_outputs[pos][max_index];
        }

        std::cout << "\n";
    }

    std::cout << "\n";
    std::cout
        << "Global max diff: "
        << global_max_diff
        << "\n";

    if (global_max_diff < 1e-5f) {

        std::cout
            << "RESULT: Attention KV-cache CUDA PASS\n";

    } else {

        std::cout
            << "RESULT: Attention KV-cache CUDA FAIL\n";
    }

    attention.SetUseKVCache(false);
    attention.ResetCache();

    std::cout << "========================================\n";
}


// ============================================================
// FULL vs KV CACHE for LanguageModel
// ============================================================

void TestFullVsKVCache(
    LanguageModel& model,
    const std::vector<size_t>& tokens
) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " FULL FORWARD vs KV-CACHE CUDA\n";
    std::cout << "========================================\n";

    if (tokens.empty()) {
        std::cout << "ERROR: empty token sequence\n";
        return;
    }

    // --------------------------------------------------------
    // FULL
    // --------------------------------------------------------

    std::cout << "\n[1] FULL FORWARD\n";

    model.SetUseKVCache(false);

    std::vector<std::vector<float>> full_logits;

    for (size_t len = 1; len <= tokens.size(); len++) {

        model.ResetCache();

        std::vector<float> host_input(len);

        for (size_t i = 0; i < len; i++) {
            host_input[i] =
                static_cast<float>(tokens[i]);
        }

        Tensor input = MakeCudaTensor(
            {1, len},
            host_input
        );

        auto input_ptr =
            std::make_shared<Tensor>(
                std::move(input)
            );

        auto output =
            model.forward(input_ptr);

        CheckTensorCUDA(
            *output,
            "model full output"
        );

        size_t vocab_size =
            output->GetShape()[2];

        std::vector<float> host_output =
            CopyToHost(*output);

        std::vector<float> logits(
            vocab_size
        );

        for (size_t v = 0; v < vocab_size; v++) {

            logits[v] =
                host_output[
                    (len - 1) * vocab_size + v
                ];
        }

        full_logits.push_back(logits);
    }

    std::cout
        << "FULL forward completed for "
        << tokens.size()
        << " positions.\n";

    // --------------------------------------------------------
    // KV CACHE
    // --------------------------------------------------------

    std::cout << "\n[2] KV-CACHE FORWARD\n";

    model.SetUseKVCache(true);
    model.ResetCache();

    std::vector<std::vector<float>> cache_logits;

    for (size_t i = 0; i < tokens.size(); i++) {

        Tensor input = MakeCudaTensor(
            {1, 1},
            {
                static_cast<float>(tokens[i])
            }
        );

        auto input_ptr =
            std::make_shared<Tensor>(
                std::move(input)
            );

        auto output =
            model.forward(input_ptr);

        CheckTensorCUDA(
            *output,
            "model cache output"
        );

        size_t vocab_size =
            output->GetShape()[2];

        std::vector<float> host_output =
            CopyToHost(*output);

        std::vector<float> logits(
            vocab_size
        );

        for (size_t v = 0; v < vocab_size; v++) {
            logits[v] =
                host_output[v];
        }

        cache_logits.push_back(logits);
    }

    std::cout
        << "KV-cache forward completed for "
        << tokens.size()
        << " positions.\n";

    // --------------------------------------------------------
    // COMPARE
    // --------------------------------------------------------

    std::cout << "\n[3] COMPARISON\n\n";

    size_t vocab_size =
        full_logits[0].size();

    float global_max_diff = 0.0f;

    size_t global_position = 0;
    size_t global_token = 0;

    size_t different_predictions = 0;

    for (size_t pos = 0;
         pos < tokens.size();
         pos++) {

        float max_diff = 0.0f;
        size_t max_diff_token = 0;

        for (size_t v = 0;
             v < vocab_size;
             v++) {

            float diff =
                std::abs(
                    full_logits[pos][v] -
                    cache_logits[pos][v]
                );

            if (diff > max_diff) {
                max_diff = diff;
                max_diff_token = v;
            }

            if (diff > global_max_diff) {
                global_max_diff = diff;
                global_position = pos;
                global_token = v;
            }
        }

        size_t full_prediction = 0;

        for (size_t v = 1;
             v < vocab_size;
             v++) {

            if (
                full_logits[pos][v] >
                full_logits[pos][full_prediction]
            ) {
                full_prediction = v;
            }
        }

        size_t cache_prediction = 0;

        for (size_t v = 1;
             v < vocab_size;
             v++) {

            if (
                cache_logits[pos][v] >
                cache_logits[pos][cache_prediction]
            ) {
                cache_prediction = v;
            }
        }

        bool same_prediction =
            full_prediction ==
            cache_prediction;

        if (!same_prediction) {
            different_predictions++;
        }

        std::cout
            << "Position "
            << pos
            << " | input="
            << tokens[pos]
            << " | FULL="
            << full_prediction
            << " | CACHE="
            << cache_prediction
            << " | max_diff="
            << max_diff;

        if (
            same_prediction &&
            max_diff < 1e-4f
        ) {

            std::cout << "  OK";

        } else if (same_prediction) {

            std::cout
                << "  PREDICTION OK, LOGITS DIFFER";

        } else {

            std::cout
                << "  DIFFERENT";
        }

        std::cout << "\n";

        if (max_diff >= 1e-4f) {

            std::cout
                << "    largest diff at vocab token "
                << max_diff_token
                << "\n";

            std::cout
                << "    FULL  = "
                << full_logits[pos][max_diff_token]
                << "\n";

            std::cout
                << "    CACHE = "
                << cache_logits[pos][max_diff_token]
                << "\n";
        }
    }

    std::cout << "\n";
    std::cout
        << "----------------------------------------\n";

    std::cout
        << "Different predictions: "
        << different_predictions
        << " / "
        << tokens.size()
        << "\n";

    std::cout
        << "Global max diff: "
        << global_max_diff
        << "\n";

    std::cout
        << "Global max diff position: "
        << global_position
        << "\n";

    std::cout
        << "Global max diff vocab token: "
        << global_token
        << "\n";

    bool passed =
        different_predictions == 0 &&
        global_max_diff < 1e-4f;

    if (passed) {

        std::cout
            << "\nRESULT: FULL and KV-CACHE "
            << "are equivalent. PASS\n";

    } else {

        std::cout
            << "\nRESULT: KV-CACHE IS NOT EQUIVALENT. FAIL\n";
    }

    std::cout
        << "----------------------------------------\n";

    model.SetUseKVCache(false);
    model.ResetCache();

    std::cout << "KV-cache test finished.\n";
    std::cout << "========================================\n";
}


// ============================================================
// Prediction test
// ============================================================

void TestPredictions(
    LanguageModel& model,
    const std::vector<size_t>& tokens
) {
    std::cout << "\n=== Prediction Test ===\n";

    if (tokens.size() < 2) {
        return;
    }

    for (size_t i = 0;
         i < tokens.size() - 1;
         i++) {

        model.ResetCache();

        std::vector<float> host_input(i + 1);

        for (size_t j = 0; j <= i; j++) {
            host_input[j] =
                static_cast<float>(tokens[j]);
        }

        Tensor input = MakeCudaTensor(
            {1, i + 1},
            host_input
        );

        auto input_ptr =
            std::make_shared<Tensor>(
                std::move(input)
            );

        auto logits_ptr =
            model.forward(input_ptr);

        CheckTensorCUDA(
            *logits_ptr,
            "prediction logits"
        );

        size_t vocab_size =
            logits_ptr->GetShape()[2];

        std::vector<float> host_logits =
            CopyToHost(*logits_ptr);

        size_t pos = i;

        size_t predicted = 0;

        float best =
            host_logits[
                pos * vocab_size
            ];

        for (size_t j = 1;
             j < vocab_size;
             j++) {

            float value =
                host_logits[
                    pos * vocab_size + j
                ];

            if (value > best) {
                best = value;
                predicted = j;
            }
        }

        std::cout
            << "Step "
            << i
            << ": expected "
            << tokens[i + 1]
            << ", predicted "
            << predicted;

        if (
            predicted ==
            static_cast<size_t>(tokens[i + 1])
        ) {

            std::cout << "  OK\n";

        } else {

            std::cout << "  FAIL\n";
        }
    }

    model.ResetCache();
}


// ============================================================
// Prepare tokens
// ============================================================

std::vector<size_t> PrepareBatch(
    const std::string& text,
    BPETokenizer& tokenizer
) {
    std::vector<size_t> ids =
        tokenizer.Encode(text);

    std::vector<size_t> tokens;

    for (size_t id : ids) {
        tokens.push_back(
            static_cast<int>(id)
        );
    }

    return tokens;
}


// ============================================================
// Print tokens
// ============================================================

void PrintTokens(
    const std::vector<size_t>& tokens,
    BPETokenizer& tokenizer
) {
    std::vector<size_t> ids;

    for (size_t t : tokens) {
        ids.push_back(t);
    }

    std::cout
        << tokenizer.Decode(ids)
        << std::endl;
}


// ============================================================
// MAIN
// ============================================================

int main() {
    try {
        // ============================================================
        // CUDA
        // ============================================================

        int device_count = 0;
        CheckCUDA(
            cudaGetDeviceCount(&device_count),
            "cudaGetDeviceCount failed"
        );

        if (device_count == 0) {
            throw std::runtime_error("No CUDA devices found");
        }

        cudaDeviceProp prop{};
        CheckCUDA(
            cudaGetDeviceProperties(&prop, 0),
            "cudaGetDeviceProperties failed"
        );

        std::cout << "\n========================================\n";
        std::cout << "       CUDA TRAINING TEST\n";
        std::cout << "========================================\n";

        std::cout << "Device: " << prop.name << "\n";
        std::cout << "VRAM: "
                  << static_cast<double>(prop.totalGlobalMem) /
                         (1024.0 * 1024.0)
                  << " MB\n";

        // ============================================================
        // Tokenizer
        // ============================================================

        BPETokenizer tokenizer;

        std::string corpus =
            "hello world hello world hello world";

        tokenizer.Train(corpus, 50);
        tokenizer.Save("vocab.txt");

        size_t vocab_size =
            tokenizer.GetVocabSize();

        std::vector<size_t> tokens =
            tokenizer.Encode("hello world");

        if (tokens.size() < 2) {
            throw std::runtime_error(
                "Not enough tokens for training"
            );
        }

        std::cout << "Vocabulary: "
                  << vocab_size
                  << "\n";

        std::cout << "Input tokens: ";

        for (size_t token : tokens) {
            std::cout << token << " ";
        }

        std::cout << "\n";

        // ============================================================
        // Model
        // ============================================================

        size_t embed_dim = 16;
        size_t num_blocks = 2;
        size_t num_heads = 2;
        size_t hidden_dim = 32;

        float learning_rate = 0.0001ff;
        int epochs = 1;

        LanguageModel model(
            vocab_size,
            embed_dim,
            num_blocks,
            num_heads,
            hidden_dim,
            Device::CUDA
        );

        CrossEntropyLoss loss_fn;

        model.SetUseKVCache(false);

        std::cout << "\nModel:\n";
        std::cout << "  Embed dim: " << embed_dim << "\n";
        std::cout << "  Blocks:    " << num_blocks << "\n";
        std::cout << "  Heads:     " << num_heads << "\n";
        std::cout << "  Hidden:    " << hidden_dim << "\n";
        std::cout << "  LR:        " << learning_rate << "\n";
        std::cout << "  Epochs:    " << epochs << "\n";

        // ============================================================
        // Training
        // ============================================================

        std::cout << "\n========================================\n";
        std::cout << "           TRAINING\n";
        std::cout << "========================================\n";

        for (int epoch = 0; epoch < epochs; ++epoch) {

            float total_loss = 0.0f;

            for (size_t i = 0;
                 i < tokens.size() - 1;
                 ++i) {

                model.ResetCache();

                size_t seq_len = i + 1;

                // ----------------------------------------------------
                // Input
                // ----------------------------------------------------

                std::vector<float> host_input(seq_len);

                for (size_t j = 0; j < seq_len; ++j) {
                    host_input[j] =
                        static_cast<float>(tokens[j]);
                }

                Tensor input = MakeCudaTensor(
                    {1, seq_len},
                    host_input
                );

                auto input_ptr =
                    std::make_shared<Tensor>(
                        std::move(input)
                    );

                // ----------------------------------------------------
                // Targets
                // ----------------------------------------------------

                std::vector<float> host_targets(seq_len);

                for (size_t j = 0; j < seq_len; ++j) {
                    size_t target_index =
                        std::min(
                            j + 1,
                            tokens.size() - 1
                        );

                    host_targets[j] =
                        static_cast<float>(
                            tokens[target_index]
                        );
                }

                Tensor targets = MakeCudaTensor(
                    {1, seq_len},
                    host_targets
                );

                // ----------------------------------------------------
                // Forward
                // ----------------------------------------------------

                auto logits_ptr =
                    model.forward(input_ptr);

                CheckTensorCUDA(
                    *logits_ptr,
                    "training logits"
                );

                CheckCUDA(
                    cudaDeviceSynchronize(),
                    "forward synchronize"
                );

                // ----------------------------------------------------
                // Loss
                // ----------------------------------------------------

                Tensor loss =
                    loss_fn.forward(
                        *logits_ptr,
                        targets
                    );

                CheckTensorCUDA(loss, "loss");

                float loss_value =
                    GetCudaScalar(loss);

                total_loss += loss_value;

                // ----------------------------------------------------
                // Backward
                // ----------------------------------------------------

                Tensor grad_logits =
                    loss_fn.backward();

                CheckTensorCUDA(
                    grad_logits,
                    "grad_logits"
                );

                logits_ptr->backward(grad_logits);

                CheckCUDA(
                    cudaDeviceSynchronize(),
                    "backward synchronize"
                );

                // ----------------------------------------------------
                // Update
                // ----------------------------------------------------

                model.Update(learning_rate);

                CheckCUDA(
                    cudaDeviceSynchronize(),
                    "update synchronize"
                );

                model.ClearGrad();

                std::cout
                    << "Step "
                    << i
                    << " | Loss: "
                    << std::fixed
                    << std::setprecision(5)
                    << loss_value
                    << "\n";
            }

            float average_loss =
                total_loss /
                static_cast<float>(tokens.size() - 1);

            std::cout
                << "\nEpoch "
                << epoch
                << " | Average loss: "
                << std::fixed
                << std::setprecision(5)
                << average_loss
                << "\n";
        }

        // ============================================================
        // Generation
        // ============================================================

        std::cout << "\n========================================\n";
        std::cout << "           GENERATION\n";
        std::cout << "========================================\n";

        model.SetUseKVCache(true);
        model.ResetCache();

        std::vector<size_t> generated = tokens;

        const size_t generation_length = 10;

        for (size_t step = 0;
             step < generation_length;
             ++step) {

            size_t seq_len =
                generated.size();

            std::vector<float> host_input(seq_len);

            for (size_t i = 0; i < seq_len; ++i) {
                host_input[i] =
                    static_cast<float>(generated[i]);
            }

            Tensor input = MakeCudaTensor(
                {1, seq_len},
                host_input
            );

            auto input_ptr =
                std::make_shared<Tensor>(
                    std::move(input)
                );

            auto logits_ptr =
                model.forward(input_ptr);

            CheckTensorCUDA(
                *logits_ptr,
                "generation logits"
            );

            CheckCUDA(
                cudaDeviceSynchronize(),
                "generation synchronize"
            );

            std::vector<float> host_logits =
                CopyToHost(*logits_ptr);

            size_t vocab =
                logits_ptr->GetShape().back();

            size_t offset =
                (logits_ptr->GetShape()[0] *
                 logits_ptr->GetShape()[1] -
                 1) * vocab;

            int best_token = 0;

            float best_value =
                host_logits[offset];

            for (size_t v = 1; v < vocab; ++v) {

                float value =
                    host_logits[offset + v];

                if (value > best_value) {
                    best_value = value;
                    best_token =
                        static_cast<int>(v);
                }
            }

            generated.push_back(best_token);
        }

        std::cout << "\nGenerated tokens:\n";

        for (size_t token : generated) {
            std::cout << token << " ";
        }

        std::cout << "\n";

        std::cout
            << "Decoded: "
            << tokenizer.Decode(generated)
            << "\n";

        std::cout << "\n========================================\n";
        std::cout << "       CUDA TEST PASSED\n";
        std::cout << "========================================\n";

        return 0;
    }
    catch (const std::exception& e) {

        std::cerr << "\n========================================\n";
        std::cerr << "       CUDA TEST FAILED\n";
        std::cerr << "========================================\n";
        std::cerr << e.what() << "\n";

        return 1;
    }
}