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
    const std::vector<int>& tokens
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
    const std::vector<int>& tokens
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

std::vector<int> PrepareBatch(
    const std::string& text,
    BPETokenizer& tokenizer
) {
    std::vector<size_t> ids =
        tokenizer.Encode(text);

    std::vector<int> tokens;

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

        std::cout
            << "=== CUDA Training & Generation Test ===\n\n";

        using Clock =
            std::chrono::high_resolution_clock;


        // ====================================================
        // CUDA DEVICE
        // ====================================================

        int device_count = 0;

        CheckCUDA(
            cudaGetDeviceCount(&device_count),
            "cudaGetDeviceCount"
        );

        if (device_count == 0) {
            throw std::runtime_error(
                "No CUDA devices found"
            );
        }

        cudaDeviceProp prop{};

        CheckCUDA(
            cudaGetDeviceProperties(
                &prop,
                0
            ),
            "cudaGetDeviceProperties"
        );

        std::cout
            << "CUDA device: "
            << prop.name
            << "\n";

        std::cout
            << "VRAM: "
            << static_cast<double>(
                prop.totalGlobalMem
            ) / (1024.0 * 1024.0)
            << " MB\n\n";


        // ====================================================
        // TOKENIZER
        // ====================================================

        BPETokenizer tokenizer;

        std::string corpus =
            "hello world hello world hello world";

        tokenizer.Train(
            corpus,
            50
        );

        tokenizer.Save(
            "vocab.txt"
        );

        std::cout
            << "Vocabulary size: "
            << tokenizer.GetVocabSize()
            << "\n\n";


        // ====================================================
        // MODEL
        // ====================================================

        size_t vocab_size =
            tokenizer.GetVocabSize();

        size_t embed_dim = 16;
        size_t num_blocks = 2;
        size_t num_heads = 2;
        size_t hidden_dim = 32;

        float learning_rate = 0.01f;

        int epochs = 1000;

        LanguageModel model(
            vocab_size,
            embed_dim,
            num_blocks,
            num_heads,
            hidden_dim,
            Device::CUDA
        );

        std::cout
            << "Model created on CUDA\n\n";


        // ====================================================
        // DATA
        // ====================================================

        std::vector<int> tokens =
            PrepareBatch(
                "hello world",
                tokenizer
            );

        std::cout
            << "Tokens: ";

        for (int token : tokens) {
            std::cout
                << token
                << " ";
        }

        std::cout << "\n\n";

        if (tokens.size() < 2) {
            throw std::runtime_error(
                "Not enough tokens for training"
            );
        }


        // ====================================================
        // LOSS
        // ====================================================

        CrossEntropyLoss loss_fn;


        // ====================================================
        // TRAINING
        // ====================================================

        std::cout
            << "========================================\n";
        std::cout
            << "        CUDA TRAINING\n";
        std::cout
            << "========================================\n\n";

        model.SetUseKVCache(false);


        double total_forward_ms = 0.0;
        double total_loss_ms = 0.0;
        double total_backward_ms = 0.0;
        double total_update_ms = 0.0;
        double total_step_ms = 0.0;

        size_t total_steps = 0;

        auto training_start =
            Clock::now();


        for (int epoch = 0;
             epoch < epochs;
             epoch++) {

            float total_loss = 0.0f;

            double epoch_forward_ms = 0.0;
            double epoch_loss_ms = 0.0;
            double epoch_backward_ms = 0.0;
            double epoch_update_ms = 0.0;
            double epoch_total_ms = 0.0;


            auto epoch_start =
                Clock::now();


            // ------------------------------------------------
            // Каждая позиция является отдельным training step
            // ------------------------------------------------

            for (size_t i = 0;
                 i < tokens.size() - 1;
                 i++) {

                auto step_start =
                    Clock::now();


                // ============================================
                // RESET CACHE
                // ============================================

                model.ResetCache();


                // ============================================
                // INPUT
                //
                // Например:
                //
                // i = 0
                // input  = [hello]
                //
                // i = 1
                // input  = [hello world]
                //
                // ============================================

                size_t seq_len = i + 1;

                std::vector<float> host_input(
                    seq_len
                );

                for (size_t j = 0;
                     j < seq_len;
                     j++) {

                    host_input[j] =
                        static_cast<float>(
                            tokens[j]
                        );
                }

                Tensor input =
                    MakeCudaTensor(
                        {1, seq_len},
                        host_input
                    );

                auto input_ptr =
                    std::make_shared<Tensor>(
                        std::move(input)
                    );


                // ============================================
                // TARGETS
                //
                // Для:
                //
                // input  = [hello world]
                //
                // target = [world ?]
                //
                // Но CE работает по всей последовательности,
                // поэтому делаем target для каждой позиции.
                //
                // Здесь последний target известен точно.
                //
                // ============================================

                std::vector<float> host_targets(
                    seq_len
                );

                for (size_t j = 0;
                     j < seq_len;
                     j++) {

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

                Tensor targets =
                    MakeCudaTensor(
                        {1, seq_len},
                        host_targets
                    );


                // ============================================
                // FORWARD
                // ============================================

                auto forward_start =
                    Clock::now();

                auto logits_ptr =
                    model.forward(
                        input_ptr
                    );

                CheckTensorCUDA(
                    *logits_ptr,
                    "training logits"
                );

                CheckCUDA(
                    cudaDeviceSynchronize(),
                    "forward cudaDeviceSynchronize"
                );

                auto forward_end =
                    Clock::now();

                double forward_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        forward_end -
                        forward_start
                    ).count();

                epoch_forward_ms +=
                    forward_ms;

                total_forward_ms +=
                    forward_ms;


                // ============================================
                // LOSS
                // ============================================

                auto loss_start =
                    Clock::now();

                Tensor loss =
                    loss_fn.forward(
                        *logits_ptr,
                        targets
                    );

                CheckTensorCUDA(
                    loss,
                    "loss"
                );

                CheckCUDA(
                    cudaDeviceSynchronize(),
                    "loss cudaDeviceSynchronize"
                );

                float loss_value =
                    GetCudaScalar(loss);

                auto loss_end =
                    Clock::now();

                double loss_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        loss_end -
                        loss_start
                    ).count();

                epoch_loss_ms +=
                    loss_ms;

                total_loss_ms +=
                    loss_ms;

                total_loss +=
                    loss_value;


                // ============================================
                // BACKWARD
                // ============================================

                auto backward_start =
                    Clock::now();

                Tensor grad_logits =
                    loss_fn.backward();

                CheckTensorCUDA(
                    grad_logits,
                    "grad_logits"
                );

                logits_ptr->backward(
                    grad_logits
                );

                CheckCUDA(
                    cudaDeviceSynchronize(),
                    "backward cudaDeviceSynchronize"
                );

                auto backward_end =
                    Clock::now();

                double backward_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        backward_end -
                        backward_start
                    ).count();

                epoch_backward_ms +=
                    backward_ms;

                total_backward_ms +=
                    backward_ms;


                // ============================================
                // UPDATE
                // ============================================

                auto update_start =
                    Clock::now();

                model.Update(
                    learning_rate
                );

                CheckCUDA(
                    cudaDeviceSynchronize(),
                    "update cudaDeviceSynchronize"
                );

                model.ClearGrad();

                auto update_end =
                    Clock::now();

                double update_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        update_end -
                        update_start
                    ).count();

                epoch_update_ms +=
                    update_ms;

                total_update_ms +=
                    update_ms;


                // ============================================
                // STEP TOTAL
                // ============================================

                auto step_end =
                    Clock::now();

                double step_ms =
                    std::chrono::duration<
                        double,
                        std::milli
                    >(
                        step_end -
                        step_start
                    ).count();

                epoch_total_ms +=
                    step_ms;

                total_step_ms +=
                    step_ms;

                total_steps++;
            }


            auto epoch_end =
                Clock::now();

            double real_epoch_ms =
                std::chrono::duration<
                    double,
                    std::milli
                >(
                    epoch_end -
                    epoch_start
                ).count();


            // =================================================
            // LOGGING
            // =================================================

            if (
                epoch % 100 == 0 ||
                epoch == epochs - 1
            ) {

                double steps_per_second =
                    epoch_total_ms > 0.0
                        ? static_cast<double>(
                              tokens.size() - 1
                          ) /
                          (
                              epoch_total_ms /
                              1000.0
                          )
                        : 0.0;

                std::cout
                    << "Epoch "
                    << std::setw(4)
                    << epoch
                    << " | Loss: "
                    << std::fixed
                    << std::setprecision(6)
                    << total_loss /
                       static_cast<float>(
                           tokens.size() - 1
                       )
                    << "\n";

                std::cout
                    << "    Forward : "
                    << std::setprecision(3)
                    << epoch_forward_ms
                    << " ms\n";

                std::cout
                    << "    Loss    : "
                    << epoch_loss_ms
                    << " ms\n";

                std::cout
                    << "    Backward: "
                    << epoch_backward_ms
                    << " ms\n";

                std::cout
                    << "    Update  : "
                    << epoch_update_ms
                    << " ms\n";

                std::cout
                    << "    Total   : "
                    << real_epoch_ms
                    << " ms\n";

                std::cout
                    << "    Steps/s : "
                    << steps_per_second
                    << "\n\n";
            }
        }


        auto training_end =
            Clock::now();

        double training_ms =
            std::chrono::duration<
                double,
                std::milli
            >(
                training_end -
                training_start
            ).count();


        // ====================================================
        // PROFILE
        // ====================================================

        std::cout << "\n";
        std::cout
            << "========================================\n";
        std::cout
            << "        CUDA TRAINING PROFILE\n";
        std::cout
            << "========================================\n";

        std::cout
            << "Total training time: "
            << training_ms / 1000.0
            << " s\n";

        std::cout
            << "Total steps: "
            << total_steps
            << "\n\n";

        std::cout
            << "Forward total : "
            << total_forward_ms
            << " ms\n";

        std::cout
            << "Loss total    : "
            << total_loss_ms
            << " ms\n";

        std::cout
            << "Backward total: "
            << total_backward_ms
            << " ms\n";

        std::cout
            << "Update total  : "
            << total_update_ms
            << " ms\n";

        std::cout
            << "Step total    : "
            << total_step_ms
            << " ms\n";


        if (total_step_ms > 0.0) {

            std::cout << "\n";

            std::cout
                << "Forward share : "
                << total_forward_ms /
                   total_step_ms *
                   100.0
                << " %\n";

            std::cout
                << "Loss share    : "
                << total_loss_ms /
                   total_step_ms *
                   100.0
                << " %\n";

            std::cout
                << "Backward share: "
                << total_backward_ms /
                   total_step_ms *
                   100.0
                << " %\n";

            std::cout
                << "Update share  : "
                << total_update_ms /
                   total_step_ms *
                   100.0
                << " %\n";
        }

        double steps_per_second =
            total_steps /
            (training_ms / 1000.0);

        std::cout << "\n";

        std::cout
            << "Steps/sec : "
            << steps_per_second
            << "\n";

        std::cout
            << "Tokens/sec: "
            << steps_per_second
            << "\n";

        std::cout
            << "========================================\n";


        // ====================================================
        // PREDICTIONS
        // ====================================================

        TestPredictions(
            model,
            tokens
        );


        // ====================================================
        // KV CACHE
        // ====================================================

        std::vector<int> test_tokens = {
            7, 4, 11, 11, 14,
            22, 14, 17, 11, 3
        };

        TestFullVsKVCache(
            model,
            test_tokens
        );


        // ====================================================
        // GENERATION
        // ====================================================

        std::cout
            << "\n=== CUDA Generation ===\n";

        model.SetUseKVCache(true);
        model.ResetCache();

        size_t start_token =
            static_cast<size_t>(
                tokens[0]
            );

        int end_token_id = -1;

        int max_len = 20;

        float temperature = 1.0f;

        float top_p = 0.0f;

        std::cout
            << "Generating with greedy sample:\n";

        std::vector<size_t> generated =
            model.generate(
                {start_token},
                max_len,
                temperature,
                top_p,
                end_token_id
            );

        std::cout
            << "Tokens: ";

        for (size_t t : generated) {
            std::cout
                << t
                << " ";
        }

        std::cout << "\n";

        std::cout
            << "Text: ";

        PrintTokens(
            generated,
            tokenizer
        );

        std::cout << "\n";

        model.SetUseKVCache(false);
        model.ResetCache();


        std::cout
            << "=== CUDA test completed ===\n";


        return 0;

    } catch (const std::exception& e) {

        std::cerr
            << "\n========================================\n";
        std::cerr
            << "CUDA TEST FAILED\n";
        std::cerr
            << "========================================\n";
        std::cerr
            << e.what()
            << "\n";

        cudaDeviceSynchronize();

        return 1;
    }
}