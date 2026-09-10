#include "../Engine/Layers/embedding_layer.h"
#include "../Engine/Layers/linear_layer.h"
#include "../Engine/Transformer/transformer.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>

// ============================================================
// Настройки
// ============================================================

const size_t VOCAB_SIZE = 1000;
const size_t EMBED_DIM = 32;
const size_t BLOCKS = 2;
const size_t HEADS = 2;
const size_t HIDDEN = 64;

const size_t BATCH_SIZE = 4;
const size_t CONTEXT = 16;

const float LR = 0.001f;

// ============================================================
// CUDA → CPU
// ============================================================

std::vector<float> CopyToCPU(const Tensor& tensor) {
    std::vector<float> result(tensor.GetSize());

    if (tensor.GetDevice() == Device::CPU) {
        for (size_t i = 0; i < tensor.GetSize(); ++i) {
            result[i] = tensor.Data()[i];
        }

        return result;
    }

    cudaMemcpy(
        result.data(),
        tensor.Data(),
        tensor.GetSize() * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    return result;
}

// ============================================================
// Норма разницы
// ============================================================

float DifferenceNorm(
    const std::vector<float>& before,
    const std::vector<float>& after
) {
    if (before.size() != after.size()) {
        throw std::runtime_error(
            "DifferenceNorm: size mismatch"
        );
    }

    float sum = 0.0f;

    for (size_t i = 0; i < before.size(); ++i) {
        float diff = after[i] - before[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

// ============================================================
// Норма разницы двух моделей
// ============================================================

void CompareTensors(
    const std::string& name,
    const Tensor& before_a,
    const Tensor& after_a,
    const Tensor& before_b,
    const Tensor& after_b
) {
    auto a_before = CopyToCPU(before_a);
    auto a_after = CopyToCPU(after_a);

    auto b_before = CopyToCPU(before_b);
    auto b_after = CopyToCPU(after_b);

    float delta_a =
        DifferenceNorm(a_before, a_after);

    float delta_b =
        DifferenceNorm(b_before, b_after);

    std::vector<float> delta_difference(
        delta_a > 0 ? 1 : 1
    );

    float max_difference = 0.0f;

    for (size_t i = 0; i < a_before.size(); ++i) {
        float da = a_after[i] - a_before[i];
        float db = b_after[i] - b_before[i];

        max_difference =
            std::max(
                max_difference,
                std::fabs(da - db)
            );
    }

    std::cout
        << name
        << "\n"
        << "  Old Update delta: "
        << delta_a
        << "\n"
        << "  AdamW delta:      "
        << delta_b
        << "\n"
        << "  Max delta diff:   "
        << max_difference
        << "\n\n";
}

// ============================================================
// main
// ============================================================

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "   OLD UPDATE vs ADAMW TEST\n";
        std::cout << "========================================\n\n";

        int device_count = 0;

        cudaError_t error =
            cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA error: ")
                + cudaGetErrorString(error)
            );
        }

        if (device_count == 0) {
            throw std::runtime_error(
                "No CUDA devices found"
            );
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);

        std::cout
            << "GPU: "
            << prop.name
            << "\n\n";

        // ====================================================
        // Одинаковые данные
        // ====================================================

        std::vector<float> input_data(
            BATCH_SIZE * CONTEXT
        );

        std::vector<float> target_data(
            BATCH_SIZE * CONTEXT
        );

        for (size_t b = 0; b < BATCH_SIZE; ++b) {
            for (size_t i = 0; i < CONTEXT; ++i) {
                size_t index =
                    b * CONTEXT + i;

                input_data[index] =
                    static_cast<float>(
                        (index * 17 + 13) % VOCAB_SIZE
                    );

                target_data[index] =
                    static_cast<float>(
                        (index * 31 + 7) % VOCAB_SIZE
                    );
            }
        }

        Tensor input_cpu(
            {BATCH_SIZE, CONTEXT},
            input_data
        );

        Tensor target_cpu(
            {BATCH_SIZE, CONTEXT},
            target_data
        );

        Tensor input(
            {BATCH_SIZE, CONTEXT},
            Device::CUDA
        );

        Tensor target(
            {BATCH_SIZE, CONTEXT},
            Device::CUDA
        );

        input_cpu.CopyToCUDA(input);
        target_cpu.CopyToCUDA(target);

        // ====================================================
        // Две модели
        // ====================================================

        LanguageModel old_model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        LanguageModel adamw_model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        // ====================================================
        // ВАЖНО:
        // модели сейчас могут иметь разные random weights.
        //
        // Поэтому сначала сохраняем old_model,
        // а затем загружаем его в adamw_model.
        // ====================================================

        const std::string TEMP_MODEL =
            "/tmp/adamw_compare_model";

        old_model.SaveModel(TEMP_MODEL);

        adamw_model.LoadModel(TEMP_MODEL);

        // ====================================================
        // Forward / backward
        // ====================================================

        std::cout
            << "Running forward/backward...\n\n";

        auto input_ptr_old =
            std::make_shared<Tensor>(input);

        auto input_ptr_adamw =
            std::make_shared<Tensor>(input);

        old_model.ClearGrad();

        auto logits_old =
            old_model.forward(input_ptr_old);

        CrossEntropyLoss loss_old;

        Tensor loss_value_old =
            loss_old.forward(
                *logits_old,
                target
            );

        Tensor loss_grad_old =
            loss_old.backward();

        logits_old->backward(
            loss_grad_old
        );

        adamw_model.ClearGrad();

        auto logits_adamw =
            adamw_model.forward(input_ptr_adamw);

        CrossEntropyLoss loss_adamw;

        Tensor loss_value_adamw =
            loss_adamw.forward(
                *logits_adamw,
                target
            );

        Tensor loss_grad_adamw =
            loss_adamw.backward();

        logits_adamw->backward(
            loss_grad_adamw
        );

        cudaDeviceSynchronize();

        auto old_loss_cpu = CopyToCPU(loss_value_old);
        auto adamw_loss_cpu = CopyToCPU(loss_value_adamw);

        std::cout
            << "Old loss:   "
            << old_loss_cpu[0]
            << "\n";

        std::cout
            << "AdamW loss: "
            << adamw_loss_cpu[0]
            << "\n\n";

        // ====================================================
        // Сохраняем веса ДО update
        // ====================================================

        old_model.SaveModel(
            "/tmp/old_before"
        );

        adamw_model.SaveModel(
            "/tmp/adamw_before"
        );

        // ====================================================
        // Один update
        // ====================================================

        std::cout
            << "Running OLD Update()...\n";

        old_model.Update(LR);

        cudaDeviceSynchronize();

        std::cout
            << "Running AdamW UpdateAdamW()...\n";

        adamw_model.UpdateAdamW(
            LR
        );

        cudaDeviceSynchronize();

        // ====================================================
        // Загружаем копии ДО update
        // ====================================================

        LanguageModel old_before(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        LanguageModel adamw_before(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        old_before.LoadModel(
            "/tmp/old_before"
        );

        adamw_before.LoadModel(
            "/tmp/adamw_before"
        );

        // ====================================================
        // Сравниваем основные параметры
        // ====================================================

        std::cout
            << "\n========================================\n";
        std::cout
            << "          PARAMETER DELTAS\n";
        std::cout
            << "========================================\n\n";

        CompareTensors(
            "Embedding",
            old_before.GetEmbeddings(),
            old_model.GetEmbeddings(),
            adamw_before.GetEmbeddings(),
            adamw_model.GetEmbeddings()
        );

        std::cout
            << "========================================\n";
        std::cout
            << "              TEST DONE\n";
        std::cout
            << "========================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "\n[FAILED] "
            << e.what()
            << "\n";

        return 1;
    }
}