#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <algorithm>

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
// CUDA -> CPU
// ============================================================

std::vector<float> CopyToCPU(const Tensor& tensor) {
    std::vector<float> result(tensor.GetSize());

    if (tensor.GetDevice() == Device::CPU) {
        for (size_t i = 0; i < tensor.GetSize(); ++i) {
            result[i] = tensor.Data()[i];
        }

        return result;
    }

    cudaError_t error = cudaMemcpy(
        result.data(),
        tensor.Data(),
        tensor.GetSize() * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaMemcpy failed: ") +
            cudaGetErrorString(error)
        );
    }

    return result;
}

// ============================================================
// L2 difference
// ============================================================

float DifferenceNorm(
    const std::vector<float>& a,
    const std::vector<float>& b
) {
    if (a.size() != b.size()) {
        throw std::runtime_error(
            "DifferenceNorm: size mismatch"
        );
    }

    float sum = 0.0f;

    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return std::sqrt(sum);
}

// ============================================================
// Max difference
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

    for (size_t i = 0; i < a.size(); ++i) {
        result = std::max(
            result,
            std::fabs(a[i] - b[i])
        );
    }

    return result;
}

// ============================================================
// Один train step
// ============================================================

float TrainStep(
    LanguageModel& model,
    Tensor& input,
    Tensor& target
) {
    model.ClearGrad();

    auto input_ptr =
        std::make_shared<Tensor>(input);

    auto logits =
        model.forward(input_ptr);

    CrossEntropyLoss loss;

    Tensor loss_value =
        loss.forward(
            *logits,
            target
        );

    Tensor loss_grad =
        loss.backward();

    logits->backward(
        loss_grad
    );

    cudaDeviceSynchronize();

    auto loss_cpu =
        CopyToCPU(loss_value);

    model.UpdateAdamW(LR);

    cudaDeviceSynchronize();

    return loss_cpu[0];
}

// ============================================================
// main
// ============================================================

int main() {
    try {
        std::cout
            << "========================================\n"
            << "       ADAMW MODEL INTEGRATION TEST\n"
            << "========================================\n\n";

        // ====================================================
        // CUDA
        // ====================================================

        int device_count = 0;

        cudaError_t error =
            cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDA error: ") +
                cudaGetErrorString(error)
            );
        }

        if (device_count == 0) {
            throw std::runtime_error(
                "No CUDA devices found"
            );
        }

        cudaDeviceProp prop;

        error =
            cudaGetDeviceProperties(&prop, 0);

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDeviceProperties failed: ") +
                cudaGetErrorString(error)
            );
        }

        std::cout
            << "GPU: "
            << prop.name
            << "\n\n";

        // ====================================================
        // Данные
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
                        (index * 17 + 13)
                        % VOCAB_SIZE
                    );

                target_data[index] =
                    static_cast<float>(
                        (index * 31 + 7)
                        % VOCAB_SIZE
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

        std::cout
            << "[OK] Input and target on CUDA.\n";

        // ====================================================
        // Модель
        // ====================================================

        LanguageModel model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        std::cout
            << "[OK] Model created.\n\n";

        // ====================================================
        // Состояние ДО обучения
        // ====================================================

        auto embedding_before =
            CopyToCPU(
                model.GetEmbeddings()
            );

        // ====================================================
        // Первый train step
        // ====================================================

        std::cout
            << "========================================\n"
            << "             STEP 1\n"
            << "========================================\n\n";

        float loss1 =
            TrainStep(
                model,
                input,
                target
            );

        auto embedding_after_step1 =
            CopyToCPU(
                model.GetEmbeddings()
            );

        float change1 =
            DifferenceNorm(
                embedding_before,
                embedding_after_step1
            );

        std::cout
            << "Loss: "
            << loss1
            << "\n";

        std::cout
            << "Embedding change: "
            << change1
            << "\n\n";

        if (change1 == 0.0f) {
            throw std::runtime_error(
                "AdamW step 1 did not change embedding weights"
            );
        }

        std::cout
            << "[OK] AdamW changed embedding weights.\n";

        // ====================================================
        // Второй train step
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "             STEP 2\n"
            << "========================================\n\n";

        float loss2 =
            TrainStep(
                model,
                input,
                target
            );

        auto embedding_after_step2 =
            CopyToCPU(
                model.GetEmbeddings()
            );

        float change2 =
            DifferenceNorm(
                embedding_after_step1,
                embedding_after_step2
            );

        float total_change =
            DifferenceNorm(
                embedding_before,
                embedding_after_step2
            );

        float step_difference =
            MaxDifference(
                embedding_after_step1,
                embedding_after_step2
            );

        std::cout
            << "Loss: "
            << loss2
            << "\n";

        std::cout
            << "Embedding change: "
            << change2
            << "\n";

        std::cout
            << "Total embedding change: "
            << total_change
            << "\n";

        std::cout
            << "Max parameter difference: "
            << step_difference
            << "\n\n";

        if (change2 == 0.0f) {
            throw std::runtime_error(
                "AdamW step 2 did not change embedding weights"
            );
        }

        std::cout
            << "[OK] AdamW changed embedding again.\n";

        // ====================================================
        // Проверка loss
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "             RESULT\n"
            << "========================================\n\n";

        std::cout
            << "Loss step 1: "
            << loss1
            << "\n";

        std::cout
            << "Loss step 2: "
            << loss2
            << "\n";

        if (!std::isfinite(loss1) ||
            !std::isfinite(loss2)) {
            throw std::runtime_error(
                "Loss became NaN or Inf"
            );
        }

        std::cout
            << "[OK] Loss values are finite.\n";

        std::cout
            << "\n========================================\n"
            << "       [OK] ADAMW INTEGRATION PASSED\n"
            << "========================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "\n========================================\n"
            << "                FAILED\n"
            << "========================================\n\n"
            << e.what()
            << "\n";

        return 1;
    }
}