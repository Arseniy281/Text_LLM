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
// L2-норма разницы
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
// Максимальная абсолютная разница
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
// Сравнение двух тензоров
// ============================================================

void CompareTensors(
    const std::string& name,
    const Tensor& a,
    const Tensor& b
) {
    auto a_cpu = CopyToCPU(a);
    auto b_cpu = CopyToCPU(b);

    float diff_norm =
        DifferenceNorm(a_cpu, b_cpu);

    float max_diff =
        MaxDifference(a_cpu, b_cpu);

    std::cout
        << name << "\n"
        << "  L2 difference: "
        << diff_norm << "\n"
        << "  Max difference: "
        << max_diff << "\n\n";
}

// ============================================================
// Сравнение изменения параметра
// ============================================================

void CompareUpdate(
    const std::string& name,
    const Tensor& before_old,
    const Tensor& after_old,
    const Tensor& before_adamw,
    const Tensor& after_adamw
) {
    auto old_before = CopyToCPU(before_old);
    auto old_after = CopyToCPU(after_old);

    auto adamw_before = CopyToCPU(before_adamw);
    auto adamw_after = CopyToCPU(after_adamw);

    if (old_before.size() != adamw_before.size()) {
        throw std::runtime_error(
            name + ": size mismatch"
        );
    }

    float old_delta = 0.0f;
    float adamw_delta = 0.0f;
    float max_delta_difference = 0.0f;

    for (size_t i = 0; i < old_before.size(); ++i) {
        float old_change =
            old_after[i] - old_before[i];

        float adamw_change =
            adamw_after[i] - adamw_before[i];

        old_delta +=
            old_change * old_change;

        adamw_delta +=
            adamw_change * adamw_change;

        max_delta_difference =
            std::max(
                max_delta_difference,
                std::fabs(
                    old_change - adamw_change
                )
            );
    }

    old_delta = std::sqrt(old_delta);
    adamw_delta = std::sqrt(adamw_delta);

    std::cout
        << name << "\n"
        << "  Old Update delta: "
        << old_delta << "\n"
        << "  AdamW delta:      "
        << adamw_delta << "\n"
        << "  Max delta diff:   "
        << max_delta_difference << "\n\n";
}

// ============================================================
// main
// ============================================================

int main() {
    try {
        std::cout
            << "========================================\n"
            << "       OLD UPDATE vs ADAMW TEST\n"
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

        std::cout
            << "[1] Creating input data...\n";

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
            << "[2] Input and target copied to CUDA.\n";

        // ====================================================
        // Создаём первую модель
        // ====================================================

        LanguageModel old_model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        std::cout
            << "[3] Old model created.\n";

        // ====================================================
        // Сохраняем её
        // ====================================================

        const std::string INITIAL_MODEL =
            "/tmp/adamw_compare_initial";

        old_model.SaveModel(
            INITIAL_MODEL
        );

        std::cout
            << "[4] Initial model saved.\n";

        // ====================================================
        // Загружаем абсолютно такую же модель
        // ====================================================

        LanguageModel adamw_model(
            VOCAB_SIZE,
            EMBED_DIM,
            BLOCKS,
            HEADS,
            HIDDEN,
            Device::CUDA
        );

        adamw_model.LoadModel(
            INITIAL_MODEL
        );

        std::cout
            << "[5] AdamW model loaded.\n";

        // ====================================================
        // Проверяем, что модели действительно одинаковые
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "        INITIAL MODEL CHECK\n"
            << "========================================\n\n";

        CompareTensors(
            "Embedding before training",
            old_model.GetEmbeddings(),
            adamw_model.GetEmbeddings()
        );

        // ====================================================
        // Forward / backward OLD
        // ====================================================

        std::cout
            << "[6] Running OLD forward/backward...\n";

        old_model.ClearGrad();

        auto input_ptr_old =
            std::make_shared<Tensor>(input);

        auto logits_old =
            old_model.forward(input_ptr_old);

        CrossEntropyLoss loss_old;

        Tensor old_loss =
            loss_old.forward(
                *logits_old,
                target
            );

        Tensor old_loss_grad =
            loss_old.backward();

        logits_old->backward(
            old_loss_grad
        );

        // ====================================================
        // Forward / backward ADAMW
        // ====================================================

        std::cout
            << "[7] Running AdamW forward/backward...\n";

        adamw_model.ClearGrad();

        auto input_ptr_adamw =
            std::make_shared<Tensor>(input);

        auto logits_adamw =
            adamw_model.forward(
                input_ptr_adamw
            );

        CrossEntropyLoss loss_adamw;

        Tensor adamw_loss =
            loss_adamw.forward(
                *logits_adamw,
                target
            );

        Tensor adamw_loss_grad =
            loss_adamw.backward();

        logits_adamw->backward(
            adamw_loss_grad
        );

        cudaDeviceSynchronize();

        // ====================================================
        // Loss
        // ====================================================

        auto old_loss_cpu =
            CopyToCPU(old_loss);

        auto adamw_loss_cpu =
            CopyToCPU(adamw_loss);

        std::cout
            << "\n========================================\n"
            << "              LOSS CHECK\n"
            << "========================================\n\n";

        std::cout
            << "Old loss:   "
            << old_loss_cpu[0]
            << "\n";

        std::cout
            << "AdamW loss: "
            << adamw_loss_cpu[0]
            << "\n";

        float loss_difference =
            std::fabs(
                old_loss_cpu[0] -
                adamw_loss_cpu[0]
            );

        std::cout
            << "Difference:  "
            << loss_difference
            << "\n\n";

        if (loss_difference > 1e-4f) {
            std::cout
                << "[WARNING] Initial losses differ.\n";
        } else {
            std::cout
                << "[OK] Initial losses match.\n";
        }

        // ====================================================
        // Сохраняем веса ДО update
        // ====================================================

        std::cout
            << "\n[8] Saving parameters before update...\n";

        const std::string OLD_BEFORE =
            "/tmp/old_before";

        const std::string ADAMW_BEFORE =
            "/tmp/adamw_before";

        old_model.SaveModel(
            OLD_BEFORE
        );

        adamw_model.SaveModel(
            ADAMW_BEFORE
        );

        // ====================================================
        // UPDATE
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "              OPTIMIZER STEP\n"
            << "========================================\n\n";

        std::cout
            << "Running OLD Update()...\n";

        old_model.Update(
            LR
        );

        cudaDeviceSynchronize();

        std::cout
            << "OLD Update finished.\n";

        std::cout
            << "\nRunning AdamW UpdateAdamW()...\n";

        adamw_model.UpdateAdamW(
            LR
        );

        cudaDeviceSynchronize();

        std::cout
            << "AdamW UpdateAdamW finished.\n";

        // ====================================================
        // Проверяем embedding
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "           EMBEDDING UPDATE\n"
            << "========================================\n\n";

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
            OLD_BEFORE
        );

        adamw_before.LoadModel(
            ADAMW_BEFORE
        );

        CompareUpdate(
            "Embedding",
            old_before.GetEmbeddings(),
            old_model.GetEmbeddings(),
            adamw_before.GetEmbeddings(),
            adamw_model.GetEmbeddings()
        );

        // ====================================================
        // Финальная проверка
        // ====================================================

        auto old_initial_embedding =
            CopyToCPU(
                old_before.GetEmbeddings()
            );

        auto old_final_embedding =
            CopyToCPU(
                old_model.GetEmbeddings()
            );

        auto adamw_initial_embedding =
            CopyToCPU(
                adamw_before.GetEmbeddings()
            );

        auto adamw_final_embedding =
            CopyToCPU(
                adamw_model.GetEmbeddings()
            );

        float old_change =
            DifferenceNorm(
                old_initial_embedding,
                old_final_embedding
            );

        float adamw_change =
            DifferenceNorm(
                adamw_initial_embedding,
                adamw_final_embedding
            );

        std::cout
            << "\n========================================\n"
            << "              FINAL CHECK\n"
            << "========================================\n\n";

        std::cout
            << "Old Update changed weights: ";

        if (old_change > 0.0f) {
            std::cout << "YES\n";
        } else {
            std::cout << "NO\n";
        }

        std::cout
            << "AdamW changed weights:      ";

        if (adamw_change > 0.0f) {
            std::cout << "YES\n";
        } else {
            std::cout << "NO\n";
        }

        std::cout
            << "\nOld embedding change:   "
            << old_change
            << "\n";

        std::cout
            << "AdamW embedding change: "
            << adamw_change
            << "\n";

        if (old_change == 0.0f) {
            throw std::runtime_error(
                "OLD Update did not change embedding weights"
            );
        }

        if (adamw_change == 0.0f) {
            throw std::runtime_error(
                "AdamW did not change embedding weights"
            );
        }

        std::cout
            << "\n========================================\n"
            << "          [OK] TEST PASSED\n"
            << "========================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr
            << "\n========================================\n"
            << "             [FAILED]\n"
            << "========================================\n\n"
            << e.what()
            << "\n";

        return 1;
    }
}