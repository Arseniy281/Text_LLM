#include "../Engine/Tensor/tensor.h"
#include "../Engine/Tensor/backend.h"

#include <cuda_runtime.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <stdexcept>

// ============================================================
// Настройки AdamW
// ============================================================

const float LR = 0.001f;
const float BETA1 = 0.9f;
const float BETA2 = 0.999f;
const float EPS = 1e-8f;
const float WEIGHT_DECAY = 0.01f;

// ============================================================
// Проверка
// ============================================================

bool AlmostEqual(float a, float b, float eps = 1e-6f) {
    return std::fabs(a - b) <= eps;
}

void CheckVector(
    const std::string& name,
    const std::vector<float>& actual,
    const std::vector<float>& expected,
    float eps = 1e-6f
) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(name + ": size mismatch");
    }

    bool ok = true;

    for (size_t i = 0; i < actual.size(); ++i) {
        if (!AlmostEqual(actual[i], expected[i], eps)) {
            ok = false;

            std::cout
                << name << "[" << i << "] mismatch: "
                << "actual=" << actual[i]
                << ", expected=" << expected[i]
                << std::endl;
        }
    }

    if (!ok) {
        throw std::runtime_error(name + ": test failed");
    }

    std::cout << "[OK] " << name << std::endl;
}

// ============================================================
// CPU расчёт AdamW
// ============================================================

void AdamWCPU(
    std::vector<float>& parameter,
    std::vector<float>& m,
    std::vector<float>& v,
    const std::vector<float>& gradient,
    size_t step
) {
    float bias_correction1 =
        1.0f - std::pow(BETA1, static_cast<float>(step));

    float bias_correction2 =
        1.0f - std::pow(BETA2, static_cast<float>(step));

    for (size_t i = 0; i < parameter.size(); ++i) {
        float g = gradient[i];

        m[i] =
            BETA1 * m[i]
            + (1.0f - BETA1) * g;

        v[i] =
            BETA2 * v[i]
            + (1.0f - BETA2) * g * g;

        float m_hat =
            m[i] / bias_correction1;

        float v_hat =
            v[i] / bias_correction2;

        parameter[i] -= LR * (
            m_hat / (std::sqrt(v_hat) + EPS)
            + WEIGHT_DECAY * parameter[i]
        );
    }
}

// ============================================================
// main
// ============================================================

int main() {
    try {
        std::cout << "========================================\n";
        std::cout << "        CUDA ADAMW TEST\n";
        std::cout << "========================================\n\n";

        int device_count = 0;

        cudaError_t error =
            cudaGetDeviceCount(&device_count);

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaGetDeviceCount failed: ")
                + cudaGetErrorString(error)
            );
        }

        std::cout
            << "CUDA devices: "
            << device_count
            << "\n";

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
        // Исходные данные
        // ====================================================

        std::vector<float> initial_parameter = {
            1.0f,
            -2.0f,
            0.5f
        };

        std::vector<float> gradient = {
            0.1f,
            -0.2f,
            0.05f
        };

        std::vector<float> initial_m = {
            0.0f,
            0.0f,
            0.0f
        };

        std::vector<float> initial_v = {
            0.0f,
            0.0f,
            0.0f
        };

        // ====================================================
        // CPU expected
        // ====================================================

        std::vector<float> expected_parameter =
            initial_parameter;

        std::vector<float> expected_m =
            initial_m;

        std::vector<float> expected_v =
            initial_v;

        AdamWCPU(
            expected_parameter,
            expected_m,
            expected_v,
            gradient,
            1
        );

        std::cout << "Expected after step 1:\n";

        for (size_t i = 0; i < expected_parameter.size(); ++i) {
            std::cout
                << "  parameter[" << i << "] = "
                << expected_parameter[i]
                << "\n";
        }

        // ====================================================
        // CUDA tensors
        // ====================================================

        Tensor parameter(
            {3},
            Device::CUDA
        );

        Tensor m(
            {3},
            0.0f,
            Device::CUDA
        );

        Tensor v(
            {3},
            0.0f,
            Device::CUDA
        );

        Tensor gradient_tensor(
            {3},
            Device::CUDA
        );

        // CPU → CUDA
        Tensor parameter_cpu(
            {3},
            std::move(initial_parameter)
        );

        Tensor gradient_cpu(
            {3},
            std::move(gradient)
        );

        parameter_cpu.CopyToCUDA(parameter);
        gradient_cpu.CopyToCUDA(gradient_tensor);

        // ====================================================
        // Получаем CUDA backend
        // ====================================================

        Backend& backend =
            GetBackend(Device::CUDA);

        // ====================================================
        // STEP 1
        // ====================================================

        std::cout << "\nRunning AdamW step 1...\n";

        backend.AdamW(
            parameter,
            m,
            v,
            gradient_tensor,
            LR,
            BETA1,
            BETA2,
            EPS,
            WEIGHT_DECAY,
            1
        );

        cudaDeviceSynchronize();

        // ====================================================
        // Копируем результаты CUDA → CPU
        // ====================================================

        Tensor parameter_result(
            {3},
            Device::CPU
        );

        Tensor m_result(
            {3},
            Device::CPU
        );

        Tensor v_result(
            {3},
            Device::CPU
        );

        cudaMemcpy(
            parameter_result.Data(),
            parameter.Data(),
            3 * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            m_result.Data(),
            m.Data(),
            3 * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            v_result.Data(),
            v.Data(),
            3 * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        std::vector<float> actual_parameter(3);
        std::vector<float> actual_m(3);
        std::vector<float> actual_v(3);

        for (size_t i = 0; i < 3; ++i) {
            actual_parameter[i] =
                parameter_result.Data()[i];

            actual_m[i] =
                m_result.Data()[i];

            actual_v[i] =
                v_result.Data()[i];
        }

        // ====================================================
        // Expected m/v после первого шага
        // ====================================================

        std::vector<float> expected_m_step1 = {
            0.01f,
            -0.02f,
            0.005f
        };

        std::vector<float> expected_v_step1 = {
            0.00001f,
            0.00004f,
            0.0000025f
        };

        // ====================================================
        // Проверяем
        // ====================================================

        CheckVector(
            "Parameter step 1",
            actual_parameter,
            expected_parameter,
            1e-5f
        );

        CheckVector(
            "m step 1",
            actual_m,
            expected_m_step1,
            1e-7f
        );

        CheckVector(
            "v step 1",
            actual_v,
            expected_v_step1,
            1e-8f
        );

        // ====================================================
        // STEP 2
        // ====================================================

        std::cout << "\nRunning AdamW step 2...\n";

        AdamWCPU(
            expected_parameter,
            expected_m,
            expected_v,
            {
                0.1f,
                -0.2f,
                0.05f
            },
            2
        );

        backend.AdamW(
            parameter,
            m,
            v,
            gradient_tensor,
            LR,
            BETA1,
            BETA2,
            EPS,
            WEIGHT_DECAY,
            2
        );

        cudaDeviceSynchronize();

        // ====================================================
        // CUDA → CPU
        // ====================================================

        cudaMemcpy(
            parameter_result.Data(),
            parameter.Data(),
            3 * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            m_result.Data(),
            m.Data(),
            3 * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        cudaMemcpy(
            v_result.Data(),
            v.Data(),
            3 * sizeof(float),
            cudaMemcpyDeviceToHost
        );

        for (size_t i = 0; i < 3; ++i) {
            actual_parameter[i] =
                parameter_result.Data()[i];

            actual_m[i] =
                m_result.Data()[i];

            actual_v[i] =
                v_result.Data()[i];
        }

        // ====================================================
        // Проверяем второй шаг
        // ====================================================

        CheckVector(
            "Parameter step 2",
            actual_parameter,
            expected_parameter,
            1e-5f
        );

        CheckVector(
            "m step 2",
            actual_m,
            expected_m,
            1e-7f
        );

        CheckVector(
            "v step 2",
            actual_v,
            expected_v,
            1e-8f
        );

        // ====================================================
        // Финал
        // ====================================================

        std::cout << "\n========================================\n";
        std::cout << "       [OK] ADAMW TEST PASSED\n";
        std::cout << "========================================\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n[FAILED] " << e.what() << "\n";
        return 1;
    }
}