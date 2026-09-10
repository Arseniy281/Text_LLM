#include "../Engine/Layers/language_model.h"
#include "../Engine/Layers/ce_loss.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"
#include "../Engine/Tensor/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// ============================================================
// Настройки модели
// ============================================================

const size_t VOCAB_SIZE = 1000;

const size_t EMBED_DIM = 128;
const size_t BLOCKS = 4;
const size_t HEADS = 4;
const size_t HIDDEN = 512;

const size_t CONTEXT = 128;
const size_t BATCH_SIZE = 8;

// ============================================================
// Настройки обучения
// ============================================================

const size_t STEPS = 20000;

const float LR = 0.001f;
const float BETA1 = 0.9f;
const float BETA2 = 0.999f;
const float EPS = 1e-8f;
const float WEIGHT_DECAY = 0.01f;

// ============================================================
// Logging
// ============================================================

const size_t LOG_EVERY = 100;
const size_t VALIDATE_EVERY = 500;
const size_t CHECKPOINT_EVERY = 2000;

// Сколько random validation batch'ей использовать.
// Не нужно гонять весь корпус для validation.
const size_t VALIDATION_BATCHES = 20;

// ============================================================
// Пути
// ============================================================

const std::string CORPUS_PATH =
    "../Data/master_and_margarita.txt";

const std::string TOKENIZER_PATH =
    "../Models/MargaritaTokenizer";

const std::string CHECKPOINT_DIR =
    "../Models/MargaritaCUDA";

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
// Scalar Tensor -> float
// ============================================================

float GetScalar(const Tensor& tensor) {
    auto values = CopyToCPU(tensor);

    if (values.empty()) {
        throw std::runtime_error(
            "GetScalar: tensor is empty"
        );
    }

    return values[0];
}

// ============================================================
// Загрузка текста
// ============================================================

std::string LoadText(const std::string& filename) {
    std::ifstream file(filename);

    if (!file) {
        throw std::runtime_error(
            "Cannot open corpus: " + filename
        );
    }

    std::string text(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    if (text.empty()) {
        throw std::runtime_error(
            "Corpus is empty"
        );
    }

    return text;
}

// ============================================================
// Создание CUDA batch
// ============================================================

void CreateBatch(
    const std::vector<size_t>& tokens,
    size_t begin,
    size_t end,
    std::mt19937& generator,
    Tensor& input,
    Tensor& target
) {
    if (end <= begin) {
        throw std::runtime_error(
            "CreateBatch: invalid token range"
        );
    }

    const size_t available =
        end - begin;

    if (available <= CONTEXT) {
        throw std::runtime_error(
            "CreateBatch: not enough tokens"
        );
    }

    const size_t max_start =
        available - CONTEXT - 1;

    std::uniform_int_distribution<size_t> distribution(
        0,
        max_start
    );

    std::vector<float> input_data(
        BATCH_SIZE * CONTEXT
    );

    std::vector<float> target_data(
        BATCH_SIZE * CONTEXT
    );

    for (size_t b = 0; b < BATCH_SIZE; ++b) {
        size_t start =
            begin + distribution(generator);

        for (size_t i = 0; i < CONTEXT; ++i) {
            input_data[
                b * CONTEXT + i
            ] = static_cast<float>(
                tokens[start + i]
            );

            target_data[
                b * CONTEXT + i
            ] = static_cast<float>(
                tokens[start + i + 1]
            );
        }
    }

    Tensor input_cpu(
        {BATCH_SIZE, CONTEXT},
        std::move(input_data)
    );

    Tensor target_cpu(
        {BATCH_SIZE, CONTEXT},
        std::move(target_data)
    );

    input = Tensor(
        {BATCH_SIZE, CONTEXT},
        Device::CUDA
    );

    target = Tensor(
        {BATCH_SIZE, CONTEXT},
        Device::CUDA
    );

    input_cpu.CopyToCUDA(input);
    target_cpu.CopyToCUDA(target);
}

// ============================================================
// Один train step
// ============================================================

float TrainStep(
    LanguageModel& model,
    CrossEntropyLoss& loss,
    Tensor& input,
    Tensor& target
) {
    model.ClearGrad();

    auto input_ptr =
        std::make_shared<Tensor>(input);

    auto logits =
        model.forward(input_ptr);

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

    float value =
        GetScalar(loss_value);

    model.UpdateAdamW(
        LR,
        BETA1,
        BETA2,
        EPS,
        WEIGHT_DECAY
    );

    cudaDeviceSynchronize();

    if (!std::isfinite(value)) {
        throw std::runtime_error(
            "Training loss became NaN or Inf"
        );
    }

    return value;
}

// ============================================================
// Validation
// ============================================================

float Validate(
    LanguageModel& model,
    CrossEntropyLoss& loss,
    const std::vector<size_t>& tokens,
    size_t begin,
    size_t end,
    std::mt19937& generator
) {
    double total_loss = 0.0;

    for (size_t i = 0; i < VALIDATION_BATCHES; ++i) {
        Tensor input(
            {BATCH_SIZE, CONTEXT},
            Device::CUDA
        );

        Tensor target(
            {BATCH_SIZE, CONTEXT},
            Device::CUDA
        );

        CreateBatch(
            tokens,
            begin,
            end,
            generator,
            input,
            target
        );

        model.ClearGrad();

        auto input_ptr =
            std::make_shared<Tensor>(input);

        auto logits =
            model.forward(input_ptr);

        Tensor loss_value =
            loss.forward(
                *logits,
                target
            );

        cudaDeviceSynchronize();

        float value =
            GetScalar(loss_value);

        if (!std::isfinite(value)) {
            throw std::runtime_error(
                "Validation loss became NaN or Inf"
            );
        }

        total_loss += value;
    }

    return static_cast<float>(
        total_loss /
        static_cast<double>(VALIDATION_BATCHES)
    );
}

// ============================================================
// main
// ============================================================

int main() {
    try {
        std::cout
            << "========================================\n"
            << "     MARGARITA CUDA TRAINING\n"
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
            cudaGetDeviceProperties(
                &prop,
                0
            );

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string(
                    "cudaGetDeviceProperties failed: "
                ) +
                cudaGetErrorString(error)
            );
        }

        std::cout
            << "GPU: "
            << prop.name
            << "\n\n";

        // ====================================================
        // Configuration
        // ====================================================

        std::cout
            << "Model:\n"
            << "  vocab      = " << VOCAB_SIZE << "\n"
            << "  embed_dim  = " << EMBED_DIM << "\n"
            << "  blocks     = " << BLOCKS << "\n"
            << "  heads      = " << HEADS << "\n"
            << "  hidden     = " << HIDDEN << "\n"
            << "\n"
            << "Training:\n"
            << "  context    = " << CONTEXT << "\n"
            << "  batch      = " << BATCH_SIZE << "\n"
            << "  steps      = " << STEPS << "\n"
            << "  lr         = " << LR << "\n"
            << "  weight dec = " << WEIGHT_DECAY << "\n"
            << "\n";

        // ====================================================
        // Corpus
        // ====================================================

        std::cout
            << "Loading corpus...\n";

        std::string corpus =
            LoadText(CORPUS_PATH);

        std::cout
            << "Corpus chars: "
            << corpus.size()
            << "\n\n";

        // ====================================================
        // Tokenizer
        // ====================================================

        std::cout
            << "Loading tokenizer...\n";

        BPETokenizer tokenizer;

        tokenizer.Load(
            TOKENIZER_PATH
        );

        if (tokenizer.GetVocabSize() != VOCAB_SIZE) {
            throw std::runtime_error(
                "Tokenizer vocabulary size does not match model VOCAB_SIZE"
            );
        }

        std::cout
            << "Tokenizer vocab: "
            << tokenizer.GetVocabSize()
            << "\n";

        // ====================================================
        // Encode
        // ====================================================

        std::cout
            << "Encoding corpus...\n";

        std::vector<size_t> tokens =
            tokenizer.Encode(corpus);

        std::cout
            << "Tokens: "
            << tokens.size()
            << "\n\n";

        if (tokens.size() <
            CONTEXT + 100) {
            throw std::runtime_error(
                "Not enough tokens for training"
            );
        }

        // ====================================================
        // Train / validation split
        // ====================================================

        const size_t validation_tokens =
            tokens.size() / 10;

        const size_t train_end =
            tokens.size() - validation_tokens;

        const size_t validation_begin =
            train_end;

        std::cout
            << "Dataset split:\n"
            << "  train tokens = "
            << train_end
            << "\n"
            << "  valid tokens = "
            << validation_tokens
            << "\n\n";

        // ====================================================
        // Random generator
        // ====================================================

        std::mt19937 generator(42);

        // Отдельный генератор validation,
        // чтобы validation не зависел
        // от количества train steps.
        std::mt19937 validation_generator(12345);

        // ====================================================
        // Model
        // ====================================================

        std::cout
            << "Creating model...\n";

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

        CrossEntropyLoss loss;

        // ====================================================
        // Training
        // ====================================================

        std::filesystem::create_directories(
            CHECKPOINT_DIR
        );

        std::ofstream log_file(
            CHECKPOINT_DIR + "/training.log",
            std::ios::app
        );

        if (!log_file) {
            throw std::runtime_error(
                "Cannot open training.log"
            );
        }

        log_file
            << "\n========================================\n"
            << "NEW TRAINING RUN\n"
            << "========================================\n";

        log_file
            << "Vocab: " << VOCAB_SIZE << "\n"
            << "Embed: " << EMBED_DIM << "\n"
            << "Blocks: " << BLOCKS << "\n"
            << "Heads: " << HEADS << "\n"
            << "Hidden: " << HIDDEN << "\n"
            << "Context: " << CONTEXT << "\n"
            << "Batch: " << BATCH_SIZE << "\n"
            << "Steps: " << STEPS << "\n"
            << "LR: " << LR << "\n"
            << "WeightDecay: " << WEIGHT_DECAY << "\n";

        double loss_sum = 0.0;

        float best_validation_loss =
            std::numeric_limits<float>::infinity();

        auto training_start =
            std::chrono::steady_clock::now();

        auto last_log_time =
            training_start;

        std::cout
            << "========================================\n"
            << "             TRAINING\n"
            << "========================================\n\n";

        for (size_t step = 1;
             step <= STEPS;
             ++step) {

            Tensor input(
                {BATCH_SIZE, CONTEXT},
                Device::CUDA
            );

            Tensor target(
                {BATCH_SIZE, CONTEXT},
                Device::CUDA
            );

            CreateBatch(
                tokens,
                0,
                train_end,
                generator,
                input,
                target
            );

            float current_loss =
                TrainStep(
                    model,
                    loss,
                    input,
                    target
                );

            loss_sum += current_loss;

            // =================================================
            // Logging
            // =================================================

            if (step % LOG_EVERY == 0 ||
                step == 1) {

                const double avg_loss =
                    loss_sum /
                    static_cast<double>(
                        step % LOG_EVERY == 0
                            ? LOG_EVERY
                            : step
                    );

                loss_sum = 0.0;

                auto now =
                    std::chrono::steady_clock::now();

                double elapsed =
                    std::chrono::duration<double>(
                        now - last_log_time
                    ).count();

                double steps_per_second =
                    LOG_EVERY / elapsed;

                last_log_time = now;

                std::cout
                    << "Step "
                    << std::setw(6)
                    << step
                    << " | Loss: "
                    << std::fixed
                    << std::setprecision(5)
                    << current_loss
                    << " | Avg: "
                    << avg_loss
                    << " | "
                    << std::setprecision(2)
                    << steps_per_second
                    << " step/s\n";

                log_file
                    << "Step "
                    << step
                    << " | Loss: "
                    << std::fixed
                    << std::setprecision(6)
                    << current_loss
                    << " | Avg: "
                    << avg_loss
                    << " | "
                    << steps_per_second
                    << " step/s\n";

                log_file.flush();
            }

            // =================================================
            // Validation
            // =================================================

            if (step % VALIDATE_EVERY == 0) {
                float validation_loss =
                    Validate(
                        model,
                        loss,
                        tokens,
                        validation_begin,
                        tokens.size(),
                        validation_generator
                    );

                std::cout
                    << "           Validation loss: "
                    << std::fixed
                    << std::setprecision(5)
                    << validation_loss
                    << "\n";

                log_file
                    << "Validation "
                    << step
                    << " | Loss: "
                    << std::fixed
                    << std::setprecision(6)
                    << validation_loss
                    << "\n";

                log_file.flush();

                if (validation_loss <
                    best_validation_loss) {

                    best_validation_loss =
                        validation_loss;

                    const std::string best_path =
                        CHECKPOINT_DIR +
                        "/best";

                    std::cout
                        << "           New best validation loss.\n"
                        << "           Saving: "
                        << best_path
                        << "\n";

                    cudaDeviceSynchronize();

                    model.SaveModel(
                        best_path
                    );

                    cudaDeviceSynchronize();
                }
            }

            // =================================================
            // Checkpoint
            // =================================================

            if (step % CHECKPOINT_EVERY == 0) {
                const std::string checkpoint =
                    CHECKPOINT_DIR +
                    "/step_" +
                    std::to_string(step);

                std::cout
                    << "           Saving checkpoint: "
                    << checkpoint
                    << "\n";

                cudaDeviceSynchronize();

                model.SaveModel(
                    checkpoint
                );

                cudaDeviceSynchronize();

                std::cout
                    << "           [OK] Checkpoint saved.\n";
            }
        }

        // ====================================================
        // Final checkpoint
        // ====================================================

        std::cout
            << "\nSaving final model...\n";

        cudaDeviceSynchronize();

        model.SaveModel(
            CHECKPOINT_DIR + "/final"
        );

        cudaDeviceSynchronize();

        auto training_end =
            std::chrono::steady_clock::now();

        double total_seconds =
            std::chrono::duration<double>(
                training_end - training_start
            ).count();

        // ====================================================
        // Result
        // ====================================================

        std::cout
            << "\n========================================\n"
            << "          TRAINING FINISHED\n"
            << "========================================\n\n";

        std::cout
            << "Steps: "
            << STEPS
            << "\n";

        std::cout
            << "Time: "
            << std::fixed
            << std::setprecision(2)
            << total_seconds
            << " s\n";

        std::cout
            << "Average: "
            << STEPS / total_seconds
            << " step/s\n";

        std::cout
            << "Best validation loss: "
            << best_validation_loss
            << "\n";

        std::cout
            << "\nModel saved to:\n"
            << CHECKPOINT_DIR
            << "\n";

        std::cout
            << "\n[OK] TRAINING COMPLETED\n";

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