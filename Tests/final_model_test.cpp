#include <chrono>
#include <iomanip>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "../Engine/Layers/language_model.h"
#include "../Engine/Tokenizer/bpe_tokenizer.h"
#include "../Engine/Tensor/tensor.h"
#include <iostream>
#include <vector>
#include <string>
#include <cmath>

#include <cuda_runtime.h>

void TestAttentionKVCache() {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " ATTENTION FULL vs KV-CACHE\n";
    std::cout << "========================================\n";

    const size_t embed_dim = 16;
    const size_t num_heads = 2;
    const size_t seq_len = 4;

    MultiHeadAttention attention(
        embed_dim,
        num_heads
    );

    // ============================================================
    // Создаём один и тот же вход
    // ============================================================

    Tensor x({1, seq_len, embed_dim});

    for (size_t i = 0; i < x.GetSize(); i++) {
        x.at(i) =
            0.01f * static_cast<float>(i + 1);
    }

    // ============================================================
    // FULL FORWARD
    // ============================================================

    std::cout << "\n[1] FULL FORWARD\n";

    attention.SetUseKVCache(false);

    std::vector<std::vector<float>> full_outputs;

    for (size_t len = 1; len <= seq_len; len++) {

        attention.ResetCache();

        auto input =
            std::make_shared<Tensor>(
                Tensor({1, len, embed_dim})
            );

        for (size_t i = 0; i < len; i++) {
            for (size_t j = 0; j < embed_dim; j++) {

                input->at({0, i, j}) =
                    x.at({0, i, j});
            }
        }

        auto output =
            attention.forward(input);

        std::vector<float> last(embed_dim);

        for (size_t j = 0; j < embed_dim; j++) {

            last[j] =
                output->at({
                    0,
                    len - 1,
                    j
                });
        }

        full_outputs.push_back(last);

        std::cout
            << "FULL position "
            << (len - 1)
            << " completed\n";
    }

    // ============================================================
    // KV-CACHE FORWARD
    // ============================================================

    std::cout << "\n[2] KV-CACHE FORWARD\n";

    attention.SetUseKVCache(true);
    attention.ResetCache();

    std::vector<std::vector<float>> cache_outputs;

    for (size_t pos = 0; pos < seq_len; pos++) {

        auto input =
            std::make_shared<Tensor>(
                Tensor({1, 1, embed_dim})
            );

        for (size_t j = 0; j < embed_dim; j++) {

            input->at({0, 0, j}) =
                x.at({0, pos, j});
        }

        auto output =
            attention.forward(input);

        std::vector<float> current(embed_dim);

        for (size_t j = 0; j < embed_dim; j++) {

            current[j] =
                output->at({
                    0,
                    0,
                    j
                });
        }

        cache_outputs.push_back(current);

        std::cout
            << "CACHE position "
            << pos
            << " | rope_start_pos="
            << attention.GetRopeStartPos()
            << "\n";
    }

    // ============================================================
    // COMPARE
    // ============================================================

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

        if (max_diff > global_max_diff) {
            global_max_diff = max_diff;
        }

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

    // ============================================================
    // RESULT
    // ============================================================

    std::cout << "\n";
    std::cout
        << "Global max diff: "
        << global_max_diff
        << "\n";

    if (global_max_diff < 1e-5f) {
        std::cout
            << "RESULT: Attention KV-cache PASS\n";
    } else {
        std::cout
            << "RESULT: Attention KV-cache FAIL\n";
    }

    attention.SetUseKVCache(false);
    attention.ResetCache();

    std::cout << "========================================\n";
}

void TestFullVsKVCache(
    LanguageModel& model,
    const std::vector<int>& tokens
) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << " FULL FORWARD vs KV-CACHE FORWARD\n";
    std::cout << "========================================\n";

    if (tokens.empty()) {
        std::cout << "❌ ERROR: empty token sequence\n";
        return;
    }

    // ============================================================
    // 1. FULL FORWARD
    //
    // Для каждого prefix полностью прогоняем последовательность.
    //
    // [7]
    // [7 4]
    // [7 4 11]
    // ...
    //
    // KV-cache здесь ПОЛНОСТЬЮ выключен.
    // ============================================================

    std::cout << "\n[1] FULL FORWARD\n";

    model.SetUseKVCache(false);

    std::vector<std::vector<float>> full_logits;

    for (size_t len = 1; len <= tokens.size(); len++) {
        model.ResetCache();

        auto input =
            std::make_shared<Tensor>(
                Tensor({1, len})
            );

        for (size_t i = 0; i < len; i++) {
            input->at({0, i}) =
                static_cast<float>(tokens[i]);
        }

        auto output = model.forward(input);

        size_t vocab_size =
            output->GetShape()[2];

        size_t last_pos = len - 1;

        std::vector<float> logits(vocab_size);

        for (size_t v = 0; v < vocab_size; v++) {
            logits[v] =
                output->at({0, last_pos, v});
        }

        full_logits.push_back(logits);
    }

    std::cout
        << "FULL forward completed for "
        << tokens.size()
        << " positions.\n";


    // ============================================================
    // 2. KV-CACHE FORWARD
    //
    // Здесь KV-cache ЯВНО включаем.
    //
    // forward(7)
    // forward(4)
    // forward(11)
    // ...
    //
    // Только первый token является начальным состоянием.
    // Каждый следующий token использует накопленный KV-cache.
    // ============================================================

    std::cout << "\n[2] KV-CACHE FORWARD\n";

    model.SetUseKVCache(true);
    model.ResetCache();

    std::vector<std::vector<float>> cache_logits;

    for (size_t i = 0; i < tokens.size(); i++) {
        auto input =
            std::make_shared<Tensor>(
                Tensor({1, 1})
            );

        input->at({0, 0}) =
            static_cast<float>(tokens[i]);

        auto output = model.forward(input);

        size_t vocab_size =
            output->GetShape()[2];

        std::vector<float> logits(vocab_size);

        for (size_t v = 0; v < vocab_size; v++) {
            logits[v] =
                output->at({0, 0, v});
        }

        cache_logits.push_back(logits);
    }

    std::cout
        << "KV-cache forward completed for "
        << tokens.size()
        << " positions.\n";


    // ============================================================
    // 3. COMPARE
    // ============================================================

    std::cout << "\n[3] COMPARISON\n\n";

    size_t vocab_size =
        full_logits[0].size();

    float global_max_diff = 0.0f;

    size_t global_position = 0;
    size_t global_token = 0;

    size_t different_predictions = 0;


    for (size_t pos = 0; pos < tokens.size(); pos++) {

        float max_diff = 0.0f;
        size_t max_diff_token = 0;

        // --------------------------------------------------------
        // Сравниваем все logits
        // --------------------------------------------------------

        for (size_t v = 0; v < vocab_size; v++) {

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


        // --------------------------------------------------------
        // Argmax FULL
        // --------------------------------------------------------

        size_t full_prediction = 0;

        for (size_t v = 1; v < vocab_size; v++) {
            if (full_logits[pos][v] >
                full_logits[pos][full_prediction]) {

                full_prediction = v;
            }
        }


        // --------------------------------------------------------
        // Argmax CACHE
        // --------------------------------------------------------

        size_t cache_prediction = 0;

        for (size_t v = 1; v < vocab_size; v++) {
            if (cache_logits[pos][v] >
                cache_logits[pos][cache_prediction]) {

                cache_prediction = v;
            }
        }


        bool same_prediction =
            full_prediction == cache_prediction;


        if (!same_prediction) {
            different_predictions++;
        }


        // --------------------------------------------------------
        // Вывод
        // --------------------------------------------------------

        std::cout
            << "Position " << pos
            << " | input=" << tokens[pos]
            << " | FULL=" << full_prediction
            << " | CACHE=" << cache_prediction
            << " | max_diff=" << max_diff;

        if (same_prediction &&
            max_diff < 1e-4f) {

            std::cout << "  ✅ OK";

        } else if (same_prediction) {

            std::cout << "  ⚠ PREDICTION OK, LOGITS DIFFER";

        } else {

            std::cout << "  ❌ DIFFERENT";
        }

        std::cout << "\n";


        // --------------------------------------------------------
        // Показываем конкретное расхождение
        // --------------------------------------------------------

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


    // ============================================================
    // 4. FINAL RESULT
    // ============================================================

    std::cout << "\n";
    std::cout << "----------------------------------------\n";

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


    std::cout << "\n";

    if (passed) {

        std::cout
            << "RESULT: FULL and KV-CACHE "
            << "are equivalent. ✅ PASS\n";

    } else {

        std::cout
            << "RESULT: KV-CACHE IS NOT EQUIVALENT! ❌ FAIL\n";
    }

    std::cout << "----------------------------------------\n";


    // ============================================================
    // 5. Возвращаем модель в обычный режим
    // ============================================================

    model.SetUseKVCache(false);
    model.ResetCache();

    std::cout << "KV-cache test finished.\n";
    std::cout << "========================================\n";
}


void TestPredictions(
    LanguageModel& model,
    const std::vector<int>& tokens
) {
    std::cout << "\n=== Prediction Test ===\n";

    for (size_t i = 0; i < tokens.size() - 1; i++) {
        model.ResetCache();

        auto input = std::make_shared<Tensor>(Tensor({1, i + 1}));

        for (size_t j = 0; j <= i; j++) {
            input->at({0, j}) = static_cast<float>(tokens[j]);
        }

        auto logits_ptr = model.forward(input);
        Tensor& logits = *logits_ptr;

        size_t vocab_size = logits.GetShape()[2];
        size_t pos = i;

        int predicted = 0;
        float best = logits.at({0, pos, 0});

        for (size_t j = 1; j < vocab_size; j++) {
            float value = logits.at({0, pos, j});

            if (value > best) {
                best = value;
                predicted = static_cast<int>(j);
            }
        }

        std::cout
            << "Step " << i
            << ": expected " << tokens[i + 1]
            << ", predicted " << predicted;

        if (predicted == tokens[i + 1]) {
            std::cout << "  ✅\n";
        } else {
            std::cout << "  ❌\n";
        }
    }

    model.ResetCache();
}

class CrossEntropyLoss {
private:
    Tensor logits_;
    size_t pos_;
    int target_;
    std::vector<float> softmax_output_;

public:
    // forward: принимает логиты [batch, seq_len, vocab_size], позицию pos и target
    Tensor forward(const Tensor& logits, size_t pos, int target) {
        logits_ = logits;
        pos_ = pos;
        target_ = target;

        size_t vocab_size = logits.GetShape()[2];

        // 1. Извлекаем последний логит (по позиции pos)
        Tensor last_logits({vocab_size});
        for (size_t i = 0; i < vocab_size; i++) {
            last_logits.at(i) = logits.at({0, pos, i});
        }

        const float* data = last_logits.Data();

        // 2. Численная стабильность: вычитаем максимум
        float max_val = data[0];
        for (size_t i = 1; i < vocab_size; i++) {
            if (data[i] > max_val) max_val = data[i];
        }

        // 3. exp(x - max) и сумма
        std::vector<float> exp_vals(vocab_size);
        float sum_exp = 0.0f;
        for (size_t i = 0; i < vocab_size; i++) {
            exp_vals[i] = std::exp(data[i] - max_val);
            sum_exp += exp_vals[i];
        }

        // 4. Softmax
        softmax_output_.resize(vocab_size);
        for (size_t i = 0; i < vocab_size; i++) {
            softmax_output_[i] = exp_vals[i] / sum_exp;
        }

        // 5. Loss = -log(softmax[target])
        float loss_value = -std::log(softmax_output_[target]);
        return Tensor({1, 1}, loss_value);
    }

    // backward: возвращает градиент для logits (3D)
    Tensor backward() {
        size_t vocab_size = logits_.GetShape()[2];

        // Градиент для last_logits (softmax - 1 для target)
        Tensor grad_last_logits({vocab_size});
        float* grad_data = grad_last_logits.Data();

        for (size_t i = 0; i < vocab_size; i++) {
            grad_data[i] = softmax_output_[i];
        }
        grad_data[target_] -= 1.0f;

        // Создаём градиент для всего logits (3D)
        Tensor grad_logits(logits_.GetShape());
        float* grad_logits_data = grad_logits.Data();

        // Заполняем нулями
        for (size_t i = 0; i < grad_logits.GetSize(); i++) {
            grad_logits_data[i] = 0.0f;
        }

        // Копируем градиент в нужную позицию (pos)
        for (size_t i = 0; i < vocab_size; i++) {
            grad_logits.at({0, pos_, i}) = grad_data[i];
        }

        return grad_logits;
    }
};

std::vector<int> PrepareBatch(const std::string& text, BPETokenizer& tokenizer) {
    std::vector<size_t> ids = tokenizer.Encode(text);
    std::vector<int> tokens;
    for (size_t id : ids) {
        tokens.push_back(static_cast<int>(id));
    }
    return tokens;
}

void PrintTokens(const std::vector<size_t>& tokens, BPETokenizer& tokenizer) {
    std::vector<size_t> ids;
    for (int t : tokens) ids.push_back(static_cast<size_t>(t));
    std::cout << tokenizer.Decode(ids) << std::endl;
}

// --- Главный тест ---

int main() {
    std::cout << "=== Training & Generation Test ===\n\n";

    using Clock = std::chrono::high_resolution_clock;

    // ============================================================
    // 1. Токенизатор
    // ============================================================

    BPETokenizer tokenizer;

    std::string corpus =
        "hello world hello world hello world";

    tokenizer.Train(corpus, 50);
    tokenizer.Save("vocab.txt");

    std::cout
        << "Vocabulary size: "
        << tokenizer.GetVocabSize()
        << "\n\n";


    // ============================================================
    // 2. Создаём модель
    // ============================================================

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
        hidden_dim
    );

    std::cout
        << "Model created\n\n";


    // ============================================================
    // 3. Подготовка данных
    // ============================================================

    std::vector<int> tokens =
        PrepareBatch(
            "hello world",
            tokenizer
        );

    std::cout << "Tokens: ";

    for (int t : tokens) {
        std::cout << t << " ";
    }

    std::cout << "\n\n";


    // ============================================================
    // 4. Training
    // ============================================================

    std::cout
        << "Training started...\n\n";

    CrossEntropyLoss loss_fn;

    model.SetUseKVCache(false);


    // ------------------------------------------------------------
    // Общая статистика
    // ------------------------------------------------------------

    double total_forward_ms = 0.0;
    double total_loss_ms = 0.0;
    double total_backward_ms = 0.0;
    double total_update_ms = 0.0;
    double total_step_ms = 0.0;

    size_t total_steps = 0;


    auto training_start =
        Clock::now();


    for (int epoch = 0; epoch < epochs; epoch++) {

        float total_loss = 0.0f;

        double epoch_forward_ms = 0.0;
        double epoch_loss_ms = 0.0;
        double epoch_backward_ms = 0.0;
        double epoch_update_ms = 0.0;
        double epoch_total_ms = 0.0;


        auto epoch_start =
            Clock::now();


        for (size_t i = 0;
             i < tokens.size() - 1;
             i++) {

            auto step_start =
                Clock::now();


            // ====================================================
            // Reset cache
            // ====================================================

            model.ResetCache();


            // ====================================================
            // Prepare prefix
            // ====================================================

            std::vector<int> prefix(
                tokens.begin(),
                tokens.begin() + i + 1
            );

            auto input =
                std::make_shared<Tensor>(
                    Tensor({1, prefix.size()})
                );

            for (size_t j = 0;
                 j < prefix.size();
                 j++) {

                input->at({0, j}) =
                    static_cast<float>(prefix[j]);
            }


            // ====================================================
            // FORWARD
            // ====================================================

            auto forward_start =
                Clock::now();

            auto logits_ptr =
                model.forward(input);

            auto forward_end =
                Clock::now();


            double forward_ms =
                std::chrono::duration<double, std::milli>(
                    forward_end - forward_start
                ).count();


            epoch_forward_ms += forward_ms;
            total_forward_ms += forward_ms;


            Tensor& logits =
                *logits_ptr;


            // ====================================================
            // LOSS
            // ====================================================

            auto loss_start =
                Clock::now();

            Tensor loss =
                loss_fn.forward(
                    logits,
                    prefix.size() - 1,
                    tokens[i + 1]
                );

            auto loss_end =
                Clock::now();


            double loss_ms =
                std::chrono::duration<double, std::milli>(
                    loss_end - loss_start
                ).count();


            epoch_loss_ms += loss_ms;
            total_loss_ms += loss_ms;


            total_loss +=
                loss.at(0);


            // ====================================================
            // BACKWARD
            // ====================================================

            auto backward_start =
                Clock::now();

            Tensor grad_logits =
                loss_fn.backward();

            logits.backward(
                grad_logits
            );

            auto backward_end =
                Clock::now();


            double backward_ms =
                std::chrono::duration<double, std::milli>(
                    backward_end - backward_start
                ).count();


            epoch_backward_ms += backward_ms;
            total_backward_ms += backward_ms;


            // ====================================================
            // UPDATE
            // ====================================================

            auto update_start =
                Clock::now();

            model.Update(
                learning_rate
            );

            model.ClearGrad();

            auto update_end =
                Clock::now();


            double update_ms =
                std::chrono::duration<double, std::milli>(
                    update_end - update_start
                ).count();


            epoch_update_ms += update_ms;
            total_update_ms += update_ms;


            // ====================================================
            // STEP TOTAL
            // ====================================================

            auto step_end =
                Clock::now();

            double step_ms =
                std::chrono::duration<double, std::milli>(
                    step_end - step_start
                ).count();


            epoch_total_ms += step_ms;
            total_step_ms += step_ms;

            total_steps++;
        }


        auto epoch_end =
            Clock::now();


        double real_epoch_ms =
            std::chrono::duration<double, std::milli>(
                epoch_end - epoch_start
            ).count();


        // ========================================================
        // Logging
        // ========================================================

        if (epoch % 100 == 0 ||
            epoch == epochs - 1) {

            double steps_per_second =
                (epoch_total_ms > 0.0)
                    ? (tokens.size() - 1) /
                      (epoch_total_ms / 1000.0)
                    : 0.0;

            std::cout
                << "Epoch "
                << std::setw(4)
                << epoch
                << " | Loss: "
                << total_loss /
                   static_cast<float>(tokens.size() - 1)
                << "\n";

            std::cout
                << "    Forward : "
                << std::fixed
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
                << "\n";

            std::cout
                << "\n";
        }
    }


    auto training_end =
        Clock::now();


    double training_ms =
        std::chrono::duration<double, std::milli>(
            training_end - training_start
        ).count();


    // ============================================================
    // 5. FINAL PROFILE
    // ============================================================

    std::cout << "\n";
    std::cout
        << "========================================\n";

    std::cout
        << "        TRAINING PROFILE\n";

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


    // ============================================================
    // Percentages
    // ============================================================

    if (total_step_ms > 0.0) {

        std::cout << "\n";

        std::cout
            << "Forward share : "
            << total_forward_ms /
               total_step_ms * 100.0
            << " %\n";

        std::cout
            << "Loss share    : "
            << total_loss_ms /
               total_step_ms * 100.0
            << " %\n";

        std::cout
            << "Backward share: "
            << total_backward_ms /
               total_step_ms * 100.0
            << " %\n";

        std::cout
            << "Update share  : "
            << total_update_ms /
               total_step_ms * 100.0
            << " %\n";
    }


    double steps_per_second =
        total_steps /
        (training_ms / 1000.0);


    double tokens_per_second =
        total_steps /
        (training_ms / 1000.0);


    std::cout << "\n";

    std::cout
        << "Steps/sec : "
        << steps_per_second
        << "\n";

    std::cout
        << "Tokens/sec: "
        << tokens_per_second
        << "\n";

    std::cout
        << "========================================\n";


    // ============================================================
    // 6. Prediction
    // ============================================================

    TestPredictions(
        model,
        tokens
    );


    // ============================================================
    // 7. KV-cache test
    // ============================================================

    std::vector<int> test_tokens = {
        7, 4, 11, 11, 14,
        22, 14, 17, 11, 3
    };

    TestFullVsKVCache(
        model,
        test_tokens
    );


    // ============================================================
    // 8. Generation
    // ============================================================

    std::cout
        << "\n=== Generation ===\n";

    model.ResetCache();

    size_t start_token =
        tokens[0];

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


    std::cout
        << "=== Test completed ===\n";


    return 0;
}