#pragma once
#include "../Autograd/add_op.h"
#include "../Autograd/mul_op.h"
#include "../Tensor/tensor.h"
#include <memory>

class LinearLayer {
private:
    std::shared_ptr<Tensor> W_;
    std::shared_ptr<Tensor> b_;
    size_t input_size_;
    size_t  output_size_;

    std::shared_ptr<Tensor> saved_mult_;
    std::shared_ptr<Tensor> saved_added_;
public:
    LinearLayer(size_t in, size_t out);
    LinearLayer() = default;

    void ClearGrad();
    void Update(float lr);
    void ScaleGrad(float factor);

    // std::shared_ptr<Tensor> forward(const Tensor& x);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);

    void Save(const std::string& folder, const std::string& name) const;
    void Load(const std::string& folder, const std::string& name);

    Tensor& GetWeights();
    Tensor& GetBias();

    const Tensor& GetWeights() const;
    const Tensor& GetBias() const;

    void SetWeights(const Tensor& weights, const Tensor& bias) {
        *W_ = weights;
        *b_ = bias;
    }

    float GetWeightGradNorm() const;
    float GetBiasGradNorm() const;

    float GetWeightNorm() const;
    float GetBiasNorm() const;
};