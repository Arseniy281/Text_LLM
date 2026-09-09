#include "../Autograd/add_op.h"
#include "../Autograd/mul_op.h"
#include "../Tensor/tensor.h"
#include "linear_layer.h"

#include <cmath>
#include <memory>

LinearLayer::LinearLayer(size_t in, size_t out) : input_size_(in),output_size_(out) {
    float limit = std::sqrt(6.0f / static_cast<float>(in + out));

    W_ = std::make_shared<Tensor>(Tensor::Random({in, out}, -limit, limit));
    b_ = std::make_shared<Tensor>(Tensor({1, out}, 0.0f));
}

void LinearLayer::ClearGrad() {
    W_->ClearGrad();
    b_->ClearGrad();
}

void LinearLayer::Update(float lr) {
    if (W_->Grad() != nullptr) {
        Tensor update =
            GetBackend(W_->GetDevice()).MulScalar(*W_->Grad(), lr);

        Tensor new_weights = GetBackend(W_->GetDevice()).Sub(*W_, update);
        *W_ = std::move(new_weights);
    }

    if (b_->Grad() != nullptr) {
        Tensor update = GetBackend(b_->GetDevice()).MulScalar(
            *b_->Grad(), lr);

        Tensor new_bias = GetBackend(b_->GetDevice()).Sub(*b_, update);
        *b_ = std::move(new_bias);
    }
}

void LinearLayer::ScaleGrad(float factor) {
    if (W_->Grad() != nullptr) {
        Tensor scaled =GetBackend(
            W_->Grad()->GetDevice()
        ).MulScalar(*W_->Grad(), factor);

        *W_->Grad() = std::move(scaled);
    }

    if (b_->Grad() != nullptr) {
        Tensor scaled = GetBackend(b_->Grad()->GetDevice()
            ).MulScalar(*b_->Grad(), factor);
        *b_->Grad() = std::move(scaled);
    }
}

std::shared_ptr<Tensor> LinearLayer::forward(const std::shared_ptr<Tensor>& x) {
    auto mul_op = std::make_shared<MulOp>();
    saved_mult_ = mul_op->forward({x, W_});
    auto add_op = std::make_shared<AddOp>();
    saved_added_ = add_op->forward({saved_mult_, b_});

    return saved_added_;
}

void LinearLayer::Save(const std::string& folder, const std::string& name) const {
    W_->SaveTensor(folder + "/" + name + "_W");
    b_->SaveTensor(folder + "/" + name + "_b");
}

void LinearLayer::Load(const std::string& folder, const std::string& name) {
    *W_ = Tensor::LoadTensor(folder + "/" + name + "_W");
    *b_ = Tensor::LoadTensor(folder + "/" + name + "_b");
}

Tensor& LinearLayer::GetWeights() {
    return *W_;
}

Tensor& LinearLayer::GetBias() {
    return *b_;
}

const Tensor& LinearLayer::GetWeights() const {
    return *W_;
}

const Tensor& LinearLayer::GetBias() const {
    return *b_;
}

float LinearLayer::GetWeightGradNorm() const {
    if (W_->Grad() == nullptr) { return 0.0f; }
    Tensor squared = GetBackend(W_->Grad()->GetDevice()).Mul(*W_->Grad(), *W_->Grad());

    Tensor sum = GetBackend(squared.GetDevice()).Sum(squared);
    return std::sqrt(sum.at(0));
}

float LinearLayer::GetBiasGradNorm() const {
    if (b_->Grad() == nullptr) { return 0.0f; }

    Tensor squared = GetBackend(b_->Grad()->GetDevice()).Mul(*b_->Grad(), *b_->Grad());
    Tensor sum = GetBackend(squared.GetDevice()).Sum(squared);

    return std::sqrt(sum.at(0));
}

float LinearLayer::GetWeightNorm() const {
    Tensor squared = GetBackend(W_->GetDevice()).Mul(*W_, *W_);
    Tensor sum = GetBackend(squared.GetDevice()).Sum(squared);

    return std::sqrt(sum.at(0));
}

float LinearLayer::GetBiasNorm() const {
    Tensor squared = GetBackend(b_->GetDevice()).Mul(*b_, *b_);
    Tensor sum = GetBackend(squared.GetDevice()).Sum(squared);

    return std::sqrt(sum.at(0));
}