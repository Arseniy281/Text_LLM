#include "rms_norm.h"
#include "../Autograd/rms_norm_op.h"
#include "../Tensor/device.h"


RMSNorm::RMSNorm(size_t embed_dim, Device device)
    : gamma_(std::make_shared<Tensor>(
        std::vector<size_t>{embed_dim},
        1.0f,
        device
    )) {}

std::shared_ptr<Tensor> RMSNorm::forward(
    const std::shared_ptr<Tensor>& x
) {
    auto operation = std::make_shared<RMSNormOp>(gamma_);
    return operation->forward({x});
}

void RMSNorm::Update(float lr) {
    if (gamma_->Grad() == nullptr) { return; }

    Tensor update =
        GetBackend(gamma_->GetDevice()).MulScalar(*gamma_->Grad(), lr);

    Tensor new_gamma =
        GetBackend(gamma_->GetDevice()).Sub(*gamma_, update);

    *gamma_ = std::move(new_gamma);
}

void RMSNorm::ClearGrad() {
    gamma_->ClearGrad();
}

void RMSNorm::ScaleGrad(float factor) {
    if (gamma_->Grad() == nullptr) { return; }

    Tensor scaled =
        GetBackend(gamma_->Grad()->GetDevice()).MulScalar(*gamma_->Grad(), factor);

    *gamma_->Grad() = std::move(scaled);
}

Tensor& RMSNorm::GetGamma() {
    return *gamma_;
}

void RMSNorm::Save(const std::string& path) const {
    gamma_->SaveTensor(path);
}

void RMSNorm::Load(const std::string& path) {
    *gamma_ = Tensor::LoadTensor(path);
}

float RMSNorm::GetGammaGradNorm() const {
    if (gamma_->Grad() == nullptr) {
        return 0.0f;
    }

    float sum = 0.0f;

    for (size_t i = 0; i < gamma_->Grad()->GetSize(); i++) {
        float x = gamma_->Grad()->at(i);
        sum += x * x;
    }

    return std::sqrt(sum);
}

float RMSNorm::GetGammaNorm() const {
    float sum = 0.0f;

    for (size_t i = 0; i < gamma_->GetSize(); i++) {
        float x = gamma_->at(i);
        sum += x * x;
    }

    return std::sqrt(sum);
}