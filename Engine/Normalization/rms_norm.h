#pragma once

#include "../Tensor/tensor.h"
#include <memory>
#include <string>
#include "../Tensor/device.h"


class RMSNorm {
private:
    std::shared_ptr<Tensor> gamma_;

public:
    RMSNorm(size_t embed_dim, Device device = Device::CPU);

    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);

    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    Tensor& GetGamma();

    void Save(const std::string& path) const;
    void Load(const std::string& path);

    float GetGammaGradNorm() const;
    float GetGammaNorm() const;
};