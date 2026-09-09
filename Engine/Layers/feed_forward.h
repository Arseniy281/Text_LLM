#pragma once
#include "../Layers/linear_layer.h"
#include "../Autograd/gelu.h"
#include "../Tensor/tensor.h"
#include "../Tensor/device.h"

#include <string>

class FeedForward {
private:
    LinearLayer fc1_;
    LinearLayer fc2_;
public:
    FeedForward(size_t embed_dim, size_t hidden_dim, Device device = Device::CPU);
    // Tensor forward(const Tensor& x);
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x);
    void Update(float lr);
    void ClearGrad();
    void ScaleGrad(float factor);

    void Save(const std::string& folder) const;
    void Load(const std::string& folder);

    LinearLayer& GetFC1() {
        return fc1_;
    }

    LinearLayer& GetFC2() {
        return fc2_;
    }

    const LinearLayer& GetFC1() const {
        return fc1_;
    }

    const LinearLayer& GetFC2() const {
        return fc2_;
    }
};