#include "tensor.h"
#include "device.h"
#include <cuda_runtime.h>

#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <cstdlib>

namespace {

double add_grad_time = 0.0;
size_t add_grad_calls = 0;

void PrintAddGradProfile() {

    std::cout
        << "\n========================================\n"
        << "       Tensor::AddGrad PROFILE\n"
        << "========================================\n";

    std::cout
        << "Calls: "
        << add_grad_calls
        << "\n";

    std::cout
        << "Total: "
        << std::fixed
        << std::setprecision(3)
        << add_grad_time
        << " ms\n";

    if (add_grad_calls > 0) {

        std::cout
            << "Avg:   "
            << add_grad_time / add_grad_calls
            << " ms\n";
    }

    std::cout
        << "========================================\n";
}

struct AddGradProfileInitializer {

    AddGradProfileInitializer() {

        std::atexit(PrintAddGradProfile);
    }
};

AddGradProfileInitializer add_grad_profile_initializer;

}

void Tensor::Allocate() {

    if (size_ == 0) {
        data_ = nullptr;
        return;
    }

    if (device_ == Device::CPU) {

        data_ = new float[size_];

        return;
    }

    cudaError_t error = cudaMalloc(
        reinterpret_cast<void**>(&data_),
        size_ * sizeof(float)
    );

    if (error != cudaSuccess) {

        throw std::runtime_error(
            std::string("cudaMalloc failed: ") +
            cudaGetErrorString(error)
        );
    }
}

void Tensor::Free() {

    if (data_ == nullptr) {
        return;
    }

    if (device_ == Device::CPU) {

        delete[] data_;

    } else {

        cudaError_t error = cudaFree(data_);

        if (error != cudaSuccess) {

            std::cerr
                << "cudaFree failed: "
                << cudaGetErrorString(error)
                << "\n";
        }
    }

    data_ = nullptr;
}

size_t Tensor::ComputeIndex(const std::vector<size_t>& indexes) const {
    if (indexes.size() != rank_) {
        std::ostringstream oss;
        oss << "Number of indexes must match rank: got " << indexes.size()
            << " indexes for tensor of rank " << rank_ << " with shape [";
        for (size_t i = 0; i < shape_.size(); ++i) {
            oss << shape_[i];
            if (i + 1 < shape_.size()) oss << ", ";
        }
        oss << "] - indexes: [";
        for (size_t i = 0; i < indexes.size(); ++i) {
            oss << indexes[i];
            if (i + 1 < indexes.size()) oss << ", ";
        }
        oss << "]";
        throw std::runtime_error(oss.str());
    }
    
    size_t index = 0;
    size_t step = 1;
    
    for (size_t i = rank_; i-- > 0;) {
        if (indexes[i] >= shape_[i]) {
            throw std::runtime_error("Index out of bounds");
        }

        index += indexes[i] * step;
        step *= shape_[i];
    }
    
    return index;
}

Tensor::Tensor(std::vector<size_t> shape)
    : shape_(std::move(shape)) {

    size_ = std::accumulate(
        shape_.begin(),
        shape_.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>()
    );

    rank_ = shape_.size();
    Allocate();
}

Tensor::Tensor(std::vector<size_t> shape, float k)
    : shape_(std::move(shape)) {

    size_ = std::accumulate(
        shape_.begin(),
        shape_.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>()
    );

    rank_ = shape_.size();

    Allocate();

    std::fill(
        data_,
        data_ + size_,
        k
    );
}

Tensor::Tensor(std::vector<size_t> shape, UninitializedTag)
    : shape_(std::move(shape)) {

    size_ = std::accumulate(
        shape_.begin(),
        shape_.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>()
    );

    rank_ = shape_.size();
    Allocate();
}

Tensor Tensor::Random(std::vector<size_t> shape, float min, float max, Device device) {
    Tensor result(shape, UninitializedTag{});

    static std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(min, max);

    std::vector<float> data(result.size_);

    for (size_t i = 0; i < result.size_; i++) {
        data[i] = dist(gen);
    }

    if (device == Device::CPU) {
        for (size_t i = 0; i < result.size_; i++) {
            result.data_[i] = data[i];
        }
    } else {
        cudaMemcpy(
            result.data_,
            data.data(),
            result.size_ * sizeof(float),
            cudaMemcpyHostToDevice
        );
    }

    return result;
}

Tensor::Tensor(
    std::vector<size_t> shape,
    std::vector<float> data
)
    : shape_(std::move(shape)) {

    rank_ = shape_.size();

    size_ = std::accumulate(
        shape_.begin(),
        shape_.end(),
        static_cast<size_t>(1),
        std::multiplies<size_t>()
    );

    Allocate();

    size_t copy_size = std::min(
        data.size(),
        size_
    );

    std::copy(
        data.begin(),
        data.begin() + copy_size,
        data_
    );
}

Tensor::Tensor(std::vector<size_t> shape, float k, Device device)
        : shape_(std::move(shape)), device_(device) {
            
    rank_ = shape_.size();
    size_ = 1;
    for (size_t dim : shape_) {
        size_ *= dim;
    }

    Allocate();

    if (size_ == 0) {
        return;
    }

    if (device_ == Device::CPU) {
        for (size_t i = 0; i < size_; ++i) {
            data_[i] = k;
        }
        return;
    }

    if (k == 0.0f) {
        cudaError_t error = cudaMemset(
            data_,
            0,
            size_ * sizeof(float)
        );

        if (error != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMemset failed: ") +
                cudaGetErrorString(error)
            );
        }

        return;
    }

    std::vector<float> host_data(size_, k);

    cudaError_t error = cudaMemcpy(
        data_,
        host_data.data(),
        size_ * sizeof(float),
        cudaMemcpyHostToDevice
    );

    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA scalar initialization failed: ") +
            cudaGetErrorString(error)
        );
    }
}

Tensor::~Tensor() {
    Free();
}

Tensor::Tensor(const Tensor& other)
    : shape_(other.shape_),
      size_(other.size_),
      rank_(other.rank_),
      device_(other.device_) {

    Allocate();

    if (size_ == 0) {
        return;
    }

    if (device_ == Device::CPU) {

        std::copy(
            other.data_,
            other.data_ + size_,
            data_
        );

    } else {

        cudaError_t error = cudaMemcpy(
            data_,
            other.data_,
            size_ * sizeof(float),
            cudaMemcpyDeviceToDevice
        );

        if (error != cudaSuccess) {

            throw std::runtime_error(
                std::string(
                    "CUDA copy constructor failed: "
                ) +
                cudaGetErrorString(error)
            );
        }
    }
}

Tensor::Tensor(
    std::vector<size_t> shape,
    Device device
)
    : shape_(std::move(shape)),
      device_(device) {
    rank_ = shape_.size();

    size_ = 1;
    for (size_t dim : shape_) {
        size_ *= dim;
    }

    Allocate();
}

Tensor::Tensor(Tensor&& other) noexcept
    : shape_(std::move(other.shape_)),
      data_(other.data_),
      size_(other.size_),
      rank_(other.rank_),
      grad_(std::move(other.grad_)),
      grad_fn_(std::move(other.grad_fn_)),
      device_(other.device_) {

    other.data_ = nullptr;
    other.size_ = 0;
    other.rank_ = 0;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this == &other) return *this;

    Free();

    shape_ = std::move(other.shape_);
    data_ = other.data_;
    size_ = other.size_;
    rank_ = other.rank_;
    grad_ = std::move(other.grad_);
    grad_fn_ = std::move(other.grad_fn_);
    device_ = other.device_;

    other.data_ = nullptr;
    other.size_ = 0;
    other.rank_ = 0;
    other.shape_.clear();
    other.grad_ = nullptr;
    other.grad_fn_ = nullptr;

    return *this;
}

Tensor& Tensor::operator=(const Tensor& other) {

    if (this == &other) {
        return *this;
    }

    Free();

    shape_ = other.shape_;
    size_ = other.size_;
    rank_ = other.rank_;
    device_ = other.device_;

    grad_ = nullptr;
    grad_fn_ = nullptr;

    Allocate();

    if (size_ == 0) {
        return *this;
    }

    if (device_ == Device::CPU) {

        std::copy(
            other.data_,
            other.data_ + size_,
            data_
        );

    } else {

        cudaError_t error = cudaMemcpy(
            data_,
            other.data_,
            size_ * sizeof(float),
            cudaMemcpyDeviceToDevice
        );

        if (error != cudaSuccess) {

            throw std::runtime_error(
                std::string(
                    "CUDA copy assignment failed: "
                ) +
                cudaGetErrorString(error)
            );
        }
    }

    return *this;
}

float& Tensor::at(size_t index) {
    return data_[index];
}
const float& Tensor::at(size_t index) const {
    return data_[index];
}

float& Tensor::at(const std::vector<size_t>& indexes) {
    return data_[ComputeIndex(indexes)];
}

const float& Tensor::at(const std::vector<size_t>& indexes) const {
    return data_[ComputeIndex(indexes)];
}

void Tensor::Set(std::vector<size_t> indexes, float value) {
    data_[ComputeIndex(indexes)] = value;
}

auto Tensor::GetIter(const std::vector<size_t>& indexes) {
    return data_ + ComputeIndex(indexes);
}

const auto Tensor::GetIter(const std::vector<size_t>& indexes) const {
    return data_ + ComputeIndex(indexes);
}

Tensor Tensor::Reshape(std::vector<size_t> new_shape) const {
    size_t first_size = std::accumulate(shape_.begin(), 
        shape_.end(), 1, std::multiplies<size_t>());
    size_t second_size = std::accumulate(new_shape.begin(), 
        new_shape.end(), 1, std::multiplies<size_t>());
    if (first_size != second_size) {
        throw std::runtime_error("This reshape is not possivle");
    }
    Tensor tensor(*this);
    tensor.shape_ = new_shape;
    tensor.rank_ = tensor.shape_.size();
    return tensor;
}

size_t Tensor::GetSize() const {
    return size_;
}

size_t Tensor::GetRank() const {
    return rank_;
}

const std::vector<size_t>& Tensor::GetShape() const {
    return shape_;
}

float* Tensor::Data() {
    return data_;
}

const float* Tensor::Data() const {
    return data_;
}

std::shared_ptr<Tensor> Tensor::Grad() const {
    return grad_;
}

void Tensor::SetGrad(std::shared_ptr<Tensor> grad) {
        grad_ = grad;
}

void Tensor::AddGrad(Tensor grad) {

    if (grad_ == nullptr) {

        grad_ =
            std::make_shared<Tensor>(
                std::move(grad)
            );

        return;
    }

    if (grad_->GetShape() != grad.GetShape()) {
        throw std::runtime_error(
            "Tensor::AddGrad: gradient shape mismatch"
        );
    }

    auto start =
        std::chrono::steady_clock::now();

    *grad_ += grad;

    auto end =
        std::chrono::steady_clock::now();

    add_grad_time +=
        std::chrono::duration<double, std::milli>(
            end - start
        ).count();

    add_grad_calls++;
}

void Tensor::ClearGrad() {
    grad_ = nullptr;
}

std::shared_ptr<Operation> Tensor::GradFn() const {
    return grad_fn_;
}

void Tensor::SetGradFn(std::shared_ptr<Operation> op) {
    if (op != nullptr) {
        grad_fn_ = op;
    }
}

void Tensor::backward(const Tensor& grad_output) {
    if (grad_fn_ == nullptr) {

        throw std::runtime_error(
            "Cannot call backward on tensor without grad_fn"
        );
    }

    std::vector<Tensor*> graph;
    std::unordered_set<Tensor*> visited;

    BuildBackwardGraph(
        graph,
        visited
    );

    AddGrad(grad_output);


    size_t operation_index = 0;

    for (auto it = graph.rbegin();
         it != graph.rend();
         ++it) {

        Tensor* tensor = *it;

        if (tensor == nullptr) {
            continue;
        }

        if (tensor->grad_fn_ == nullptr) {
            continue;
        }

        operation_index++;

        const char* operation_name =
            tensor->grad_fn_->Name();

        if (tensor->grad_ == nullptr) {
            throw std::runtime_error(
                std::string(
                    "Tensor::backward: missing gradient for operation "
                ) + operation_name
            );
        }

        std::vector<std::shared_ptr<Tensor>> inputs =
            tensor->grad_fn_->GetInputs();

        for (size_t i = 0;
             i < inputs.size();
             ++i) {

            if (inputs[i] == nullptr) {
                continue;
            }
        }

        std::vector<Tensor> gradients =
            tensor->grad_fn_->backward(
                *tensor->grad_
            );

        if (gradients.size() != inputs.size()) {

            throw std::runtime_error(
                "Number of gradients does not match "
                "number of inputs"
            );
        }

        for (size_t i = 0;
             i < inputs.size();
             ++i) {

            if (inputs[i] == nullptr) {
                continue;
            }

            inputs[i]->AddGrad(
                gradients[i]
            );
        }
    }
}

Tensor Tensor::SumAxis(int axis) const {
    return GetBackend(device_).SumAxis(*this, axis);
}

Tensor Tensor::Mean(int axis) const {
    return GetBackend(device_).Mean(*this, axis);
}

void Tensor::SaveTensor(const std::string& path) const {
    std::ofstream file(path + ".bin", std::ios::binary);

    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path + ".bin");
    }
    
    for (size_t i = 0; i < shape_.size(); i++) {
        file << shape_[i];
        if (i + 1 < shape_.size()) file << " ";
    }
    file << "\n";

    file << std::setprecision(10);
    for (size_t i = 0; i < size_; i++) {
        file << data_[i];
        if (i + 1 < size_) file << " ";
    }
    file << "\n";
}

Tensor Tensor::LoadTensor(const std::string& path) {
    std::ifstream file(path + ".bin");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + path + ".bin");
    }

    std::vector<size_t> shape;
    size_t dim;
    while (file.peek() != '\n' && file >> dim) {
        shape.push_back(dim);
    }
    file.ignore();

    std::vector<float> data;
    float value;
    while (file >> value) {
        data.push_back(value);
    }

    return Tensor(shape, data);
}

void Tensor::BuildBackwardGraph(std::vector<Tensor*>& graph, std::unordered_set<Tensor*>& visited) {
    if (visited.find(this) != visited.end()) { return; }
    visited.insert(this);

    if (grad_fn_ != nullptr) {
        std::vector<std::shared_ptr<Tensor>> inputs = grad_fn_->GetInputs();
        for (const auto& input : inputs) {
            if (input != nullptr) {
                input->BuildBackwardGraph(graph, visited);
            }
        }
    }

    graph.push_back(this);
}

size_t Tensor::Numel() const {
    size_t result = 1;

    for (size_t dim : shape_) {
        result *= dim;
    }

    return result;
}
    
Device Tensor::GetDevice() const {
    return device_;
}