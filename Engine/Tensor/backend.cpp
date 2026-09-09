#include "backend.h"
#include "cpu_backend.h"
#include "cuda_backend.h"

Backend& GetBackend(Device device) {
    switch (device) {
        case Device::CPU: {
            static CPUBackend cpu_backend;
            return cpu_backend;
        }

        case Device::CUDA: {
            static CUDABackend cuda_backend;
            return cuda_backend;
        }
    }

    throw std::runtime_error(
        "Unknown device"
    );
}