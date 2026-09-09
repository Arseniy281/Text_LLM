#include "backend.h"
#include "cpu_backend.h"

Backend& GetBackend(Device device) {
    switch (device) {
        case Device::CPU:
            static CPUBackend cpu_backend;
            return cpu_backend;

        case Device::CUDA:
            throw std::runtime_error(
                "CUDA backend is not available"
            );
    }

    throw std::runtime_error(
        "Unknown device"
    );
}