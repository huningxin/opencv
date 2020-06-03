
#include <algorithm>
#include <cstring>
#include <dawn/dawn_proc.h>
#include <dawn/dawn_wsi.h>
#include <dawn_native/DawnNative.h>
#include <dawn_wire/WireClient.h>
#include <dawn_wire/WireServer.h>
#include<dawn/webgpu_cpp.h>
#include "opencv2/core/base.hpp"
namespace cv { namespace dnn { namespace webgpu {
static std::unique_ptr<dawn_native::Instance> instance;
static wgpu::BackendType backendType = wgpu::BackendType::Vulkan;

void PrintDeviceError(WGPUErrorType errorType, const char* message, void*) {
    const char* errorTypeName = "";
    switch (errorType) {
        case WGPUErrorType_Validation:
            errorTypeName = "Validation";
            break;
        case WGPUErrorType_OutOfMemory:
            errorTypeName = "Out of memory";
            break;
        case WGPUErrorType_Unknown:
            errorTypeName = "Unknown";
            break;
        case WGPUErrorType_DeviceLost:
            errorTypeName = "Device lost";
            break;
        default: WGPUErrorType_Unknown:
            errorTypeName = "Unknown";
            return;
    }
    CV_Error(Error::StsError, errorTypeName);
}

wgpu::Device createCppDawnDevice() {
    instance = std::make_unique<dawn_native::Instance>();
    instance->DiscoverDefaultAdapters();
    // Get an adapter for the backend to use, and create the device.
    dawn_native::Adapter backendAdapter;
    {
        std::vector<dawn_native::Adapter> adapters = instance->GetAdapters();
        auto adapterIt = std::find_if(adapters.begin(), adapters.end(),
                                      [](const dawn_native::Adapter adapter) -> bool {
                                          wgpu::AdapterProperties properties;
                                          adapter.GetProperties(&properties);
                                          return properties.backendType == backendType;
                                      });
        backendAdapter = *adapterIt;
    }
    WGPUDevice backendDevice = backendAdapter.CreateDevice();
    DawnProcTable backendProcs = dawn_native::GetProcs();
    WGPUDevice cDevice = nullptr;
    DawnProcTable procs;
    procs = backendProcs;
    cDevice = backendDevice;
    dawnProcSetProcs(&procs);
    procs.deviceSetUncapturedErrorCallback(cDevice, PrintDeviceError, nullptr);
    return wgpu::Device::Acquire(cDevice);
}


}}}