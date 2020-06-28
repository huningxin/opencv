#include <algorithm>
// #ifdef HAVE_WEBGPU
#include <dawn/dawn_proc.h>
#include <dawn/dawn_wsi.h>
#include <dawn_native/DawnNative.h>
#include <dawn/webgpu_cpp.h>
#include "opencv2/core/base.hpp"
#include "dawnUtils.hpp"
// #endif
#include <cstring>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <memory>
#include <iostream>
namespace cv { namespace dnn { namespace webgpu {

// #ifdef HAVE_WEBGPU

static dawn_native::Instance* instance;
static wgpu::BackendType backendType = wgpu::BackendType::Vulkan;

void PrintDeviceError(WGPUErrorType errorType,  const char* message, void*) {
    String errorTypeName = "";
    switch (errorType) {
        case WGPUErrorType_Validation:
            errorTypeName = "WGPUErrorTyp Validation";
            break;
        case WGPUErrorType_OutOfMemory:
            errorTypeName = "WGPUErrorTyp Out of memory";
            break;
        case WGPUErrorType_Unknown:
            errorTypeName = "WGPUErrorTyp Unknown";
            break;
        case WGPUErrorType_DeviceLost:
            errorTypeName = "WGPUErrorTyp Device lost";
            break;
        default:
            errorTypeName = "WGPUErrorTyp Unknown";
            return;
    }
    errorTypeName += message;
    CV_Error(Error::StsError, errorTypeName);
}

wgpu::Device createCppDawnDevice() {
    instance = new dawn_native::Instance();
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
    dawnProcSetProcs(&backendProcs);
    backendProcs.deviceSetUncapturedErrorCallback(backendDevice, PrintDeviceError, nullptr);
    return wgpu::Device::Acquire(backendDevice);
}

wgpu::Buffer CreateBufferFromData(const wgpu::Device& device,
                                    const void* data,
                                    size_t size,
                                    wgpu::BufferUsage usage) {
    wgpu::BufferDescriptor descriptor;
    descriptor.size = size;
    descriptor.usage = usage | wgpu::BufferUsage::CopyDst;

    wgpu::Buffer buffer = device.CreateBuffer(&descriptor);
    buffer.SetSubData(0, size, data);
    return buffer;
}

wgpu::CreateBufferMappedResult CreateBufferMappedFromData(const wgpu::Device& device,
                                                        const void* data,
                                                        size_t size,
                                                        wgpu::BufferUsage usage){
    wgpu::BufferDescriptor descriptor = {};
    descriptor.size = size;
    descriptor.usage = usage | wgpu::BufferUsage::CopyDst;
    wgpu::CreateBufferMappedResult result = device.CreateBufferMapped(&descriptor);
    if(data) { memcpy(result.data, data, size); }
    return result;
}

wgpu::PipelineLayout MakeBasicPipelineLayout(const wgpu::Device& device,
                                             const wgpu::BindGroupLayout* bindGroupLayout) {
    wgpu::PipelineLayoutDescriptor descriptor;
    if (bindGroupLayout != nullptr) {
        descriptor.bindGroupLayoutCount = 1;
        descriptor.bindGroupLayouts = bindGroupLayout;
    } else {
        descriptor.bindGroupLayoutCount = 0;
        descriptor.bindGroupLayouts = nullptr;
    }
    return device.CreatePipelineLayout(&descriptor);
}

wgpu::BindGroupLayout MakeBindGroupLayout(
    const wgpu::Device& device,
    std::vector<wgpu::BindGroupLayoutEntry> entriesInitializer) {
    wgpu::BindGroupLayoutDescriptor descriptor;
    descriptor.entryCount = static_cast<uint32_t>(entriesInitializer.size());
    descriptor.entries = entriesInitializer.data();
    return device.CreateBindGroupLayout(&descriptor);
}

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu