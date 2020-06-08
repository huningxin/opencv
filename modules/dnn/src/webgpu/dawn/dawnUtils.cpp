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
namespace cv { namespace dnn { namespace webgpu {

// #ifdef HAVE_WEBGPU

static std::unique_ptr<dawn_native::Instance> instance;
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
        default: WGPUErrorType_Unknown:
            errorTypeName = "WGPUErrorTyp Unknown";
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
    memcpy(result.data, data, size);
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

wgpu::BindGroup MakeBindGroup(
    const wgpu::Device& device,
    const wgpu::BindGroupLayout& layout,
    std::vector<BindingInitializationHelper> entriesInitializer) {
    std::vector<wgpu::BindGroupEntry> entries;
    for (const BindingInitializationHelper& helper : entriesInitializer) {
        entries.push_back(helper.GetAsBinding());
    }

    wgpu::BindGroupDescriptor descriptor;
    descriptor.layout = layout;
    descriptor.entryCount = entries.size();
    descriptor.entries = entries.data();

    return device.CreateBindGroup(&descriptor);
}

BindingInitializationHelper::BindingInitializationHelper(uint32_t binding,
                                                            const wgpu::Sampler& sampler)
    : binding(binding), sampler(sampler) {}

BindingInitializationHelper::BindingInitializationHelper(uint32_t binding,
                                                            const wgpu::TextureView& textureView)
    : binding(binding), textureView(textureView) {
}

BindingInitializationHelper::BindingInitializationHelper(uint32_t binding,
                                                            const wgpu::Buffer& buffer,
                                                            uint64_t offset,
                                                            uint64_t size)
    : binding(binding), buffer(buffer), offset(offset), size(size) {
}

wgpu::BindGroupEntry BindingInitializationHelper::GetAsBinding() const {
    wgpu::BindGroupEntry result;

    result.binding = binding;
    result.sampler = sampler;
    result.textureView = textureView;
    result.buffer = buffer;
    result.offset = offset;
    result.size = size;

    return result;
}

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu