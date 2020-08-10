#include <cstring>
#include <memory>
#include "dawnUtils.hpp"
#include "opencv2/core/base.hpp"
namespace cv { namespace dnn { namespace webgpu {
#ifdef __EMSCRIPTEN__
wgpu::Device createCppDawnDevice() {
    return wgpu::Device::Acquire(emscripten_webgpu_get_device());
}
#else
#ifdef HAVE_WEBGPU

static std::shared_ptr<dawn_native::Instance> instance;
static wgpu::BackendType backendType = wgpu::BackendType::Vulkan;

void PrintDeviceError(WGPUErrorType errorType, const char* message, void*) {
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
    errorTypeName += "Error message: ";
    errorTypeName += message;
    CV_Error(Error::StsError, errorTypeName);
}

wgpu::Device createCppDawnDevice() {
    instance = std::make_shared<dawn_native::Instance>();
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

#endif  // HAVE_WEBGPU
#endif  //__EMSCRIPTEN__
}}}  //namespace cv::dnn::webgpu