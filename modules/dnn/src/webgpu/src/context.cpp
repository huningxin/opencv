#include "../../precomp.hpp"
#include "common.hpp"
#include "context.hpp"
#include<dawn/webgpu_cpp.h>
#include "../dawn/dawnUtils.hpp"
#include <dawn_native/DawnNative.h>
namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU

std::shared_ptr<Context> wCtx;
std::shared_ptr<wgpu::Device> wDevice;
std::shared_ptr<wgpu::Queue> wQueue;
cv::Mutex wContextMtx;

// internally used
void createContext()
{
    cv::AutoLock lock(wContextMtx);
    if (!wCtx && !wDevice)
    {
        wCtx.reset(new Context());
    }
}

bool isAvailable()
{
    try
    {
        createContext();
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_ERROR(NULL, 
        "Failed to init WebGPU-Dawn environment. " << e.what());
        return false;
    }

    return true;
}
Context::Context() {
    // create wgpu::Device
    wDevice = std::make_shared<wgpu::Device>(createCppDawnDevice());
    wQueue = std::make_shared<wgpu::Queue>(wDevice->GetDefaultQueue());
}
Context::~Context() {
    // how to release object
    wDevice->Release();
    wQueue->Release();
}

// #endif

}}}  // namespace cv::dnn::webgpu