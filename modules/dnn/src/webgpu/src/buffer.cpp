#include <string.h>
#include "../dawn/dawnUtils.hpp"
#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/buffer.hpp"

namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU

Buffer::Buffer(std::shared_ptr<wgpu::Device> device)
{
    device_ = device;
    usage_ = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst |
             wgpu::BufferUsage::CopySrc;
}

Buffer::Buffer(std::shared_ptr<wgpu::Device> device, 
               const void* data, size_t size, 
               wgpu::BufferUsage usage) 
{
    device_ = device;
    usage_ = usage;
    size_ = size;
    wgpu::BufferDescriptor descriptor = {};
    descriptor.size = size;
    descriptor.usage = usage;
    WGPU_CHECK_POINTER_RET_VOID(device_);
    buffer_ = device_->CreateBuffer(& descriptor);
    if(data) 
    {
        wQueue->WriteBuffer(buffer_, 0, data, size_);
    }
}

Buffer::Buffer(const void* data, size_t size,  
               wgpu::BufferUsage usage)
{
    createContext();
    device_ = wDevice;
    usage_ = usage;
    size_ = size;
    wgpu::BufferDescriptor descriptor = {};
    descriptor.size = size;
    descriptor.usage = usage;
    WGPU_CHECK_POINTER_RET_VOID(device_)
    buffer_ = device_->CreateBuffer(& descriptor);
    if(data) 
    {
        wQueue->WriteBuffer(buffer_, 0, data, size_);
    }
}

void Buffer::setBufferData(const void * data, size_t size)
{
    size_ = size;
    if(data) 
    {
        wQueue->WriteBuffer(buffer_, 0, data, size_);
    }
}

const void* Buffer::MapReadAsyncAndWait() 
{
    if(size_ == 0) CV_Error(Error::StsError, "GPU buffer is null");
    if(! gpuReadBuffer_)
    {
        wgpu::BufferDescriptor desc = {};
        desc.size = size_;
        desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
        gpuReadBuffer_ = device_->CreateBuffer(& desc);
    }
    wgpu::CommandEncoder encoder = device_->CreateCommandEncoder();
    encoder.CopyBufferToBuffer(buffer_, 0, 
                               gpuReadBuffer_, 0, size_);
    wgpu::CommandBuffer cmdBuffer = encoder.Finish();
    wQueue->Submit(1, &cmdBuffer);
    encoder.Release();
    cmdBuffer.Release();
    gpuReadBuffer_.MapAsync(wgpu::MapMode::Read, 0, size_, 
    [](WGPUBufferMapAsyncStatus status, void* userdata) {
        Buffer * buffer= static_cast<Buffer *>(userdata);
        buffer->mappedData = buffer->gpuReadBuffer_.GetConstMappedRange(0, buffer->size_);
    }, this);
    while(mappedData == nullptr) 
    {
#ifdef __EMSCRIPTEN__
        emscripten_sleep(10);
#else
        device_->Tick();
#endif
    }
    if(mappedData == nullptr) CV_Error(Error::StsError, "Buffer mapAsync failed");
    return mappedData;
}

#endif  // HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu
