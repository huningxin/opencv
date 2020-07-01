#include "../include/buffer.hpp"
#include "../dawn/dawnUtils.hpp"
#include "common.hpp"
#include <string.h>
namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU

Buffer::Buffer(std::shared_ptr<wgpu::Device> device)
{
    device_ =  device;
    usage_ = wgpu::BufferUsage::Storage;
    bufferMapped_->buffer = nullptr;
    bufferMapped_->dataLength = 0;
    bufferMapped_->data = nullptr;
}

Buffer::Buffer( std::shared_ptr<wgpu::Device> device, 
                const void* data, size_t size, 
                wgpu::BufferUsage usage) 
{
    device_ = device;
    usage_ = usage;
    size_ = size;
    wgpu::BufferDescriptor descriptor = {};
    descriptor.size = size;
    descriptor.usage = usage;
    bufferMapped_ = std::make_shared<wgpu::CreateBufferMappedResult>
                    (device_->CreateBufferMapped(&descriptor));
    if(data) { memcpy(bufferMapped_->data, data, size); }
}

const void* Buffer::MapReadAsyncAndWait() 
{
    wgpu::BufferDescriptor desc = {};
    desc.size = size_;
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer gpuReadBuffer = device_->CreateBuffer(& desc);
    wgpu::CommandEncoder encoder = device_->CreateCommandEncoder();
    encoder.CopyBufferToBuffer( bufferMapped_->buffer, 0, 
                                gpuReadBuffer, 0, size_);
    wgpu::CommandBuffer cmdBuffer = encoder.Finish();
    wQueue->Submit(1, &cmdBuffer);
    gpuReadBuffer.MapReadAsync(BufferMapReadCallback, this);
    usleep(100);
    while(mappedData == nullptr) 
    {
        device_->Tick();
        usleep(100);
    }
    return mappedData;
}

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu
