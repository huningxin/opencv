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
    buffer_ = device_->CreateBuffer(& descriptor);
    if(data) buffer_.SetSubData(0, size_, data);
}

void Buffer::setBufferData(const void * data, size_t size)
{
    size_ = size;
    buffer_.SetSubData(0, size_, data);
}

const void* Buffer::MapReadAsyncAndWait() 
{
    wgpu::BufferDescriptor desc = {};
    desc.size = size_;
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer gpuReadBuffer = device_->CreateBuffer(& desc);
    wgpu::CommandEncoder encoder = device_->CreateCommandEncoder();
    encoder.CopyBufferToBuffer( buffer_, 0, 
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
    gpuReadBuffer.Release(); 
    return mappedData;
}

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu
