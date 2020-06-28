#include "../include/buffer.hpp"
#include "../dawn/dawnUtils.hpp"
#include <string.h>
namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU

Buffer::Buffer(std::shared_ptr<wgpu::Device> device){
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
    wgpu::BufferDescriptor descriptor = {};
    descriptor.size = size;
    descriptor.usage = usage;
    bufferMapped_ = std::make_shared<wgpu::CreateBufferMappedResult>
                    (device_->CreateBufferMapped(&descriptor));
    if(data) { memcpy(bufferMapped_->data, data, size); }
}

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu
