#include "../include/buffer.hpp"
#include "../dawn/dawnUtils.hpp"

namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU

Buffer::Buffer(const wgpu::Device& device){
    device_ = device;
    usage_ = wgpu::BufferUsage::CopyDst;
    bufferMapped_.buffer = nullptr;
    bufferMapped_.dataLength = 0;
    bufferMapped_.data = nullptr;
}

// 3 different ways to set buffer data 
Buffer::Buffer(const wgpu::Device& device, const void* data, size_t size, wgpu::BufferUsage usage) {
    device_ = device;
    usage_ = usage | wgpu::BufferUsage::CopyDst;
    bufferMapped_ = CreateBufferMappedFromData(device_, data, size, usage_);
}

//steSubdata which is used in tfjs
// Buffer::Buffer(const wgpu::Device& device, const void* data, size_t size, wgpu::BufferUsage usage) {
//     device_ = device;
//     usage_ = usage | wgpu::BufferUsage::CopyDst;
//     buffer_ = CreateBufferFromData(device_, data, size, usage_);
// }

// async callback
// Buffer::Buffer(const wgpu::Device& device, const void* data, size_t size, wgpu::BufferUsage usage) {
//     device_ = device;
//     usage_ = usage | wgpu::BufferUsage::CopyDst;
//     wgpu::BufferDescriptor desc;
//     desc.size = size;
//     desc.usage =usage | wgpu::BufferUsage::CopyDst;
//     buffer_ = device.CreateBuffer(&desc);

//     BufferWriteAsync tmp;
//     void* mappedData = tmp.MapWriteAsyncAndWait(buffer_);
//     memcpy(mappedData, data, size);
//     tmp.UnmapBuffer(buffer_);
// }

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu
