#include "../include/buffer.hpp"
#include "../dawn/dawnUtils.h"

namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU

Buffer::Buffer(const wgpu::Device& device){
    device_ = device;
    usage_ = wgpu::BufferUsage::CopyDst;
    bufferMapped_.buffer = nullptr;
    bufferMapped_.dataLength = 0;
    bufferMapped_.data = nullptr;
}

Buffer::Buffer(const wgpu::Device& device, const char* data, size_t size, wgpu::BufferUsage usage) {
    device_ = device;
    usage_ = usage | wgpu::BufferUsage::CopyDst;
    bufferMapped_ = CreateBufferMappedFromData(device_, data, size, usage_);
}

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu
