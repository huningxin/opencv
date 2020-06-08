#ifndef OPENCV_DNN_WEBGPU_BUFFER_HPP
#define OPENCV_DNN_WEBGPU_BUFFER_HPP
// #ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
// #endif
namespace cv { namespace dnn { namespace webgpu {

class Buffer {
// #ifdef HAVE_WEBGPU
public:
    Buffer(const wgpu::Device& device);
    Buffer(const wgpu::Device& device, const void* data, size_t size, wgpu::BufferUsage usage = wgpu::BufferUsage::CopyDst);

    ~Buffer() {bufferMapped_.buffer.Destroy(); device_.Release();}
    wgpu::Buffer getWebGPUBuffer() {return bufferMapped_.buffer;}
    wgpu::BufferUsage getBufferUsage() {return usage_;}
    wgpu::CreateBufferMappedResult getBufferMapped() {return bufferMapped_;}
    void* getBufferMappedData() {return bufferMapped_.data;}
private:
    Buffer();
    wgpu::Device device_;
    wgpu::CreateBufferMappedResult bufferMapped_;
    wgpu::BufferUsage usage_;
};

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu

#endif  //OPENCV_DNN_WEBGPU_OP_BASE_HPP