#ifndef OPENCV_DNN_WEBGPU_BUFFER_HPP
#define OPENCV_DNN_WEBGPU_BUFFER_HPP
// #ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include <unistd.h>
#include <memory>
// #endif
namespace cv { namespace dnn { namespace webgpu {

class Buffer 
{
// #ifdef HAVE_WEBGPU
public:
    Buffer(const std::shared_ptr<wgpu::Device> device);
    Buffer(const std::shared_ptr<wgpu::Device> device, 
           const void* data, size_t size, 
           wgpu::BufferUsage usage = wgpu::BufferUsage::Storage | 
           wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::CopySrc);
    Buffer(const void* data, size_t size,  
           wgpu::BufferUsage usage = wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst);
    ~Buffer() 
    {
        if(buffer_) buffer_.Release();
        if(gpuReadBuffer_) gpuReadBuffer_.Release();
    }
    wgpu::Buffer * getWebGPUBuffer() { return & buffer_; }
    wgpu::BufferUsage getBufferUsage() { return usage_;}
    
    static void BufferMapReadCallback(WGPUBufferMapAsyncStatus status,
                                      const void* data,
                                      uint64_t dataLength,
                                      void* userdata)
    {
        static_cast<Buffer*>(userdata)->mappedData = data;
    }
    void setBufferData(const void * data, size_t size);
    const void* MapReadAsyncAndWait(); 
    void unMap() { if(gpuReadBuffer_) gpuReadBuffer_.Unmap(); }
    size_t getSize() { return size_; }
private:
    Buffer();
    std::shared_ptr<wgpu::Device> device_;
    wgpu::Buffer buffer_;
    wgpu::Buffer gpuReadBuffer_ = nullptr;
    wgpu::BufferUsage usage_;
    size_t size_;
    const void * mappedData = nullptr;
};

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu

#endif  //OPENCV_DNN_WEBGPU_OP_BASE_HPP