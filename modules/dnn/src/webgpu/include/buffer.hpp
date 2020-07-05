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
    Buffer( const std::shared_ptr<wgpu::Device> device, 
            const void* data, size_t size, 
            wgpu::BufferUsage usage = wgpu::BufferUsage::Storage);

    ~Buffer() {bufferMapped_->buffer.Release();}
    wgpu::Buffer * getWebGPUBuffer() 
    { return & bufferMapped_->buffer; }
    wgpu::BufferUsage getBufferUsage() { return usage_;}
    std::shared_ptr<wgpu::CreateBufferMappedResult> getBufferMapped()
    { return bufferMapped_; }
    void* getBufferMappedData() { return bufferMapped_->data; }
    
    static void BufferMapReadCallback(WGPUBufferMapAsyncStatus status,
                                   const void* data,
                                   uint64_t dataLength,
                                   void* userdata)
    {
        static_cast<Buffer*>(userdata)->mappedData = data;
    }
    const void* MapReadAsyncAndWait(); 
    
private:
    Buffer();
    std::shared_ptr<wgpu::Device> device_;
    std::shared_ptr<wgpu::CreateBufferMappedResult>  bufferMapped_;
    wgpu::BufferUsage usage_;
    size_t size_;
    const void * mappedData = nullptr;
};

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu

#endif  //OPENCV_DNN_WEBGPU_OP_BASE_HPP