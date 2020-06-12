#ifndef OPENCV_DNN_WEBGPU_BUFFER_HPP
#define OPENCV_DNN_WEBGPU_BUFFER_HPP
// #ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include<unistd.h>
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
    wgpu::Buffer buffer_;
};

class BufferReadAsync{
public:
void WaitABit() {
    device_.Tick();
    usleep(100);
}
static void BufferMapReadCallback(WGPUBufferMapAsyncStatus status,
                                   const void* data,
                                   uint64_t,
                                   void* userdata)
{
    static_cast<BufferReadAsync*>(userdata)->mappedData = data;
}

const void* MapReadAsyncAndWait(const wgpu::Buffer& buffer) {
    buffer.MapReadAsync(BufferMapReadCallback, this);
    while(mappedData == nullptr) {
        WaitABit();
    }
    return mappedData;
}
void UnmapBuffer(const wgpu::Buffer& buffer) {
    buffer.Unmap();
    mappedData = nullptr;
}
private:
wgpu::Device device_;
const void * mappedData = nullptr;
};

class BufferWriteAsync{
public:
void WaitABit() {
    device_.Tick();
    usleep(100);
}
static void BufferMapWriteCallback(WGPUBufferMapAsyncStatus status,
                                   void* data,
                                   uint64_t,
                                   void* userdata)
{
    userdata = data;
}
void* MapWriteAsyncAndWait(const wgpu::Buffer& buffer) {
    buffer.MapWriteAsync(BufferMapWriteCallback, this);
    while(mappedData == nullptr){
        WaitABit();
    }
    void* resultPointer = mappedData;
    mappedData = nullptr;
    return resultPointer;
}
void UnmapBuffer(const wgpu::Buffer& buffer) {
    buffer.Unmap();
    mappedData = nullptr;
}
private:
void * mappedData;
wgpu::Device device_;
};

// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu

#endif  //OPENCV_DNN_WEBGPU_OP_BASE_HPP