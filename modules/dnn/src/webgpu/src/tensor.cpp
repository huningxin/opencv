#include "../include/tensor.hpp"
#include "common.hpp"
#include "internal.hpp"
#include <unistd.h>
namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU
Tensor::Tensor(Format fmt) : size_in_byte_(0), format_(fmt)
{
    createContext();
    device_ = wDevice;
}

Tensor::Tensor(const char* data, size_t size_in_byte, wgpu::BufferUsage usage, Format fmt)
{
    createContext();
    device_ = wDevice;
    size_in_byte_ = size_in_byte;
    usage_ = usage;
    format_ = fmt;
    setUniform(data, fmt);
}

Tensor::Tensor(const char* data, std::vector<int>& shape, wgpu::BufferUsage usage, Format fmt) 
{
    createContext();
    device_ = wDevice;
    size_in_byte_ = 0;
    usage_ = usage;
    format_ = fmt;
    reshape(data, shape);
}

void* Tensor::map()
{
    return buffer_->getBufferMappedData();

}

void Tensor::unMap()
{
    buffer_->getWebGPUBuffer()->Unmap();
}

Shape Tensor::getShape() const{
    return shape_;
}

int Tensor::count(const int start_axis, const int end_axis) const
{
    return shapeCount(shape_, start_axis, end_axis);
}

int Tensor::dimSize(const int axis) const
{
    CV_Assert(axis >= 0);
    CV_Assert(axis < shape_.size());

    return shape_[axis];
}

int Tensor::dimNum() const
{
    return shape_.size();
}

Tensor Tensor::reshape(const void* data, const std::vector<int>& shape, bool alloc, Format fmt)
{
    if (device_ == nullptr)
    {
        CV_Error(Error::StsError, "device is NULL");
        return *this;
    }
    CV_Assert(shape.size() > 0 && shape.size() <= 6);

    if (shape_ != shape) shape_ = shape;
    if (checkFormat(fmt) && fmt != format_) format_ = fmt;

    size_t new_size = shapeCount(shape_) * elementSize(format_);
    if (alloc || new_size > size_in_byte_)
        alloc = true;
    size_in_byte_ = new_size;
    if (alloc || !buffer_)
    {
        // TODO: specific data type 
        buffer_.reset(new Buffer(device_, data, size_in_byte_, usage_));
    }
    else if (data)
    {
        memcpy(buffer_->getBufferMapped()->data, data, size_in_byte_);
    }
    return *this;
}

Tensor Tensor::setUniform(const void * data, Format fmt) 
{
    if (device_ == nullptr)
    {
        CV_Error(Error::StsError, "device is NULL");
        return *this;
    }
    if (checkFormat(fmt) && fmt != format_) format_ = fmt;
    if (!buffer_)
    {
        // TODO: specific data type 
        buffer_.reset(new Buffer(device_, data, size_in_byte_, usage_));
    }
    else if (data)
    {
        memcpy(buffer_->getBufferMapped()->data, data, size_in_byte_);
    }
    return *this;
}

int Tensor::getFormat() const
{
    return format_;
}

void Tensor::copyTo(Tensor & dst) {
    dst.reshape(buffer_->getBufferMapped()->data, shape_, true, format_);
}

void readBufferMapReadCallback(WGPUBufferMapAsyncStatus status,
                                const void* ptr,
                                uint64_t dataLength,
                                void* userdata) {
	(void)status;
	(void)userdata;
    if(dataLength == 0) {
        CV_Error(cv::Error::StsAssert, "Result Buffer is NULL");
    }
    memcpy(userdata, ptr, dataLength);
}

void Tensor::mapReadAsync(void * result) {
    wgpu::BufferDescriptor desc = {};
    desc.size = size_in_byte_;
    desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    wgpu::Buffer gpuReadBuffer = device_->CreateBuffer(& desc);
    wgpu::CommandEncoder encoder = device_->CreateCommandEncoder();
    encoder.CopyBufferToBuffer(* buffer_->getWebGPUBuffer(), 0, gpuReadBuffer, 0, size_in_byte_);
    wgpu::CommandBuffer cmdBuffer = encoder.Finish();
    wQueue->Submit(1, &cmdBuffer);
    gpuReadBuffer.MapReadAsync(readBufferMapReadCallback, result);
}

// #endif   //HAVE_WEBGPU

}}}  //namespace cv::dnn:webgpu