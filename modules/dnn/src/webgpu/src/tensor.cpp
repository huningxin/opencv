#include "../include/tensor.hpp"
#include "common.hpp"
#include "internal.hpp"
namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU
Tensor::Tensor(Format fmt) : size_in_byte_(0), format_(fmt)
{
    createContext();
    device_ = wDevice;
}
    
Tensor::Tensor(const char* data, std::vector<int>& shape, Format fmt = wFormatFp32) {
    createContext();
    device_ = wDevice;
    reshape(data, shape);
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

Tensor Tensor::reshape(const char* data, const std::vector<int>& shape, bool alloc, Format fmt)
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

    if (alloc)
    {
        buffer_.reset(new Buffer(device_, size_in_byte_, data, buffer_.getBufferUsage());
    }
    else if (data)
    {
        memcpy(buffer_.getBufferMapped().data, data, size_in_byte_);
    }

    return *this;
}
int Tensor::getFormat() const
{
    return format_;
}

void Tensor::copyTo(Tensor & dst) {
    dst.reshape(buffer_.getBufferMapped().data, shape_, true, format_);
}

// #endif   //HAVE_WEBGPU

}}}  //namespace cv::dnn:webgpu