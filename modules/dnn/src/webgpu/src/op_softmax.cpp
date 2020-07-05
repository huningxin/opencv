#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_softmax.hpp"

namespace cv { namespace dnn { namespace webgpu {

// #ifdef HAVE_WEBGPU

#define LOCAL_SZ_X 256
#define LOCAL_SZ_Y 1
#define LOCAL_SZ_Z 1

struct SoftmaxParam 
{
    int channel_size;
    int outer_size;
    int channels;
    int logsoftmax;
};

OpSoftmax::OpSoftmax(const int axis, const bool log_softmax)
{
    init(axis, log_softmax);
    type_ = "Softmax";
    needsUniform = true;
}

OpSoftmax::~OpSoftmax()
{
    if (max_tensor_)
        delete max_tensor_;
    if (sum_tensor_)
        delete sum_tensor_;
}

void OpSoftmax::reshapeOutTensor(Tensor& in, Tensor& out)
{
    Shape shape = in.getShape();
    out.reshape(NULL, shape);
}

bool OpSoftmax::init(const int axis, const bool log_softmax)
{
    axis_ = axis;
    log_softmax_ = log_softmax;
    max_tensor_ = NULL;
    sum_tensor_ = NULL;
    createBindGroupLayout(4);
    return true;
}

bool OpSoftmax::forward(std::vector<Tensor>& ins,
                        std::vector<Tensor>& blobs,
                        std::vector<Tensor>& outs)
{
    return forward(ins[0], outs[0]);
}

bool OpSoftmax::forward(Tensor& in, Tensor& out)
{
    channels_ = in.dimSize(axis_);
    channel_size_ = in.count(axis_+1);
    outer_size_ = in.count(0, axis_);

    if (pipeline_ == nullptr)
    {
        config_.local_size_x = LOCAL_SZ_X;
        config_.local_size_y = LOCAL_SZ_Y;
        config_.local_size_z = LOCAL_SZ_Z;
        config_.block_height = 1;
        config_.block_width  = 1;
        config_.block_depth  = 1;
        createShaderModule(softmax_spv, sizeof(softmax_spv)/sizeof(uint32_t));
        createComputePipeline();
        computeGroupCount();
    }
    if (max_tensor_ == NULL || sum_tensor_ == NULL)
    {
        std::vector<int> shape = {outer_size_, channel_size_};
        max_tensor_ = new Tensor(NULL, shape, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
        sum_tensor_ = new Tensor(NULL, shape, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc);
    }
    if(uniformTensor_ == NULL && needsUniform) 
    {
        SoftmaxParam param = {channel_size_, outer_size_, channels_, 
                              log_softmax_ == true ? 1 : 0};
        uniformTensor_ = new Tensor((const void*) &param, 
                                    sizeof(SoftmaxParam), 
                                    wgpu::BufferUsage::Uniform | wgpu::BufferUsage::CopyDst, 
                                    wFormatInt32);
    }
    
    bindTensor( in,  0, bgEntries);
    bindTensor( *max_tensor_,  1, bgEntries);
    bindTensor( *sum_tensor_,  2, bgEntries);
    bindTensor( out, 3, bgEntries);
    bindTensor( *uniformTensor_, 4, bgEntries);
    in.unMap(); 
    max_tensor_->unMap();
    sum_tensor_->unMap();
    out.unMap();
    uniformTensor_->unMap();
    createBindGroup();
    createCommandBuffer();
    runCommandBuffer();
    return true;
}

bool OpSoftmax::computeGroupCount()
{
    group_x_ = alignSize(outer_size_, config_.local_size_x) / config_.local_size_x;
    group_y_ = 1;
    group_z_ = 1;
    return true;
}

// #endif // HAVE_WEBGPU

}}} // namespace cv::dnn::webgpu
