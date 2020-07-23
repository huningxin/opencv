#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_relu.hpp"

namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU

#define LOCAL_SZ_X 32

struct ReLUParam {
      int total;
      float slope;
};

OpReLU::OpReLU(const float slope) : slope_(slope)
{
    createBindGroupLayout(2);
    type_ = "ReLU";
}

OpReLU::~OpReLU()
{
    if (uniformBuffer_)
        uniformBuffer_->getWebGPUBuffer()->Release();
}

void OpReLU::reshapeOutTensor(Tensor& in, Tensor& out)
{
    Shape shape = in.getShape();
    out.reshape(NULL, shape);
}

bool OpReLU::forward(std::vector<Tensor>& ins,
                     std::vector<Tensor>& blobs,
                     std::vector<Tensor>& outs)
{
    return forward(ins[0], outs[0]);
}

bool OpReLU::forward(Tensor& in, Tensor& out)
{
    if (pipeline_ == nullptr)
    {
        total_ = in.count();
#define maxComputeWorkGroupCount 65535
        computeGroupCount();
        createShaderModule(relu_spv, sizeof(relu_spv)/sizeof(uint32_t));
        createComputePipeline();
    }

    bindTensor(in,  0, bgEntries);
    bindTensor(out, 1, bgEntries);
    ReLUParam param = { total_, slope_ };
    if(! uniformBuffer_) uniformBuffer_ = new Buffer(&param, sizeof(ReLUParam));
    else uniformBuffer_->setBufferData(&param, sizeof(ReLUParam));
    bindUniform(* uniformBuffer_, 2, bgEntries);
    createBindGroup();
    createCommandBuffer();
    runCommandBuffer();
    return true;
}

bool OpReLU::computeGroupCount()
{
    group_x_ = alignSize(total_, LOCAL_SZ_X) / LOCAL_SZ_X;
    if (group_x_ > maxComputeWorkGroupCount)
        group_x_ = maxComputeWorkGroupCount;
    group_y_ = 1;
    group_z_ = 1;
    return true;
}

#endif  // HAVE_WEBGPU

}}} // namespace cv::dnn::webgpu
