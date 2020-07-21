#ifndef OPENCV_DNN_OP_WEBGPU_HPP
#define OPENCV_DNN_OP_WEBGPU_HPP

#include <opencv2/dnn/shape_utils.hpp>
// #ifdef HAVE_WEBGP&U
#include "webgpu/include/wgpucom.hpp"
// #endif  // HAVE_WEBGPU

namespace cv
{
namespace dnn
{
// #ifdef HAVE_WEBGPU
    std::vector<webgpu::Tensor> WGPUTensors(
        const std::vector<Ptr<BackendWrapper> >& ptrs);
    webgpu::Tensor WGPUTensor(const Ptr<BackendWrapper>& ptr);
    void copyToTensor(webgpu::Tensor &dst, const Mat &src);

    void copyToMat(Mat &dst, const webgpu::Tensor &src);

    class WGPUBackendWrapper : public BackendWrapper
    {
    public:
        WGPUBackendWrapper(Mat& m);
        WGPUBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m);

        virtual void copyToHost() CV_OVERRIDE;
        virtual void setHostDirty() CV_OVERRIDE;
        void setDeviceDirty();
        void copyToDevice();
        webgpu::Tensor getTensor();

    private:
        webgpu::Tensor tensor;
        Mat* host;
        bool hostDirty;
        bool deviceDirty;
    };

    class WGPUBackendNode : public BackendNode 
    {
    public:
        WGPUBackendNode(const std::vector<Ptr<BackendWrapper> >& inputsWrapper,
                         const std::shared_ptr<webgpu::OpBase> &op,
                         const std::vector<Ptr<BackendWrapper> >& blobsWrapper =
                         std::vector<Ptr<BackendWrapper> >());

        bool forward(std::vector<webgpu::Tensor>& outs);

    private:
        std::vector<webgpu::Tensor> ins;
        std::vector<webgpu::Tensor> blobs;
        std::vector<Ptr<BackendWrapper> > inputsWrapper_;
        std::shared_ptr<webgpu::OpBase> operation;
    };
// #endif  //HAVE_WEBGPU

    void forwardWGPU(std::vector<Ptr<BackendWrapper> > &outputs, 
                    const Ptr<BackendNode>& node);

    bool haveWGPU();
}  // namespace dnn

}  //namespace cv

#endif  //OPENCV_DNN_OP_WEBGPU_HPP
