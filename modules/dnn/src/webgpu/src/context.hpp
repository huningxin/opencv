#ifndef OPENCV_DNN_WEBGPU_CONTEXT_HPP
#define OPENCV_DNN_WEBGPU_CONTEXT_HPP

namespace cv { namespace dnn { namespace webgpu {
class Context
{
// #ifdef HAVE_WEBGPU

public:
    Context();
    ~Context();
std::shared_ptr<Context> wCtx;
std::shared_ptr<wgpu::Device> wDevice = nullptr;
std::shared_ptr<wgpu::Queue> wQueue = nullptr;
cv::Mutex wContextMtx;
};

void createContext();

// #endif  //HAVE_WEBGPU

}}}  // namespace cv::dnn::webgpu

#endif // OPENCV_DNN_WEBGPU_CONTEXT_HPP
