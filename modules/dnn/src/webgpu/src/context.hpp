#ifndef OPENCV_DNN_WEBGPU_CONTEXT_HPP
#define OPENCV_DNN_WEBGPU_CONTEXT_HPP

namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU
class Context
{

public:
    Context();
    ~Context();
};

void createContext();

#endif  // HAVE_WEBGPU

}}}  // namespace cv::dnn::webgpu

#endif // OPENCV_DNN_WEBGPU_CONTEXT_HPP
