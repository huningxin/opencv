#ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#endif  // HAVE_WEBGPU
namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU

    wgpu::Device createCppDawnDevice();

#endif   //HAVE_WEBGPU

}}}     // namespace cv::dnn::webgpu
