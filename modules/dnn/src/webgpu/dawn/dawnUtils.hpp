#include "../../precomp.hpp"
#ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn_native/DawnNative.h>
#endif  // HAVE_WEBGPU
namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU

    wgpu::Device createCppDawnDevice();

#endif   //HAVE_WEBGPU

}}}     // namespace cv::dnn::webgpu
