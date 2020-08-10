#include "../../precomp.hpp"
#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webgpu.h>
#include <webgpu/webgpu_cpp.h>
#else
#ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn_native/DawnNative.h>
#endif  // HAVE_WEBGPU
#endif  //__EMSCRIPTEN__
namespace cv { namespace dnn { namespace webgpu {
#if defined(HAVE_WEBGPU) || defined(__EMSCRIPTEN__)

    wgpu::Device createCppDawnDevice();

#endif   //HAVE_WEBGPU

}}}     // namespace cv::dnn::webgpu
