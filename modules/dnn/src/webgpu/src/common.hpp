#ifndef OPENCV_DNN_WEBGPU_COMMON_HPP
#define OPENCV_DNN_WEBGPU_COMMON_HPP

#include <math.h>
#include <string.h>
#include <map>
#include <mutex>
#include <thread>
#include <vector>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <memory>
#ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#endif  // HAVE_WEBGPU
#include "opencv2/core/utils/logger.hpp"
#include "../shader/spv_shader.hpp"
#include "../include/wgpucom.hpp"
namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU
extern std::shared_ptr<wgpu::Device> wDevice;
extern std::shared_ptr<wgpu::Queue> wQueue;
extern cv::Mutex wContextMtx;

enum ShapeIdx
{
    wShapeIdxBatch = 0,
    wShapeIdxChannel,
    wShapeIdxHeight,
    wShapeIdxWidth,
};

#define WGPU_CHECK_BOOL_RET_VAL(val, ret) \
{ \
    bool res = (val); \
    if (!res) \
    { \
        CV_LOG_WARNING(NULL, "Check bool failed"); \
        return ret; \
    } \
}

#define WGPU_CHECK_POINTER_RET_VOID(p) \
{ \
    if (NULL == (p)) \
    { \
        CV_LOG_WARNING(NULL, "Check pointer failed"); \
        return; \
    } \
}

#define WGPU_CHECK_POINTER_RET_VAL(p, val) \
{ \
    if (NULL == (p)) \
    { \
        CV_LOG_WARNING(NULL, "Check pointer failed"); \
        return (val); \
    } \
}

#endif  // HAVE_WEBGPU

}}} // namespace cv::dnn::webgpu

#endif // OPENCV_DNN_WEBGPU_COMMON_HPP