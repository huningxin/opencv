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
// #ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
// #endif
#include "opencv2/core/utils/logger.hpp"
#include "../shader/spv_shader.hpp"
#include "../../precomp.hpp"
#include "../include/wgpucom.hpp"
#include "context.hpp"
namespace cv { namespace dnn { namespace webgpu {

// #ifdef HAVE_WEBGPU

extern wgpu::Device wDevice;
extern wgpu::Queue wQueue;
extern wgpu::CommandEncoder wEncoder;
extern cv::Mutex wContextMtx;

enum ShapeIdx
{
    kShapeIdxBatch = 0,
    kShapeIdxChannel,
    kShapeIdxHeight,
    kShapeIdxWidth,
};


// #endif  //HAVE_WEBGPU

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_WEBGPU_COMMON_HPP