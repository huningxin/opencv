// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

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
#include <dawn/webgpu_cpp.h>
#include "opencv2/core/utils/logger.hpp"
#include "../shader/spv_shader.hpp"
#include "../../precomp.hpp"

namespace cv { namespace dnn { namespace webgpu {

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

}}} // namespace cv::dnn::vkcom

#endif // OPENCV_DNN_VKCOM_COMMON_HPP