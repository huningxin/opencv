// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_WEBGPU_HPP
#define OPENCV_DNN_WEBGPU_HPP

#include <vector>

namespace cv { namespace dnn { namespace webgpu {
//#ifndef HAVE_WEBGPU
enum Format{
    wFormatInvalid = -1,
    wFormatFp16,
    wFormatFp32,
    wFormatFp64,
    wFormatInt32,
    wFormatNum
};
enum OpType {
    wOpTypeConv,
    wOpTypePool,
    wOpTypeDWConv,
    wOpTypeLRN,
    wOpTypeConcat,
    wOpTypeSoftmax,
    wOpTypeReLU,
    wOpTypePriorBox,
    wOpTypePermute,
    wOpTypeNum
};
enum PaddingMode { wPaddingModeSame, wPaddingModeValid, wPaddingModeCaffe, wPaddingModeNum };
enum FusedActivationType { wNone, wRelu, wRelu1, wRelu6, wActivationNum };
typedef std::vector<int> Shape;

bool isAvailable();

//#endif  //HAVE_WEBGPU

}}} //namespace cv::dnn::webgpu

typedef std::vector<int> Shape;
bool isAvailable();

#include "tensor.hpp"
#include "buffer.hpp"

#endif//    OPENCV_DNN_WEBGPU_HPP