// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <fstream>
#include "op_webnn.hpp"

#include <opencv2/core/utils/configuration.private.hpp>
#include <opencv2/core/utils/logger.hpp>

#include "opencv2/core/utils/filesystem.hpp"
#include "opencv2/core/utils/filesystem.private.hpp"

#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

#ifdef HAVE_WEBNN

template<typename T>
static inline std::vector<T> getShape(const Mat& mat)
{
    std::vector<T> result(mat.dims);
    for (int i = 0; i < mat.dims; i++)
        result[i] = (T)mat.size[i];
    return result;
}

// WebnnGraph

WebnnGraph::WebnnGraph()
{
    mContext = ml::Context(webnn_native::CreateContext());
    builder = ml::CreateGraphBuilder(mContext);
}

bool WebnnGraph::isInitialized()
{
    return isInit;
}

// For common use
ml::Operand WebnnGraph::addOperand(std::string name, std::vector<int32_t> inputDimension, 
                          const std::vector<cv::Ptr<BackendWrapper> >& ptr,
                          std::unique_ptr<char> weightData,
                          uint32_t & byteOffset)
{
    const ml::Operand input = utils::BuildInput(builder, "input", inputDimension.data());
    Ptr<WebnnBackendWrapper> weightDimension = ptr.dynamicCast<WebnnBackendWrapper>();
    const std::vector<int32_t> weightShape = weightDimension.dimensions;
    const float* weightData = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t weightDataLength = product(weightShape) * sizeof(float);
    byteOffset += weightDataLength;
    const ml::Operand weightConstant = utils::BuildConstant(builder, weightShape, 
                                                            weightData,  weightDataLength);
    // the builder builds the first operand according to the input layer name,
    // like const ml::Operand conv1 = builder.Conv2d(input, weightConstant);
}

ml::Operand WebnnGraph::addOperand(std::string name, const ml::Operand inputOperand,
                          const std::vector<cv::Ptr<BackendWrapper> >& ptr,
                          std::unique_ptr<char> weightData,
                          uint32_t & byteOffset)
{
    Ptr<WebnnBackendWrapper> weightDimension = ptr.dynamicCast<WebnnBackendWrapper>();
    const std::vector<int32_t> weightShape = weightDimension.dimensions;
    const float* weightData = reinterpret_cast<float*>(weightsData.get() + byteOffset);
    const uint32_t weightDataLength = product(weightShape) * sizeof(float);
    byteOffset += weightDataLength;
    const ml::Operand weightConstant = utils::BuildConstant(builder, weightShape, 
                                                            weightData,  weightDataLength);
    // the builder builds the first operand according to the input layer name,
    // like const ml::Operand conv1 = builder.Conv2d(input, weightConstant);
}

ml::Operand addReLU(std::string name, std::vector<int32_t> inputDimension);
{
    const ml::Operand input = utils::BuildInput(builder, "input", inputDimension.data());
    const ml::Operand relu = builder.Relu(input);
    return relu;
}



void WebnnGraph::forward(const void* inputData, size_t inputLength)
{
    mResults = utils::AwaitCompute(mGraph, {{"input", {inputData, inputLength}}});
}

void WebnnGraph::createGraph(const ml::Operand outputOperand)
{
    mGraph = utils::AwaitBuild(builder, {{"output", outputOperand}});
    isInit = true;
}

// WebnnBackendNode

WebnnBackendNode::WebnnBackendNode(const ml::Operand layerOperand)
    : BackendNode(DNN_BACKEND_WEBNN))
{
    operand = layerOperand;
    graph = WebnnGraph();
}

WebnnBackendNode::WebnnBackendNode(std::shared_ptr<ml::Operand>&& _operand)
    : BackendNode(DNN_BACKEND_WEBNN), operand(std::move(_operand)) {}

WebnnBackendNode::WebnnBackendNode(std::shared_ptr<ml::Operand>&& _operand)
    : BackendNode(DNN_BACKEND_WEBNN), operand(_operand) {}

// WebnnBackendWrapper

WebnnBackendWrapper::WebnnBackendWrapper(const Mat& m)
{
    dimensions = getShape<uint32_t>(m);
    if (m.type() == CV_16F)
    {
        dataSize = m.total() * m.elemSize();
        buffer.reset(new char[dataSize]);
        std::memcpy(buffer.get(), (float16_t*)m.data, dataSize);
        descriptor = {ml::Operand::Float16, dimensions.data(), dimensions.size()};
    }
    else if (m.type() == CV_32F)
    {
        dataSize = m.total() * m.elemSize();
        buffer.reset(new char[dataSize]);
        std::memcpy(buffer.get(), (float32_t*)m.data, dataSize);
        descriptor = {ml::Operand::Float32, dimensions.data(), dimensions.size()};
    }
    else if (m.type() == CV_8U)
    {
        dataSize = m.total() * m.elemSize();
        buffer.reset(new char[dataSize]);
        std::memcpy(buffer.get(), (uint8_t*)m.data, dataSize);
        descriptor = {ml::Operand::Uint8, dimensions.data(), dimensions.size()};
    }
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %s", typeToString(m.type()).c_str()));
}

WebnnBackendWrapper::WebnnBackendWrapper(Ptr<BackendWrapper> wrapper)
{
    Ptr<WebnnBackendWrapper> webnnWrapper = wrapper.dynamicCast<WebnnBackendWrapper>();
    CV_Assert(!webnnWrapper.empty());
    buffer = webnnWrapper->buffer;
    descriptor = webnnWrapper->descriptor;
}

static Ptr<BackendWrapper> WebnnBackendWrapper::create(Ptr<BackendWrapper> wrapper)
{
    return Ptr<BackendWrapper>(new WebnnBackendWrapper(wrapper));
}

void WebnnBackendWrapper::copyToHost()
{
    CV_LOG_DEBUG(NULL, "WebnnBackendWrapper::copyToHost()");
}

void WebnnBackendWrapper::setHostDirty()
{
    CV_LOG_DEBUG(NULL, "WebnnBackendWrapper::setHostDirty()");
}

void * WebnnBackendWrapper::getBuffer()
{
    return buffer;
}

#else
void forwardWebnn(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                   Ptr<BackendNode>& operand)
{
    CV_Assert(false && "WebNN is not enabled in this OpenCV build");
}

#endif

}
}