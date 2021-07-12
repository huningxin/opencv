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

static std::string kDefaultInpLayerName = "opencv_webnn_empty_inp_layer_name";

template<typename T>
static inline std::vector<T> getShape(const Mat& mat)
{
    std::vector<T> result(mat.dims);
    for (int i = 0; i < mat.dims; i++)
        result[i] = (T)mat.size[i];
    return result;
}

static std::vector<Ptr<WebnnBackendWrapper> >
webnnWrappers(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    std::vector<Ptr<WebnnBackendWrapper> > wrappers(ptrs.size());
    for (int i = 0; i < ptrs.size(); ++i)
    {
        CV_Assert(!ptrs[i].empty());
        wrappers[i] = ptrs[i].dynamicCast<WebnnBackendWrapper>();
        CV_Assert(!wrappers[i].empty());
    }
    return wrappers;
}

// WebnnNet
WebnnNet::WebnnNet()
{
    hasNetOwner = false;
    device_name = "CPU";
}

void WebnnNet::addOutput(const std::string& name)
{
    requestedOutputs.push_back(name);
}

void WebnnNet::createNet(Target targetId) {
    init(targetId);
}

void WebnnNet::init(Target targetId)
{
    switch (targetId)
    {
        case DNN_TARGET_CPU:
            device_name = "CPU";
            break;
        case DNN_TARGET_OPENCL:
            device_name = "GPU";
            break;
        default:
            CV_Error(Error::StsNotImplemented, "Unknown target");
    };

    CV_Error(Error::StsNotImplemented, "Create ml::Graph");
}

std::vector<ml::Operand> WebnnNet::setInputs(const std::vector<cv::Mat>& inputs,
                                            const std::vector<std::string>& names) {
    CV_Assert_N(inputs.size() == names.size());
    std::vector<ml::Operand> current_inp;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        CV_Error(Error::StsNotImplemented, "Create ml::Operand");
    }
    return current_inp;
}

void WebnnNet::setUnconnectedNodes(Ptr<WebnnBackendNode>& node) {
    unconnectedNodes.push_back(node);
}

bool WebnnNet::isInitialized()
{
    return isInit;
}

void WebnnNet::reset()
{
    allBlobs.clear();
    isInit = false;
}

void WebnnNet::addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs)
{
    auto wrappers = webnnWrappers(ptrs);
    for (const auto& wrapper : wrappers)
    {
        std::string name = wrapper->name;
        name = name.empty() ? kDefaultInpLayerName : name;
        allBlobs.insert({name, wrapper});
    }
}

void WebnnNet::forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers, bool isAsync)
{
    CV_LOG_DEBUG(NULL, "WebnnNet::forward(" << (isAsync ? "async" : "sync") << ")");
    CV_Error(Error::StsNotImplemented, "Implement ml::Graph.compute");
}

// WebnnBackendNode
WebnnBackendNode::WebnnBackendNode(ml::Operand&& _operand)
    : BackendNode(DNN_BACKEND_WEBNN), operand(std::move(_operand)) {}

WebnnBackendNode::WebnnBackendNode(ml::Operand& _operand)
    : BackendNode(DNN_BACKEND_WEBNN), operand(_operand) {}

// WebnnBackendWrapper
WebnnBackendWrapper::WebnnBackendWrapper(int targetId, const cv::Mat& m)
    : BackendWrapper(DNN_BACKEND_WEBNN, targetId)
{
    size_t dataSize = m.total() * m.elemSize();
    buffer.reset(new char[dataSize]);
    std::memcpy(buffer.get(), m.data, dataSize);
    dimensions = getShape<int32_t>(m);
    descriptor.dimensions = dimensions.data();
    descriptor.dimensionsCount = dimensions.size();
    if (m.type() == CV_32F)
    {
        descriptor.type = ml::OperandType::Float32;
    }
    else
        CV_Error(Error::StsNotImplemented, format("Unsupported data type %s", typeToString(m.type()).c_str()));
}

WebnnBackendWrapper::~WebnnBackendWrapper()
{
    // nothing
}

void WebnnBackendWrapper::copyToHost()
{
    CV_LOG_DEBUG(NULL, "WebnnBackendWrapper::copyToHost()");
    //CV_Error(Error::StsNotImplemented, "");
}

void WebnnBackendWrapper::setHostDirty()
{
    CV_LOG_DEBUG(NULL, "WebnnBackendWrapper::setHostDirty()");
    //CV_Error(Error::StsNotImplemented, "");
}

void forwardWebnn(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                  Ptr<BackendNode>& node, bool isAsync)
{
    CV_Assert(!node.empty());
    Ptr<WebnnBackendNode> webnnNode = node.dynamicCast<WebnnBackendNode>();
    CV_Assert(!webnnNode.empty());
    webnnNode->net->forward(outBlobsWrappers, isAsync);
}


#else
void forwardWebnn(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                   Ptr<BackendNode>& operand, bool isAsync)
{
    CV_Assert(false && "WebNN is not enabled in this OpenCV build");
}

#endif

}
}