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

// WebNNGraph

WebNNGraph::WebNNGraph()
{

}

bool WebNNGraph::isInitialized()
{

}

void WebNNGraph::init(Target targetId)
{

}

void WebNNGraph::forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers)
{

}

void WebNNGraph::createGraph(Target targetId)
{

}

// WebNNBackendNode

WebNNBackendNode::WebNNBackendNode(const std::vector<Ptr<BackendNode> >& nodes,
                                         Ptr<Layer>& cvLayer_, std::vector<Mat*>& inputs,
                                         std::vector<Mat>& outputs, std::vector<Mat>& internals)
    : BackendNode(DNN_BACKEND_WEBNN), cvLayer(cvLayer_)
{
    std::ostringstream oss;
    oss << (size_t)cvLayer.get();

    std::map<std::string, InferenceEngine::Parameter> params = {
        {"impl", oss.str()},
        {"outputs", shapesToStr(outputs)},
        {"internals", shapesToStr(internals)}
    };

    for (const auto& node : nodes)
        inp_nodes.emplace_back(node.dynamicCast<WebNNBackendNode>()->node);
    node = std::make_shared<NgraphCustomOp>(inp_nodes, params);

    CV_Assert(!cvLayer->name.empty());
    setName(cvLayer->name);
}

void WebNNBackendNode::setName(const std::string& name)
{

}

// WebNNBackendWrapper

WebNNBackendWrapper::WebNNBackendWrapper(int targetId, const Mat& m)
{

}

WebNNBackendWrapper::WebNNBackendWrapper(Ptr<BackendWrapper> wrapper)
{

}

static Ptr<BackendWrapper> WebNNBackendWrapper::create(Ptr<BackendWrapper> wrapper)
{

}

void WebNNBackendWrapper::copyToHost()
{
    CV_LOG_DEBUG(NULL, "WebNNBackendWrapper::copyToHost()");
}

void WebNNBackendWrapper::setHostDirty()
{
    CV_LOG_DEBUG(NULL, "WebNNBackendWrapper::setHostDirty()");
}

void * WebNNBackendWrapper::getBuffer()
{
    return buffer;
}

#else
void forwardWebNN(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                   Ptr<BackendNode>& operand)
{
    CV_Assert(false && "WebNN is not enabled in this OpenCV build");
}

#endif

}
}