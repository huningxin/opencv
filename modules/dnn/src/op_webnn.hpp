// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_OP_WEBNN_HPP__
#define __OPENCV_DNN_OP_WEBNN_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/dnn.hpp"

#ifdef HAVE_WEBNN

#include <webnn/webnn_cpp.h>
#include <webnn/webnn.h>
#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>

#endif  // HAVE_WEBNN

namespace cv { namespace dnn {

constexpr bool haveWebNN() {
#ifdef HAVE_WEBNN
        return true;
#else
        return false;
#endif
}

#ifdef HAVE_WEBNN

class WebnnBackendNode;


class WebnnGraph
{
public:
    WebnnGraph();

    bool isInitialized();

    void init(Target targetId);

    void forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers);

    void createGraph(Target targetId);

private:
    ml::GraphBuilder builder;
    ml::Context mContext;
    ml::Graph mGraph;
    ml::NamedOperands mResults;
    bool isInit = false;
};

class WebnnBackendNode : public BackendNode
{
public:
    WebnnBackendNode(const std::vector<Ptr<BackendNode> >& nodes, Ptr<Layer>& layer,
                        std::vector<Mat*>& inputs, std::vector<Mat>& outputs,
                        std::vector<Mat>& internals);

    ml::Operand operand;
    WebnnGraph graph;
};

class WebnnBackendWrapper : public BackendWrapper
{
public:
    WebnnBackendWrapper(const Mat& m);
    WebnnBackendWrapper(Ptr<BackendWrapper> wrapper);
    // ~WebnnBackendWrapper();

    static Ptr<BackendWrapper> create(Ptr<BackendWrapper> wrapper);

    virtual void copyToHost() CV_OVERRIDE;
    virtual void setHostDirty() CV_OVERRIDE;
    virtual void * getBuffer();

private:
    void * buffer;
    ml::OperandDescriptor descriptor;
    struct Descriptor{
        ml::OperandType type;
        std::vector<uint32_t> dataShape;
    } shapeStorer;
};

#endif  // HAVE_WebNN

void forwardWebnn(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                  Ptr<BackendNode>& operand);

}}  // namespace cv::dnn


#endif  // __OPENCV_DNN_OP_WEBNN_HPP__
