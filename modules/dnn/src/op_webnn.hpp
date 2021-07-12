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

    ml::Operand addOperand(std::string name, std::vector<int32_t> inputDimension, 
                          const std::vector<cv::Ptr<BackendWrapper> >& ptrs,
                          std::unique_ptr<char> weightData,
                          uint32_t & byteOffset);

    ml::Operand addOperand(std::string name, const ml::Operand inputOperand,
                          const std::vector<cv::Ptr<BackendWrapper> >& ptrs,
                          std::unique_ptr<char> weightData,
                          uint32_t & byteOffset);
    
    // for relu test
    ml::Operand addReLU(std::string name, std::vector<int32_t> inputDimension);

    void forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers);

    void createGraph(const ml::Operand outputOperand);
    
    void reset();

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
    WebnnBackendNode(const ml::Operand layerOperand);

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
    std::unique_ptr<char> buffer;
    ml::OperandDescriptor descriptor;
    std::vector<uint32_t> dimensions;
};

#endif  // HAVE_WebNN

void forwardWebnn(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                  Ptr<BackendNode>& operand);

}}  // namespace cv::dnn


#endif  // __OPENCV_DNN_OP_WEBNN_HPP__
