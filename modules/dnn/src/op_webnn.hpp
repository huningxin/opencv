// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018-2019, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_WEBNN_HPP__
#define __OPENCV_DNN_WEBNN_HPP__

#include "opencv2/core/cvdef.h"
#include "opencv2/core/cvstd.hpp"
#include "opencv2/dnn.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "opencv2/core/async.hpp"
#include "opencv2/core/detail/async_promise.hpp"

#ifdef HAVE_WEBNN
#include <webnn/webnn_cpp.h>
#include <webnn/webnn.h>
#include <webnn/webnn_proc.h>
#include <webnn_native/WebnnNative.h>
#endif  // HAVE_WEBNN


namespace cv { namespace dnn {

#ifdef HAVE_WEBNN

constexpr bool haveWebNN() {
#ifdef HAVE_WEBNN
        return true;
#else
        return false;
#endif
    }

Mat WebNNMLOperandsToMat(const ml::MLOperand::Ptr& blob);

void WebNNMLOperandsToMats(const std::vector<ml::MLOperand::Ptr>& blobs,
                          std::vector<Mat>& mats);

#ifdef HAVE_DNN_IE_NN_BUILDER_2019

class InfEngineBackendNet
{
public:
    InfEngineBackendNet();

    InfEngineBackendNet(InferenceEngine::CNNNetwork& net);

    void addLayer(InferenceEngine::Builder::Layer& layer);

    void addOutput(const std::string& name);

    void connect(const std::vector<Ptr<BackendWrapper> >& inputs,
                 const std::vector<Ptr<BackendWrapper> >& outputs,
                 const std::string& layerName);

    bool isInitialized();

    void init(Target targetId);

    void forward(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                 bool isAsync);

    void initPlugin(InferenceEngine::CNNNetwork& net);

    void addBlobs(const std::vector<cv::Ptr<BackendWrapper> >& ptrs);

    void reset();

private:
    InferenceEngine::Builder::Network netBuilder;

    InferenceEngine::ExecutableNetwork netExec;
    InferenceEngine::BlobMap allBlobs;
    std::string device_name;
#if INF_ENGINE_VER_MAJOR_LE(2019010000)
    InferenceEngine::InferenceEnginePluginPtr enginePtr;
    InferenceEngine::InferencePlugin plugin;
#else
    bool isInit = false;
#endif

    struct InfEngineReqWrapper
    {
        InfEngineReqWrapper() : isReady(true) {}

        void makePromises(const std::vector<Ptr<BackendWrapper> >& outs);

        InferenceEngine::InferRequest req;
        std::vector<cv::AsyncPromise> outProms;
        std::vector<std::string> outsNames;
        bool isReady;
    };

    std::vector<Ptr<InfEngineReqWrapper> > infRequests;

    InferenceEngine::CNNNetwork cnn;
    bool hasNetOwner;

    std::map<std::string, int> layers;
    std::vector<std::string> requestedOutputs;

    std::set<std::pair<int, int> > unconnectedPorts;
};

class InfEngineBackendNode : public BackendNode
{
public:
    InfEngineBackendNode(const InferenceEngine::Builder::Layer& layer);

    InfEngineBackendNode(Ptr<Layer>& layer, std::vector<Mat*>& inputs,
                         std::vector<Mat>& outputs, std::vector<Mat>& internals);

    void connect(std::vector<Ptr<BackendWrapper> >& inputs,
                 std::vector<Ptr<BackendWrapper> >& outputs);

    // Inference Engine network object that allows to obtain the outputs of this layer.
    InferenceEngine::Builder::Layer layer;
    Ptr<InfEngineBackendNet> net;
    // CPU fallback in case of unsupported Inference Engine layer.
    Ptr<dnn::Layer> cvLayer;
};

class InfEngineBackendWrapper : public BackendWrapper
{
public:
    InfEngineBackendWrapper(int targetId, const Mat& m);

    InfEngineBackendWrapper(Ptr<BackendWrapper> wrapper);

    ~InfEngineBackendWrapper();

    static Ptr<BackendWrapper> create(Ptr<BackendWrapper> wrapper);

    virtual void copyToHost() CV_OVERRIDE;

    virtual void setHostDirty() CV_OVERRIDE;

    InferenceEngine::DataPtr dataPtr;
    InferenceEngine::Blob::Ptr blob;
    AsyncArray futureMat;
};

InferenceEngine::Blob::Ptr wrapToInfEngineBlob(const Mat& m, InferenceEngine::Layout layout = InferenceEngine::Layout::ANY);

InferenceEngine::Blob::Ptr wrapToInfEngineBlob(const Mat& m, const std::vector<size_t>& shape, InferenceEngine::Layout layout);

InferenceEngine::DataPtr infEngineDataNode(const Ptr<BackendWrapper>& ptr);

// Convert Inference Engine blob with FP32 precision to FP16 precision.
// Allocates memory for a new blob.
InferenceEngine::Blob::Ptr convertFp16(const InferenceEngine::Blob::Ptr& blob);

void addConstantData(const std::string& name, InferenceEngine::Blob::Ptr data, InferenceEngine::Builder::Layer& l);

// This is a fake class to run networks from Model Optimizer. Objects of that
// class simulate responses of layers are imported by OpenCV and supported by
// Inference Engine. The main difference is that they do not perform forward pass.
class InfEngineBackendLayer : public Layer
{
public:
    InfEngineBackendLayer(const InferenceEngine::CNNNetwork &t_net_) : t_net(t_net_) {};

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE;

    virtual void forward(InputArrayOfArrays inputs, OutputArrayOfArrays outputs,
                         OutputArrayOfArrays internals) CV_OVERRIDE;

    virtual bool supportBackend(int backendId) CV_OVERRIDE;

private:
    InferenceEngine::CNNNetwork t_net;
};


class InfEngineExtension : public InferenceEngine::IExtension
{
public:
#if INF_ENGINE_VER_MAJOR_LT(INF_ENGINE_RELEASE_2020_2)
    virtual void SetLogCallback(InferenceEngine::IErrorListener&) noexcept {}
#endif
    virtual void Unload() noexcept {}
    virtual void Release() noexcept {}
    virtual void GetVersion(const InferenceEngine::Version*&) const noexcept {}

    virtual InferenceEngine::StatusCode getPrimitiveTypes(char**&, unsigned int&,
                                                          InferenceEngine::ResponseDesc*) noexcept
    {
        return InferenceEngine::StatusCode::OK;
    }

    InferenceEngine::StatusCode getFactoryFor(InferenceEngine::ILayerImplFactory*& factory,
                                              const InferenceEngine::CNNLayer* cnnLayer,
                                              InferenceEngine::ResponseDesc* resp) noexcept;
};

#endif  // HAVE_DNN_IE_NN_BUILDER_2019


CV__DNN_INLINE_NS_BEGIN

bool isMyriadX();

bool isArmComputePlugin();

CV__DNN_INLINE_NS_END


InferenceEngine::Core& getCore(const std::string& id);

template<typename T = size_t>
static inline std::vector<T> getShape(const Mat& mat)
{
    std::vector<T> result(mat.dims);
    for (int i = 0; i < mat.dims; i++)
        result[i] = (T)mat.size[i];
    return result;
}


#endif  // HAVE_INF_ENGINE

bool haveInfEngine();

void forwardInfEngine(const std::vector<Ptr<BackendWrapper> >& outBlobsWrappers,
                      Ptr<BackendNode>& node, bool isAsync);

}}  // namespace dnn, namespace cv

#endif  // __OPENCV_DNN_OP_INF_ENGINE_HPP__
