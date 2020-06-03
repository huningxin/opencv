// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "context.hpp"
#include<dawn/webgpu_cpp.h>
#include "../dawn/dawnUtils.h"
namespace cv { namespace dnn { namespace webgpu {
    std::shared_ptr<Context> wCtx;
    std::shared_ptr<Context> wContext;
    bool enableValidationLayers = false;

    wgpu::Device wDevice;
    wgpu::CommandEncoder encoder;
    wgpu::Queue wQueue;

    cv::Mutex wContextMtx;
    // internally used
    void createContext()
    {
        cv::AutoLock lock(wContextMtx);
        if (!wCtx)
        {
            wCtx.reset(new Context());
        }
    }

    bool isAvailable()
    {
        try
        {
            createContext();
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_ERROR(NULL, "Failed to init Vulkan environment. " << e.what());
            return false;
        }

        return true;
    }
    Context::Context() {
        //create wgpu::Instance and wgpu::Device
        wDevice = createCppDawnDevice();
        wQueue = wDevice.GetDefaultQueue();
        encoder = wDevice.CreateCommandEncoder();
        /* WebGPU 
        */
    }
    Context::~Context() {

    }

}}}