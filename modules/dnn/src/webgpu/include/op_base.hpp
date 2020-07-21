// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_WEBGPU_OP_BASE_HPP
#define OPENCV_DNN_WEBGPU_OP_BASE_HPP

#include "../../precomp.hpp"
#include "wgpucom.hpp"
#include "../dawn/dawnUtils.hpp"
#include "tensor.hpp"

namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU
class Context;
class Tensor;
class OpBase
{
public:
    OpBase();
    virtual ~OpBase();
    virtual bool forward(std::vector<Tensor>& ins,
                         std::vector<Tensor>& blobs,
                         std::vector<Tensor>& outs) = 0;

protected:
    void createBindGroupLayout(int buffer_num);
    void createBindGroup();
    void createShaderModule(const uint32_t* spv, 
                            uint32_t size, 
                            const std::string& source = std::string());
    void createComputePipeline();
    void createCommandBuffer();
    void runCommandBuffer();
    wgpu::FenceCompletionStatus WaitForCompletedValue(wgpu::Fence fence, 
                                                      uint64_t completedValue);
    
    std::shared_ptr<wgpu::Device> device_;
    wgpu::ComputePipeline pipeline_;
    wgpu::CommandBuffer cmd_buffer_;
    wgpu::BindGroupLayout bindgrouplayout_;
    wgpu::BindGroup bindgroup_;
    wgpu::ShaderModule module_;
    wgpu::PipelineLayout pipeline_layout_;
    std::vector<wgpu::BindGroupEntry> bgEntries;

    bool needsUniform = true;
    uint32_t group_x_;
    uint32_t group_y_;
    uint32_t group_z_;
    std::string type_;
};
// #endif  //HAVE_WEBGPU

}}}  //namespace cv::dnn::webgpu

#endif  //OPENCV_DNN_WEBGPU_OP_BASE_HPP
