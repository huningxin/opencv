// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"

namespace cv { namespace dnn { namespace webgpu {

// #ifdef HAVE_WEBGPU

// std::vector<uint32_t> compile(const std::string& name,
//                               shaderc_shader_kind kind,
//                               const std::string& data)
// {
//     std::vector<uint32_t> result;
// #ifdef USE_SHADERC
//     shaderc::Compiler compiler;
//     shaderc::CompileOptions options;

//     // Like -DMY_DEFINE=1
//     //options.AddMacroDefinition("MY_DEFINE", "1");
//     options.SetGenerateDebugInfo();
//     options.SetTargetEnvironment(shaderc_target_env_vulkan, 
//                                  shaderc_env_version_vulkan_1_1);
//     shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
//             data.c_str(), data.size(), kind, name.c_str(), options);

//     if (module.GetCompilationStatus() !=
//             shaderc_compilation_status_success) {
//         std::cerr << module.GetErrorMessage();
//     }

//     //std::vector<uint32_t> result(module.cbegin(), module.cend());
//     result.assign(module.cbegin(), module.cend());
//     return result;
// #else
//     assert(0);
//     return result;
// #endif
// }

void bindTensor(Tensor& tensor, uint32_t binding, 
                std::vector<wgpu::BindGroupEntry>& bgEntries) 
{
    wgpu::BindGroupEntry bgEntry = {};
    bgEntry.binding = binding;
    bgEntry.buffer = * tensor.getBuffer()->getWebGPUBuffer();
    bgEntry.offset = 0;
    bgEntry.size = tensor.size();
    bgEntry.sampler = nullptr;
    bgEntry.textureView = nullptr;
    bgEntries.push_back(bgEntry);
}

void bindUniform(Buffer& buffer, uint32_t binding,
                 std::vector<wgpu::BindGroupEntry>& bgEntries)
{
    wgpu::BindGroupEntry bgEntry = {};
    bgEntry.binding = binding;
    bgEntry.buffer = * buffer.getWebGPUBuffer();
    bgEntry.offset = 0;
    bgEntry.size = buffer.getSize();
    bgEntry.sampler = nullptr;
    bgEntry.textureView = nullptr;
    bgEntries.push_back(bgEntry);
}

void computeConvOutputShapeAndPadding(const PaddingMode& padding_mode,
                                      int& padding_top, int& padding_left,
                                      const int& in_h, const int& in_w,
                                      const int& filter_h, const int& filter_w,
                                      const int& dilation_h, const int& dilation_w,
                                      const int& stride_h, const int& stride_w,
                                      int& out_h, int& out_w)
{
    if (padding_mode == wPaddingModeValid)
    {
        padding_top = 0;
        padding_left = 0;
        out_h = ceil((in_h - (filter_h - 1) * dilation_h) / stride_h);
        out_w = ceil((in_w - (filter_w - 1) * dilation_w) / stride_w);
    }
    else if (padding_mode == wPaddingModeSame)
    {
        padding_top = ((filter_h - 1) * dilation_h + 1) / 2;
        padding_left = ((filter_w - 1) * dilation_w + 1) / 2;
        out_h = ceil(in_h / stride_h);
        out_w = ceil(in_w / stride_w);
    }
    else if (padding_mode == wPaddingModeCaffe)
    {
        const int filter_h_actual = dilation_h * (filter_h - 1) + 1;
        const int filter_w_actual = dilation_w * (filter_w - 1) + 1;
        out_h = (in_h + 2 * padding_top - filter_h_actual) / stride_h + 1;
        out_w = (in_w + 2 * padding_left - filter_w_actual) / stride_w + 1;
    }
    else
    {
        CV_Error(Error::StsError, format("Invalid padding mode:%d", padding_mode));
    }
}

void computePoolOutputShape(const PaddingMode& padding_mode,
                            const int& padding_top, const int& padding_left,
                            const int& in_h, const int& in_w,
                            const int& filter_h, const int& filter_w,
                            const int& stride_h, const int& stride_w,
                            int& out_h, int& out_w)
{
    if (padding_mode == wPaddingModeValid)
    {
        assert(padding_top == 0);
        assert(padding_left == 0);
        out_h = ceil((in_h - (filter_h - 1)) / stride_h);
        out_w = ceil((in_h - (filter_w - 1)) / stride_w);
    }
    else if (padding_mode == wPaddingModeSame)
    {
        const int padding_top_ = filter_h / 2;
        const int padding_left_ = filter_w / 2;
        CV_Assert(padding_top == padding_top_);
        CV_Assert(padding_left == padding_left_);
        out_h = ceil(in_h / stride_h);
        out_w = ceil(in_h / stride_w);
    }
    else if (padding_mode == wPaddingModeCaffe)
    {
        int out_h_ = static_cast<int>(ceil(static_cast<float>(
                        in_h + 2 * padding_top - filter_h) / stride_h)) + 1;
        int out_w_ = static_cast<int>(ceil(static_cast<float>(
                        in_h + 2 * padding_left - filter_w) / stride_w)) + 1;

        if (padding_top || padding_left)
        {
            // If we have padding, ensure that the last pooling starts strictly
            // inside the image (instead of at the padding); otherwise clip the last.
            if ((out_h_ - 1) * stride_h >= in_h + padding_top) {
                --out_h_;
            }
            if ((out_w - 1) * stride_h >= in_h + padding_left) {
                --out_w;
            }
            assert((out_h_ - 1) * stride_h < in_h + padding_top);
            assert((out_w_ - 1) * stride_w < in_h + padding_left);
        }
        out_h = out_h_;
        out_w = out_w_;
    }
    else
    {
        CV_Error(Error::StsError, format("Invalid padding mode:%d", padding_mode));
    }
}

// #endif

}}}     //namespace cv::dnn::webgpu

