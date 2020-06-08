#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_base.hpp"
#include "../dawn/dawnUtils.hpp"
#include <unistd.h>

namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU
OpBase::OpBase(){
    createContext();
    device_ = wDevice;
    pipeline_ = nullptr;
    cmd_buffer_ = nullptr;
    bindgrouplayout_ = nullptr;
    bindgroup_ = nullptr;
    module_ = nullptr;
    pipeline_layout_ = nullptr;
}

OpBase::~OpBase(){
    module_.Release();
    bindgrouplayout_.Release();
    bindgroup_.Release();
    pipeline_.Release();
    pipeline_layout_.Release();
}
// the wgpu::BindingType has to be specified  UniformBuffer | StorageBuffer | ReadOnlyStorageBuffer
void OpBase::createBindGroupLayout(int buffer_num) {
    if(buffer_num <= 0)
        return;
    std::vector<wgpu::BindGroupLayoutEntry> entriesInitializer;
    for(int i = 0; i < buffer_num; i++) {
        entriesInitializer.push_back({i, wgpu::ShaderStage::Compute, wgpu::BindingType::StorageBuffer});
    }
    bindgrouplayout_ = MakeBindGroupLayout(device_, entriesInitializer);
}

void OpBase::createShaderModule(const uint32_t* spv, size_t sz, const std::string& source = std::string()) {
    wgpu::ShaderModuleSPIRVDescriptor spirvDesc;
    if(spv) {
        spirvDesc.codeSize =static_cast<uint32_t>(sz / sizeof(uint32_t));
        spirvDesc.code = spv;
    }
    else {
        std::vector<uint32_t> code;
        code = compile("shader", shaderc_compute_shader, source);
        spirvDesc.codeSize =static_cast<uint32_t>(code.size());
        spirvDesc.code = code.data();
    }
    wgpu::ShaderModuleDescriptor descriptor;
    descriptor.nextInChain = &spirvDesc;

    module_ = device_.CreateShaderModule(&descriptor);
}

void OpBase::createPipeline() 
{
    pipeline_layout_ = MakeBasicPipelineLayout(device_, &bindgrouplayout_);

    wgpu::ComputePipelineDescriptor csDesc;
    csDesc.layout = pipeline_layout_;
    csDesc.computeStage.module = module_;
    csDesc.computeStage.entryPoint = "main";
    pipeline_ = device_.CreateComputePipeline(&csDesc);
}

void OpBase::createCommandBuffer() 
{
    wgpu::CommandEncoder encoder = device_.CreateCommandEncoder();
    cv::AutoLock lock(wContextMtx);
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline_);
    pass.SetBindGroup(0, bindgroup_);
    pass.Dispatch(group_x_, group_y_, group_z_);
    pass.EndPass();

    cmd_buffer_ = encoder.Finish(); 
}

wgpu::FenceCompletionStatus OpBase::WaitForCompletedValue(wgpu::Fence fence, uint64_t completedValue) 
{
    if (fence.GetCompletedValue() < completedValue) 
    {
        device_.Tick();
        usleep(100000000000);
    }
    if(fence.GetCompletedValue() != completedValue)
    {
        return wgpu::FenceCompletionStatus::Error;
    }

    return wgpu::FenceCompletionStatus::Success;
}

void OpBase::runCommandBuffer() {
    wgpu::FenceDescriptor descriptor;
    descriptor.initialValue = 1u;
    wgpu::Fence fence = wQueue.CreateFence(&descriptor);
    cv::AutoLock lock(wContextMtx);
    wQueue.Signal(fence, 1);
    wQueue.Submit(1, &cmd_buffer_);
    if(WaitForCompletedValue(fence, 1u) == wgpu::FenceCompletionStatus::Error ) {
            CV_LOG_ERROR(NULL, "WGPU Fence Failed "); 
            CV_Error(Error::StsError, "Vulkan check failed"); 
    }
    // queue submit succeed
    fence.Release();
}

// #endif //HAVE_WEBGPU

}}}  // namsspace cv::dnn::webgpu