#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_base.hpp"
#include "../dawn/dawnUtils.hpp"
#include <unistd.h>

namespace cv { namespace dnn { namespace webgpu {
// #ifdef HAVE_WEBGPU
OpBase::OpBase()
{
    createContext();
    device_ = wDevice;
    pipeline_ = nullptr;
    cmd_buffer_ = nullptr;
    bindgrouplayout_ = nullptr;
    bindgroup_ = nullptr;
    module_ = nullptr;
    pipeline_layout_ = nullptr;
}

OpBase::~OpBase()
{
    module_.Release();
    bindgrouplayout_.Release();
    bindgroup_.Release();
    pipeline_.Release();
    pipeline_layout_.Release();
    cmd_buffer_.Release();
}
// the wgpu::BindingType has to be specified  
// UniformBuffer | StorageBuffer | ReadOnlyStorageBuffer | MapReadBuffer
void OpBase::createBindGroupLayout(int buffer_num) 
{
    if(buffer_num <= 0)
        return;
    std::vector<wgpu::BindGroupLayoutEntry> entriesInitializer;
    for(int i = 0; i < buffer_num ; i++) {
        wgpu::BindGroupLayoutEntry entry = {};
        entry.binding = i;
        entry.visibility = wgpu::ShaderStage::Compute;
        entry.type = wgpu::BindingType::StorageBuffer;
        entriesInitializer.push_back(entry);
    }
    if(needsUniform) {
        wgpu::BindGroupLayoutEntry entry = {};
        entry.binding = buffer_num;
        entry.visibility = wgpu::ShaderStage::Compute;
        entry.type = wgpu::BindingType::UniformBuffer;
        entriesInitializer.push_back(entry);
    }
    wgpu::BindGroupLayoutDescriptor descriptor;
    descriptor.entryCount = entriesInitializer.size();
    descriptor.entries = entriesInitializer.data();
    bindgrouplayout_ = device_->CreateBindGroupLayout(&descriptor);
}

void OpBase::createBindGroup()
{
    wgpu::BindGroupDescriptor bgDesc;
    bgDesc.layout = bindgrouplayout_;
    bgDesc.entryCount = bgEntries.size();
    bgDesc.entries = bgEntries.data();
    bindgroup_ = device_->CreateBindGroup(&bgDesc);
}

void OpBase::createShaderModule(const uint32_t* spv, uint32_t size,  
                                const std::string& source) 
{
    wgpu::ShaderModuleSPIRVDescriptor spirvDesc;
    if(spv) {
        spirvDesc.sType = wgpu::SType::ShaderModuleSPIRVDescriptor;
        spirvDesc.codeSize = size;
        spirvDesc.code = spv;
    }
    // TODO: dynamically compile glsl
    // else {
    //     std::vector<uint32_t> code;
    //     code = compile("shader", shaderc_compute_shader, source);
    //     spirvDesc.codeSize =static_cast<uint32_t>(code.size());
    //     spirvDesc.code = code.data();
    // }
    wgpu::ShaderModuleDescriptor descriptor;
    descriptor.label = nullptr;
    descriptor.nextInChain = &spirvDesc;
    module_ = device_->CreateShaderModule(&descriptor);
}

void OpBase::createComputePipeline() 
{
    wgpu::PipelineLayoutDescriptor descriptor;
    descriptor.bindGroupLayoutCount = 1;
    descriptor.bindGroupLayouts = &bindgrouplayout_;
    pipeline_layout_ = device_->CreatePipelineLayout(&descriptor);

    wgpu::ComputePipelineDescriptor csDesc;
    csDesc.layout = pipeline_layout_;
    csDesc.computeStage.module = module_;
    csDesc.computeStage.entryPoint = "main";
    pipeline_ = device_->CreateComputePipeline(&csDesc);
}

void OpBase::createCommandBuffer() 
{
    wgpu::CommandEncoder encoder = device_->CreateCommandEncoder();
    cv::AutoLock lock(wContextMtx);
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pipeline_);
    pass.SetBindGroup(0, bindgroup_);
    pass.Dispatch(group_x_, group_y_, group_z_);
    pass.EndPass();

    cmd_buffer_ = encoder.Finish(); 
}

wgpu::FenceCompletionStatus OpBase::WaitForCompletedValue(
    wgpu::Fence fence, uint64_t completedValue) 
{
    if (fence.GetCompletedValue() < completedValue) 
    {
        device_->Tick();
        usleep(100);
    }
    if(fence.GetCompletedValue() != completedValue)
    {
        return wgpu::FenceCompletionStatus::Error;
    }

    return wgpu::FenceCompletionStatus::Success;
}

void OpBase::runCommandBuffer() 
{
    cv::AutoLock lock(wContextMtx);
    wQueue->Submit(1, &cmd_buffer_);
}

// #endif //HAVE_WEBGPU

}}}  // namsspace cv::dnn::webgpu