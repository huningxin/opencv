#include <dawn/dawn_proc.h>
#include <dawn/dawn_wsi.h>
#include <dawn_native/DawnNative.h>
#include <dawn/webgpu_cpp.h>

#include <memory>
#include <algorithm>
#include <string>
#include <array>
#include <cstring>
#include <random>
#include <iostream>
#include <unistd.h>

// Refer to https://developers.google.com/web/updates/2019/08/get-started-with-gpu-compute-on-the-web
const char* computeShaderCode = R"(#version 450

  layout(std430, set = 0, binding = 0) readonly buffer FirstMatrix {
      vec2 size;
      float numbers[];
  } firstMatrix;

  layout(std430, set = 0, binding = 1) readonly buffer SecondMatrix {
      vec2 size;
      float numbers[];
  } secondMatrix;

  layout(std430, set = 0, binding = 2) buffer ResultMatrix {
      vec2 size;
      float numbers[];
  } resultMatrix;

  void main() {
    resultMatrix.size = vec2(firstMatrix.size.x, secondMatrix.size.y);

    ivec2 resultCell = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
    float result = 0.0;
    for (int i = 0; i < firstMatrix.size.y; i++) {
      int a = i + resultCell.x * int(firstMatrix.size.y);
      int b = resultCell.y + i * int(secondMatrix.size.y);
      result += firstMatrix.numbers[a] * secondMatrix.numbers[b];
    }

    int index = resultCell.y + resultCell.x * int(secondMatrix.size.y);
    resultMatrix.numbers[index] = result;
  }
)";
const uint32_t shaderBuffer[705] = {
    0x07230203,0x00010000,0x00080009,0x0000006a,0x00000000,0x00020011,0x00000001,0x0006000b,
    0x00000001,0x4c534c47,0x6474732e,0x3035342e,0x00000000,0x0003000e,0x00000000,0x00000001,
    0x0006000f,0x00000005,0x00000004,0x6e69616d,0x00000000,0x00000026,0x00060010,0x00000004,
    0x00000011,0x00000001,0x00000001,0x00000001,0x00030003,0x00000002,0x000001c2,0x00040005,
    0x00000004,0x6e69616d,0x00000000,0x00060005,0x00000009,0x75736552,0x614d746c,0x78697274,
    0x00000000,0x00050006,0x00000009,0x00000000,0x657a6973,0x00000000,0x00050006,0x00000009,
    0x00000001,0x626d756e,0x00737265,0x00060005,0x0000000b,0x75736572,0x614d746c,0x78697274,
    0x00000000,0x00050005,0x0000000f,0x73726946,0x74614d74,0x00786972,0x00050006,0x0000000f,
    0x00000000,0x657a6973,0x00000000,0x00050006,0x0000000f,0x00000001,0x626d756e,0x00737265,
    0x00050005,0x00000011,0x73726966,0x74614d74,0x00786972,0x00060005,0x00000018,0x6f636553,
    0x614d646e,0x78697274,0x00000000,0x00050006,0x00000018,0x00000000,0x657a6973,0x00000000,
    0x00050006,0x00000018,0x00000001,0x626d756e,0x00737265,0x00060005,0x0000001a,0x6f636573,
    0x614d646e,0x78697274,0x00000000,0x00050005,0x00000023,0x75736572,0x6543746c,0x00006c6c,
    0x00080005,0x00000026,0x475f6c67,0x61626f6c,0x766e496c,0x7461636f,0x496e6f69,0x00000044,
    0x00040005,0x00000030,0x75736572,0x0000746c,0x00030005,0x00000033,0x00000069,0x00030005,
    0x0000003f,0x00000061,0x00030005,0x00000048,0x00000062,0x00040005,0x0000005d,0x65646e69,
    0x00000078,0x00040047,0x00000008,0x00000006,0x00000004,0x00050048,0x00000009,0x00000000,
    0x00000023,0x00000000,0x00050048,0x00000009,0x00000001,0x00000023,0x00000008,0x00030047,
    0x00000009,0x00000003,0x00040047,0x0000000b,0x00000022,0x00000000,0x00040047,0x0000000b,
    0x00000021,0x00000002,0x00040047,0x0000000e,0x00000006,0x00000004,0x00040048,0x0000000f,
    0x00000000,0x00000018,0x00050048,0x0000000f,0x00000000,0x00000023,0x00000000,0x00040048,
    0x0000000f,0x00000001,0x00000018,0x00050048,0x0000000f,0x00000001,0x00000023,0x00000008,
    0x00030047,0x0000000f,0x00000003,0x00040047,0x00000011,0x00000022,0x00000000,0x00040047,
    0x00000011,0x00000021,0x00000000,0x00040047,0x00000017,0x00000006,0x00000004,0x00040048,
    0x00000018,0x00000000,0x00000018,0x00050048,0x00000018,0x00000000,0x00000023,0x00000000,
    0x00040048,0x00000018,0x00000001,0x00000018,0x00050048,0x00000018,0x00000001,0x00000023,
    0x00000008,0x00030047,0x00000018,0x00000003,0x00040047,0x0000001a,0x00000022,0x00000000,
    0x00040047,0x0000001a,0x00000021,0x00000001,0x00040047,0x00000026,0x0000000b,0x0000001c,
    0x00020013,0x00000002,0x00030021,0x00000003,0x00000002,0x00030016,0x00000006,0x00000020,
    0x00040017,0x00000007,0x00000006,0x00000002,0x0003001d,0x00000008,0x00000006,0x0004001e,
    0x00000009,0x00000007,0x00000008,0x00040020,0x0000000a,0x00000002,0x00000009,0x0004003b,
    0x0000000a,0x0000000b,0x00000002,0x00040015,0x0000000c,0x00000020,0x00000001,0x0004002b,
    0x0000000c,0x0000000d,0x00000000,0x0003001d,0x0000000e,0x00000006,0x0004001e,0x0000000f,
    0x00000007,0x0000000e,0x00040020,0x00000010,0x00000002,0x0000000f,0x0004003b,0x00000010,
    0x00000011,0x00000002,0x00040015,0x00000012,0x00000020,0x00000000,0x0004002b,0x00000012,
    0x00000013,0x00000000,0x00040020,0x00000014,0x00000002,0x00000006,0x0003001d,0x00000017,
    0x00000006,0x0004001e,0x00000018,0x00000007,0x00000017,0x00040020,0x00000019,0x00000002,
    0x00000018,0x0004003b,0x00000019,0x0000001a,0x00000002,0x0004002b,0x00000012,0x0000001b,
    0x00000001,0x00040020,0x0000001f,0x00000002,0x00000007,0x00040017,0x00000021,0x0000000c,
    0x00000002,0x00040020,0x00000022,0x00000007,0x00000021,0x00040017,0x00000024,0x00000012,
    0x00000003,0x00040020,0x00000025,0x00000001,0x00000024,0x0004003b,0x00000025,0x00000026,
    0x00000001,0x00040020,0x00000027,0x00000001,0x00000012,0x00040020,0x0000002f,0x00000007,
    0x00000006,0x0004002b,0x00000006,0x00000031,0x00000000,0x00040020,0x00000032,0x00000007,
    0x0000000c,0x00020014,0x0000003d,0x0004002b,0x0000000c,0x00000051,0x00000001,0x00050036,
    0x00000002,0x00000004,0x00000000,0x00000003,0x000200f8,0x00000005,0x0004003b,0x00000022,
    0x00000023,0x00000007,0x0004003b,0x0000002f,0x00000030,0x00000007,0x0004003b,0x00000032,
    0x00000033,0x00000007,0x0004003b,0x00000032,0x0000003f,0x00000007,0x0004003b,0x00000032,
    0x00000048,0x00000007,0x0004003b,0x00000032,0x0000005d,0x00000007,0x00060041,0x00000014,
    0x00000015,0x00000011,0x0000000d,0x00000013,0x0004003d,0x00000006,0x00000016,0x00000015,
    0x00060041,0x00000014,0x0000001c,0x0000001a,0x0000000d,0x0000001b,0x0004003d,0x00000006,
    0x0000001d,0x0000001c,0x00050050,0x00000007,0x0000001e,0x00000016,0x0000001d,0x00050041,
    0x0000001f,0x00000020,0x0000000b,0x0000000d,0x0003003e,0x00000020,0x0000001e,0x00050041,
    0x00000027,0x00000028,0x00000026,0x00000013,0x0004003d,0x00000012,0x00000029,0x00000028,
    0x0004007c,0x0000000c,0x0000002a,0x00000029,0x00050041,0x00000027,0x0000002b,0x00000026,
    0x0000001b,0x0004003d,0x00000012,0x0000002c,0x0000002b,0x0004007c,0x0000000c,0x0000002d,
    0x0000002c,0x00050050,0x00000021,0x0000002e,0x0000002a,0x0000002d,0x0003003e,0x00000023,
    0x0000002e,0x0003003e,0x00000030,0x00000031,0x0003003e,0x00000033,0x0000000d,0x000200f9,
    0x00000034,0x000200f8,0x00000034,0x000400f6,0x00000036,0x00000037,0x00000000,0x000200f9,
    0x00000038,0x000200f8,0x00000038,0x0004003d,0x0000000c,0x00000039,0x00000033,0x0004006f,
    0x00000006,0x0000003a,0x00000039,0x00060041,0x00000014,0x0000003b,0x00000011,0x0000000d,
    0x0000001b,0x0004003d,0x00000006,0x0000003c,0x0000003b,0x000500b8,0x0000003d,0x0000003e,
    0x0000003a,0x0000003c,0x000400fa,0x0000003e,0x00000035,0x00000036,0x000200f8,0x00000035,
    0x0004003d,0x0000000c,0x00000040,0x00000033,0x00050041,0x00000032,0x00000041,0x00000023,
    0x00000013,0x0004003d,0x0000000c,0x00000042,0x00000041,0x00060041,0x00000014,0x00000043,
    0x00000011,0x0000000d,0x0000001b,0x0004003d,0x00000006,0x00000044,0x00000043,0x0004006e,
    0x0000000c,0x00000045,0x00000044,0x00050084,0x0000000c,0x00000046,0x00000042,0x00000045,
    0x00050080,0x0000000c,0x00000047,0x00000040,0x00000046,0x0003003e,0x0000003f,0x00000047,
    0x00050041,0x00000032,0x00000049,0x00000023,0x0000001b,0x0004003d,0x0000000c,0x0000004a,
    0x00000049,0x0004003d,0x0000000c,0x0000004b,0x00000033,0x00060041,0x00000014,0x0000004c,
    0x0000001a,0x0000000d,0x0000001b,0x0004003d,0x00000006,0x0000004d,0x0000004c,0x0004006e,
    0x0000000c,0x0000004e,0x0000004d,0x00050084,0x0000000c,0x0000004f,0x0000004b,0x0000004e,
    0x00050080,0x0000000c,0x00000050,0x0000004a,0x0000004f,0x0003003e,0x00000048,0x00000050,
    0x0004003d,0x0000000c,0x00000052,0x0000003f,0x00060041,0x00000014,0x00000053,0x00000011,
    0x00000051,0x00000052,0x0004003d,0x00000006,0x00000054,0x00000053,0x0004003d,0x0000000c,
    0x00000055,0x00000048,0x00060041,0x00000014,0x00000056,0x0000001a,0x00000051,0x00000055,
    0x0004003d,0x00000006,0x00000057,0x00000056,0x00050085,0x00000006,0x00000058,0x00000054,
    0x00000057,0x0004003d,0x00000006,0x00000059,0x00000030,0x00050081,0x00000006,0x0000005a,
    0x00000059,0x00000058,0x0003003e,0x00000030,0x0000005a,0x000200f9,0x00000037,0x000200f8,
    0x00000037,0x0004003d,0x0000000c,0x0000005b,0x00000033,0x00050080,0x0000000c,0x0000005c,
    0x0000005b,0x00000051,0x0003003e,0x00000033,0x0000005c,0x000200f9,0x00000034,0x000200f8,
    0x00000036,0x00050041,0x00000032,0x0000005e,0x00000023,0x0000001b,0x0004003d,0x0000000c,
    0x0000005f,0x0000005e,0x00050041,0x00000032,0x00000060,0x00000023,0x00000013,0x0004003d,
    0x0000000c,0x00000061,0x00000060,0x00060041,0x00000014,0x00000062,0x0000001a,0x0000000d,
    0x0000001b,0x0004003d,0x00000006,0x00000063,0x00000062,0x0004006e,0x0000000c,0x00000064,
    0x00000063,0x00050084,0x0000000c,0x00000065,0x00000061,0x00000064,0x00050080,0x0000000c,
    0x00000066,0x0000005f,0x00000065,0x0003003e,0x0000005d,0x00000066,0x0004003d,0x0000000c,
    0x00000067,0x0000005d,0x0004003d,0x00000006,0x00000068,0x00000030,0x00060041,0x00000014,
    0x00000069,0x0000000b,0x00000051,0x00000067,0x0003003e,0x00000069,0x00000068,0x000100fd,
    0x00010038
};

void PrintDeviceError(WGPUErrorType errorType,  const char* message, void*) {
    std::string errorTypeName = "";
    switch (errorType) {
        case WGPUErrorType_Validation:
            errorTypeName = "WGPUErrorTyp Validation";
            break;
        case WGPUErrorType_OutOfMemory:
            errorTypeName = "WGPUErrorTyp Out of memory";
            break;
        case WGPUErrorType_Unknown:
            errorTypeName = "WGPUErrorTyp Unknown";
            break;
        case WGPUErrorType_DeviceLost:
            errorTypeName = "WGPUErrorTyp Device lost";
            break;
        default :
            errorTypeName = "WGPUErrorTyp Unknown";
            return;
    }
    std::cout<<"Device error:" << errorTypeName<<std::endl;
}

wgpu::Device createCppDawnDevice() {
    static wgpu::BackendType backendType = wgpu::BackendType::Vulkan;
    static std::unique_ptr<dawn_native::Instance> instance;
    instance = std::make_unique<dawn_native::Instance>();
    instance->DiscoverDefaultAdapters();
    // Get an adapter for the backend to use, and create the device.
    dawn_native::Adapter backendAdapter;
    {
        std::vector<dawn_native::Adapter> adapters = instance->GetAdapters();
        auto adapterIt = std::find_if(adapters.begin(), adapters.end(),
                                    [](const dawn_native::Adapter adapter) -> bool {
                                        wgpu::AdapterProperties properties;
                                        adapter.GetProperties(&properties);
                                        return properties.backendType == backendType;
                                    });
        backendAdapter = *adapterIt;
    }
    WGPUDevice backendDevice = backendAdapter.CreateDevice();
    DawnProcTable backendProcs = dawn_native::GetProcs();
    dawnProcSetProcs(&backendProcs);
    backendProcs.deviceSetUncapturedErrorCallback(backendDevice, PrintDeviceError, nullptr);
    return wgpu::Device::Acquire(backendDevice);
}

wgpu::CreateBufferMappedResult createBufferMappedFromData(wgpu::Device& device,
                                                        const void* data,
                                                        size_t size,
                                                        wgpu::BufferUsage usage){
    wgpu::BufferDescriptor descriptor = {};
    descriptor.size = size;
    descriptor.usage = usage | wgpu::BufferUsage::CopyDst;
    wgpu::CreateBufferMappedResult result = device.CreateBufferMapped(&descriptor);
    memcpy(result.data, data, size);
    return result;
}

void readBufferMapReadCallback(WGPUBufferMapAsyncStatus status,
	const void* ptr,
	uint64_t dataLength,
	void* userdata) {
	(void)status;
	(void)userdata;
	printf("resultMatrix: [");
	size_t num = dataLength / sizeof(float);
	for (size_t i = 0; i < num; ++i) {
		printf("%f", ((const float*)ptr)[i]);
		if (i != num - 1)
			printf(", ");
	}
	printf("]\n");
}

int main(int /*argc*/, char** /*argv*/) 
{
    wgpu::Device device = createCppDawnDevice();
    wgpu::Queue queue = device.GetDefaultQueue();
    // create buffer
    wgpu::Buffer inputsBuffer1, inputsBuffer2;
    wgpu::Buffer resultBuffer,gpuReadBuffer;
    float const inputData1[] = {	2 /* rows */, 4 /* columns */,
	1, 2, 3, 4, 5, 6, 7, 8};
    float const inputData2[] = {	4 /* rows */, 2 /* columns */,
	1, 2, 3, 4, 5, 6, 7, 8};
    
    wgpu::BufferDescriptor desc0 = {};
    desc0.size = sizeof(inputData1);
    desc0.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    inputsBuffer1 =device.CreateBuffer(&desc0);
    inputsBuffer1.SetSubData(0, sizeof(inputData1), inputData1);

    wgpu::BufferDescriptor desc1 = {};
    desc1.size = sizeof(inputData2);
    desc1.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst;
    inputsBuffer2 =device.CreateBuffer(&desc1);
    inputsBuffer2.SetSubData(0, sizeof(inputData2), inputData2);

    wgpu::BufferDescriptor desc2 = {};
    desc2.size = sizeof(float) * 6;
    desc2.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc;
    resultBuffer = device.CreateBuffer(& desc2);

    wgpu::BufferDescriptor desc3 = {};
    desc3.size = sizeof(float) * 6;
    desc3.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;
    gpuReadBuffer = device.CreateBuffer(& desc3);

    std::cout << "buffer create succeed" <<std::endl;

    // create bindgroup
    std::vector<wgpu::BindGroupLayoutEntry> bglEntries {
        {0, wgpu::ShaderStage::Compute, wgpu::BindingType::ReadonlyStorageBuffer},
        {1, wgpu::ShaderStage::Compute, wgpu::BindingType::ReadonlyStorageBuffer},
        {2, wgpu::ShaderStage::Compute, wgpu::BindingType::StorageBuffer}};
    wgpu::BindGroupLayoutDescriptor bglDesc;
    bglDesc.entryCount = static_cast<uint32_t>(bglEntries.size());
    bglDesc.entries = bglEntries.data();
    wgpu::BindGroupLayout bgl = device.CreateBindGroupLayout(&bglDesc);

    std::vector<wgpu::BindGroupEntry> bgEntries {
        { 0, inputsBuffer1, 0, sizeof(inputData1), nullptr, nullptr },
        { 1, inputsBuffer2, 0, sizeof(inputData2), nullptr, nullptr },
        { 2, resultBuffer, 0, sizeof(float) * 6, nullptr, nullptr } };
    wgpu::BindGroupDescriptor bgDesc;
    bgDesc.layout = bgl;
    bgDesc.entryCount = bgEntries.size();
    bgDesc.entries = bgEntries.data();
    wgpu::BindGroup bg = device.CreateBindGroup(&bgDesc);
    std::cout<<"create BindGroup succeed"<<std::endl;

    // create shader module
    uint32_t shaderSize =sizeof(shaderBuffer) / sizeof(uint32_t);
    wgpu::ShaderModuleSPIRVDescriptor spirvDesc;
    spirvDesc.sType = wgpu::SType::ShaderModuleSPIRVDescriptor;
    spirvDesc.codeSize = shaderSize;
    spirvDesc.code = shaderBuffer;
    wgpu::ShaderModuleDescriptor descriptor;
    descriptor.label = nullptr;
    descriptor.nextInChain = &spirvDesc;
    wgpu::ShaderModule module = device.CreateShaderModule(&descriptor);
    std::cout<< "Shader module create succeed"<<std::endl;

    //create pipeline
    wgpu::PipelineLayoutDescriptor pplDesc;
    pplDesc.bindGroupLayoutCount = 1;
    pplDesc.bindGroupLayouts = &bgl;
    wgpu::PipelineLayout pplLayout = device.CreatePipelineLayout(&pplDesc);
    wgpu::ComputePipelineDescriptor csDesc;
    csDesc.layout = pplLayout;
    csDesc.computeStage.module = module;
    csDesc.computeStage.entryPoint = "main";
    wgpu::ComputePipeline pl = device.CreateComputePipeline(&csDesc);
    std::cout<<"ComputePipeline create succeed"<<std::endl;

    // create command buffer
    wgpu::CommandEncoder encoder = device.CreateCommandEncoder();
    wgpu::ComputePassEncoder pass = encoder.BeginComputePass();
    pass.SetPipeline(pl);
    pass.SetBindGroup(0,bg);
    pass.Dispatch(2,2);
    pass.EndPass();
    encoder.CopyBufferToBuffer(resultBuffer, 0, gpuReadBuffer, 0, sizeof(float) * 6);
    wgpu::CommandBuffer cmdBuffer = encoder.Finish();
    std::cout<<"CommandBuffer create succeed"<<std::endl;

    // create fence and submit cmdBuffer
    queue.Submit(1, &cmdBuffer);
    usleep(100);
    std::cout<<"cmd_buffer submitted"<<std::endl;
    gpuReadBuffer.MapReadAsync(readBufferMapReadCallback, nullptr);
    // usleep(100);
    // bg.Release();
    // inputsBuffer1.Release();
    // inputsBuffer2.Release();
    // pl.Release();
    // queue.Release();
    // device.Release();
    std::cout<<"test1 finish"<<std::endl;
}