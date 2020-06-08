// #ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
// #endif
#include <array>
#include <initializer_list>
#include <vector>
namespace cv { namespace dnn { namespace webgpu {

// #ifdef HAVE_WEBGPU

wgpu::Device CreateCppDawnDevice();

wgpu::Buffer CreateBufferFromData(const wgpu::Device& device,
                                    const void* data,
                                    size_t size,
                                    wgpu::BufferUsage usage);

wgpu::CreateBufferMappedResult CreateBufferMappedFromData(const wgpu::Device& device,
                                                          const void* data,
                                                          size_t size,
                                                          wgpu::BufferUsage usage);

template <typename T>
wgpu::Buffer CreateBufferFromData(const wgpu::Device& device,
                                    wgpu::BufferUsage usage,
                                    std::initializer_list<T> data) {
    return CreateBufferFromData(device, data.begin(), uint32_t(sizeof(T) * data.size()), usage);
}

wgpu::PipelineLayout MakeBasicPipelineLayout(const wgpu::Device& device,
                                             const wgpu::BindGroupLayout* bindGroupLayout);

wgpu::BindGroupLayout MakeBindGroupLayout(
    const wgpu::Device& device,
    std::vector<wgpu::BindGroupLayoutEntry> entriesInitializer);

// Helpers to make creating bind groups look nicer:
//
//   utils::MakeBindGroup(device, layout, {
//       {0, mySampler},
//       {1, myBuffer, offset, size},
//       {3, myTextureView}
//   });

// Structure with one constructor per-type of bindings, so that the initializer_list accepts
// bindings with the right type and no extra information.
struct BindingInitializationHelper {
    BindingInitializationHelper(uint32_t binding, const wgpu::Sampler& sampler);
    BindingInitializationHelper(uint32_t binding, const wgpu::TextureView& textureView);
    BindingInitializationHelper(uint32_t binding,
                                const wgpu::Buffer& buffer,
                                uint64_t offset = 0,
                                uint64_t size = wgpu::kWholeSize);

    wgpu::BindGroupEntry GetAsBinding() const;

    uint32_t binding;
    wgpu::Sampler sampler;
    wgpu::TextureView textureView;
    wgpu::Buffer buffer;
    uint64_t offset = 0;
    uint64_t size = 0;
};

wgpu::BindGroup MakeBindGroup(
    const wgpu::Device& device,
    const wgpu::BindGroupLayout& layout,
    std::vector<BindingInitializationHelper> entriesInitializer);

// #endif   //HAVE_WEBGPU

}}}     // namespace cv::dnn::webgpu