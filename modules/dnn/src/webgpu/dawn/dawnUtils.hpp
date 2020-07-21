// #ifdef HAVE_WEBGPU
#include <dawn/webgpu_cpp.h>
#include <dawn/dawn_proc.h>
#include <dawn/dawn_wsi.h>
#include <dawn_native/DawnNative.h>
// #endif  // HAVE_WEBGPU
#include <array>
#include <initializer_list>
#include <vector>
namespace cv { namespace dnn { namespace webgpu {

#ifdef HAVE_WEBGPU

wgpu::Device createCppDawnDevice();

wgpu::Buffer CreateBufferFromData(const wgpu::Device& device,
                                    const void* data,
                                    size_t size,
                                    wgpu::BufferUsage usage);

wgpu::CreateBufferMappedResult CreateBufferMappedFromData(const wgpu::Device& device,
                                                          const void* data,
                                                          size_t size,
                                                          wgpu::BufferUsage usage);

wgpu::PipelineLayout MakeBasicPipelineLayout(const wgpu::Device& device,
                                             const wgpu::BindGroupLayout* bindGroupLayout);

wgpu::BindGroupLayout MakeBindGroupLayout(
    const wgpu::Device& device,
    std::vector<wgpu::BindGroupLayoutEntry> entriesInitializer);

#endif   //HAVE_WEBGPU

}}}     // namespace cv::dnn::webgpu