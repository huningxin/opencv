#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../include/op_softmax.hpp"
#include "../src/context.hpp"
#include "../include/wgpucom.hpp"
#include <unistd.h>
#include "../src/common.hpp"
#include <stdlib.h>
using namespace cv::dnn;

int main(int argc, char** argv )
{
    webgpu::wDevice = std::make_shared<wgpu::Device>(webgpu::createCppDawnDevice());
    webgpu::wQueue = std::make_shared<wgpu::Queue>(webgpu::wDevice->GetDefaultQueue());
    float inputData1[] = {
	1, 2, 3, 4, 5, 6, 7, 8};
    float maxData[] = {4, 8};
    float sumData[] = {10, 26};

    std::vector<int> shape = {2,4,1}, shape2 = {2, 1}; // outer_size * channels * channel_size
    webgpu::Tensor input(inputData1, shape, wgpu::BufferUsage::Storage, webgpu::Format::wFormatFp32);
    webgpu::Tensor out(inputData1, shape, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, webgpu::Format::wFormatFp32);
    std::vector<const void *> blobs = {maxData, sumData};

    webgpu::OpSoftmax op1(1, false);
    op1.setBlobs(blobs, shape2);
    op1.forward(input, out);
    
    // out.getBuffer()->getWebGPUBuffer()->Unmap();
    const void * result = out.getBuffer()->MapReadAsyncAndWait();

    for(int i = 0; i < 10; i ++) {
        webgpu::wDevice->Tick();
        usleep(100);
    } 

    printf("result: [");
	size_t num = out.size() / sizeof(float);
	for (size_t i = 0; i < num; ++i) {
		printf("%f", ((const float*)result)[i]);
		if (i != num - 1)
			printf(", ");
	}
	printf("]\n");

    std::cout<<"Finish"<<std::endl;
}