#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "../include/op_softmax.hpp"
#include "../src/context.hpp"
#include "../include/wgpucom.hpp"
#include <unistd.h>
#include "../src/common.hpp"
#include <stdlib.h>
using namespace cv::dnn;
void printData(const void * data, int num) {
    printf("resultMatrix: [");
	for (size_t i = 0; i < num; ++i) {
		printf("%f", ((const float*)data)[i]);
		if (i != num - 1)
			printf(", ");
	}
	printf("]\n");
}
int main(int argc, char** argv )
{
    webgpu::wDevice = std::make_shared<wgpu::Device>(webgpu::createCppDawnDevice());
    webgpu::wQueue = std::make_shared<wgpu::Queue>(webgpu::wDevice->GetDefaultQueue());
    float inputData1[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> shape = {2,4,1}; // outer_size * channels * channel_size

    webgpu::Tensor input(inputData1, shape, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopyDst, webgpu::Format::wFormatFp32);
    webgpu::Tensor out(nullptr, shape, wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc, webgpu::Format::wFormatFp32);

    webgpu::OpSoftmax op1(1, false);
    op1.forward(input, out);
    
    const void * maxdata = op1.max_tensor_->getBuffer()->MapReadAsyncAndWait();
    printData(maxdata, op1.max_tensor_->size()/sizeof(float));
    const void * sumdata = op1.sum_tensor_->getBuffer()->MapReadAsyncAndWait();
    printData(sumdata, op1.sum_tensor_->size()/sizeof(float));
    const void * result = out.getBuffer()->MapReadAsyncAndWait();
    printData(result, out.size()/sizeof(float));
    std::cout<<"Finish"<<std::endl;
}