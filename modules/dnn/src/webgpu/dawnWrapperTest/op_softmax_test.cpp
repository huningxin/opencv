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
    webgpu::isAvailable();
    float inputData1[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<int> shape = {2,4,1}; // outer_size * channels * channel_size

    webgpu::Tensor input(inputData1, shape, webgpu::Format::wFormatFp32);
    webgpu::Tensor out(nullptr, shape, webgpu::Format::wFormatFp32);

    webgpu::OpSoftmax op1(1, false);
    op1.forward(input, out);

    const void * result = out.getBuffer()->MapReadAsyncAndWait();
    out.unMap();
    printData(result, out.size()/sizeof(float));
    std::cout<<"Finish"<<std::endl;
}