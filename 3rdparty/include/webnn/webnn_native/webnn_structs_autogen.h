
#ifndef WEBNN_NATIVE_WEBNN_STRUCTS_H_
#define WEBNN_NATIVE_WEBNN_STRUCTS_H_

#include "webnn/webnn_cpp.h"
#include "webnn_native/Forward.h"

namespace webnn_native {


    struct ChainedStruct {
        ChainedStruct const * nextInChain = nullptr;
    };

    struct BatchNormOptions {
        OperandBase* scale;
        OperandBase* bias;
        uint32_t axis = 1;
        float epsilon = 1e-05;
    };

    struct ClampOptions {
        OperandBase* minValue;
        OperandBase* maxValue;
    };

    struct CompilationOptions {
        webnn::PowerPreference powerPreference;
    };

    struct Conv2dOptions {
        uint32_t paddingCount = 0;
        int32_t const * padding = nullptr;
        uint32_t stridesCount = 0;
        int32_t const * strides = nullptr;
        uint32_t dilationsCount = 0;
        int32_t const * dilations = nullptr;
        int32_t groups = 1;
        webnn::OperandLayout layout = webnn::OperandLayout::Nchw;
    };

    struct GemmOptions {
        OperandBase* c;
        float alpha = 1.0;
        float beta = 1.0;
        bool aTranspose = false;
        bool bTranspose = false;
    };

    struct Input {
        void const * buffer;
        size_t size;
        int32_t const * dimensions = nullptr;
        uint32_t dimensionsCount = 0;
    };

    struct LeakyReluOptions {
        float alpha = 0.01;
    };

    struct OperandDescriptor {
        webnn::OperandType type;
        int32_t const * dimensions;
        uint32_t dimensionsCount = 0;
    };

    struct Output {
        void * buffer = nullptr;
        size_t size;
        int32_t const * dimensions = nullptr;
        uint32_t dimensionsCount = 0;
    };

    struct Pool2dOptions {
        uint32_t windowDimensionsCount = 0;
        int32_t const * windowDimensions = nullptr;
        uint32_t paddingCount = 0;
        int32_t const * padding = nullptr;
        uint32_t stridesCount = 0;
        int32_t const * strides = nullptr;
        uint32_t dilationsCount = 0;
        int32_t const * dilations = nullptr;
        webnn::OperandLayout layout = webnn::OperandLayout::Nchw;
    };

    struct TransposeOptions {
        uint32_t permutationCount = 0;
        int32_t const * permutation = nullptr;
    };


} // namespace webnn_native

#endif  // WEBNN_NATIVE_WEBNN_STRUCTS_H_
