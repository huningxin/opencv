// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef WEBNN_H_
#define WEBNN_H_

#if defined(WEBNN_SHARED_LIBRARY)
#    if defined(_WIN32)
#        if defined(WEBNN_IMPLEMENTATION)
#            define WEBNN_EXPORT __declspec(dllexport)
#        else
#            define WEBNN_EXPORT __declspec(dllimport)
#        endif
#    else  // defined(_WIN32)
#        if defined(WEBNN_IMPLEMENTATION)
#            define WEBNN_EXPORT __attribute__((visibility("default")))
#        else
#            define WEBNN_EXPORT
#        endif
#    endif  // defined(_WIN32)
#else       // defined(WEBNN_SHARED_LIBRARY)
#    define WEBNN_EXPORT
#endif  // defined(WEBNN_SHARED_LIBRARY)

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

typedef uint32_t WEBNNFlags;

typedef struct WebnnCompilationImpl* WebnnCompilation;
typedef struct WebnnModelImpl* WebnnModel;
typedef struct WebnnModelBuilderImpl* WebnnModelBuilder;
typedef struct WebnnNamedInputsImpl* WebnnNamedInputs;
typedef struct WebnnNamedOperandsImpl* WebnnNamedOperands;
typedef struct WebnnNamedOutputsImpl* WebnnNamedOutputs;
typedef struct WebnnNamedResultsImpl* WebnnNamedResults;
typedef struct WebnnNeuralNetworkContextImpl* WebnnNeuralNetworkContext;
typedef struct WebnnOperandImpl* WebnnOperand;
typedef struct WebnnResultImpl* WebnnResult;

typedef enum WebnnCompileStatus {
    WebnnCompileStatus_Success = 0x00000000,
    WebnnCompileStatus_Error = 0x00000001,
    WebnnCompileStatus_ContextLost = 0x00000002,
    WebnnCompileStatus_Unknown = 0x00000003,
    WebnnCompileStatus_Force32 = 0x7FFFFFFF
} WebnnCompileStatus;

typedef enum WebnnComputeStatus {
    WebnnComputeStatus_Success = 0x00000000,
    WebnnComputeStatus_Error = 0x00000001,
    WebnnComputeStatus_ContextLost = 0x00000002,
    WebnnComputeStatus_Unknown = 0x00000003,
    WebnnComputeStatus_Force32 = 0x7FFFFFFF
} WebnnComputeStatus;

typedef enum WebnnErrorFilter {
    WebnnErrorFilter_None = 0x00000000,
    WebnnErrorFilter_Validation = 0x00000001,
    WebnnErrorFilter_OutOfMemory = 0x00000002,
    WebnnErrorFilter_Force32 = 0x7FFFFFFF
} WebnnErrorFilter;

typedef enum WebnnErrorType {
    WebnnErrorType_NoError = 0x00000000,
    WebnnErrorType_Validation = 0x00000001,
    WebnnErrorType_OutOfMemory = 0x00000002,
    WebnnErrorType_Unknown = 0x00000003,
    WebnnErrorType_DeviceLost = 0x00000004,
    WebnnErrorType_Force32 = 0x7FFFFFFF
} WebnnErrorType;

typedef enum WebnnOperandLayout {
    WebnnOperandLayout_Nchw = 0x00000000,
    WebnnOperandLayout_Nhwc = 0x00000001,
    WebnnOperandLayout_Force32 = 0x7FFFFFFF
} WebnnOperandLayout;

typedef enum WebnnOperandType {
    WebnnOperandType_Float32 = 0x00000000,
    WebnnOperandType_Float16 = 0x00000001,
    WebnnOperandType_Int32 = 0x00000002,
    WebnnOperandType_Uint32 = 0x00000003,
    WebnnOperandType_Force32 = 0x7FFFFFFF
} WebnnOperandType;

typedef enum WebnnPowerPreference {
    WebnnPowerPreference_Default = 0x00000000,
    WebnnPowerPreference_Low_power = 0x00000001,
    WebnnPowerPreference_High_performance = 0x00000002,
    WebnnPowerPreference_Force32 = 0x7FFFFFFF
} WebnnPowerPreference;


typedef struct WebnnBatchNormOptions {
    WebnnOperand scale;
    WebnnOperand bias;
    uint32_t axis;
    float epsilon;
} WebnnBatchNormOptions;

typedef struct WebnnClampOptions {
    WebnnOperand minValue;
    WebnnOperand maxValue;
} WebnnClampOptions;

typedef struct WebnnCompilationOptions {
    WebnnPowerPreference powerPreference;
} WebnnCompilationOptions;

typedef struct WebnnConv2dOptions {
    uint32_t paddingCount;
    int32_t const * padding;
    uint32_t stridesCount;
    int32_t const * strides;
    uint32_t dilationsCount;
    int32_t const * dilations;
    int32_t groups;
    WebnnOperandLayout layout;
} WebnnConv2dOptions;

typedef struct WebnnGemmOptions {
    WebnnOperand c;
    float alpha;
    float beta;
    bool aTranspose;
    bool bTranspose;
} WebnnGemmOptions;

typedef struct WebnnInput {
    void const * buffer;
    size_t size;
    int32_t const * dimensions;
    uint32_t dimensionsCount;
} WebnnInput;

typedef struct WebnnLeakyReluOptions {
    float alpha;
} WebnnLeakyReluOptions;

typedef struct WebnnOperandDescriptor {
    WebnnOperandType type;
    int32_t const * dimensions;
    uint32_t dimensionsCount;
} WebnnOperandDescriptor;

typedef struct WebnnOutput {
    void * buffer;
    size_t size;
    int32_t const * dimensions;
    uint32_t dimensionsCount;
} WebnnOutput;

typedef struct WebnnPool2dOptions {
    uint32_t windowDimensionsCount;
    int32_t const * windowDimensions;
    uint32_t paddingCount;
    int32_t const * padding;
    uint32_t stridesCount;
    int32_t const * strides;
    uint32_t dilationsCount;
    int32_t const * dilations;
    WebnnOperandLayout layout;
} WebnnPool2dOptions;

typedef struct WebnnTransposeOptions {
    uint32_t permutationCount;
    int32_t const * permutation;
} WebnnTransposeOptions;


#ifdef __cplusplus
extern "C" {
#endif

typedef void (*WebnnCompileCallback)(WebnnCompileStatus status, WebnnCompilation compilation, char const * message, void * userdata);
typedef void (*WebnnComputeCallback)(WebnnComputeStatus status, WebnnNamedResults outputs, char const * message, void * userdata);
typedef void (*WebnnErrorCallback)(WebnnErrorType type, char const * message, void * userdata);

typedef void (*WebnnProc)(void);

#if !defined(WEBNN_SKIP_PROCS)

typedef WebnnNamedInputs (*WebnnProcCreateNamedInputs)();
typedef WebnnNamedOperands (*WebnnProcCreateNamedOperands)();
typedef WebnnNamedOutputs (*WebnnProcCreateNamedOutputs)();

// Procs of Compilation
typedef void (*WebnnProcCompilationCompute)(WebnnCompilation compilation, WebnnNamedInputs inputs, WebnnComputeCallback callback, void * userdata, WebnnNamedOutputs outputs);
typedef void (*WebnnProcCompilationReference)(WebnnCompilation compilation);
typedef void (*WebnnProcCompilationRelease)(WebnnCompilation compilation);

// Procs of Model
typedef void (*WebnnProcModelCompile)(WebnnModel model, WebnnCompileCallback callback, void * userdata, WebnnCompilationOptions const * options);
typedef void (*WebnnProcModelReference)(WebnnModel model);
typedef void (*WebnnProcModelRelease)(WebnnModel model);

// Procs of ModelBuilder
typedef WebnnOperand (*WebnnProcModelBuilderAdd)(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b);
typedef WebnnOperand (*WebnnProcModelBuilderAveragePool2d)(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options);
typedef WebnnOperand (*WebnnProcModelBuilderBatchNorm)(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand mean, WebnnOperand variance, WebnnBatchNormOptions const * options);
typedef WebnnOperand (*WebnnProcModelBuilderClamp)(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnClampOptions const * options);
typedef WebnnOperand (*WebnnProcModelBuilderConcat)(WebnnModelBuilder modelBuilder, uint32_t inputsCount, WebnnOperand const * inputs, uint32_t axis);
typedef WebnnOperand (*WebnnProcModelBuilderConstant)(WebnnModelBuilder modelBuilder, WebnnOperandDescriptor const * desc, void const * value, size_t size);
typedef WebnnOperand (*WebnnProcModelBuilderConv2d)(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand filter, WebnnConv2dOptions const * options);
typedef WebnnModel (*WebnnProcModelBuilderCreateModel)(WebnnModelBuilder modelBuilder, WebnnNamedOperands namedOperands);
typedef WebnnOperand (*WebnnProcModelBuilderGemm)(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b, WebnnGemmOptions const * options);
typedef WebnnOperand (*WebnnProcModelBuilderInput)(WebnnModelBuilder modelBuilder, char const * name, WebnnOperandDescriptor const * desc);
typedef WebnnOperand (*WebnnProcModelBuilderLeakyRelu)(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnLeakyReluOptions const * options);
typedef WebnnOperand (*WebnnProcModelBuilderMatmul)(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b);
typedef WebnnOperand (*WebnnProcModelBuilderMaxPool2d)(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options);
typedef WebnnOperand (*WebnnProcModelBuilderMul)(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b);
typedef WebnnOperand (*WebnnProcModelBuilderRelu)(WebnnModelBuilder modelBuilder, WebnnOperand input);
typedef WebnnOperand (*WebnnProcModelBuilderReshape)(WebnnModelBuilder modelBuilder, WebnnOperand input, int32_t const * newShape, uint32_t newShapeCount);
typedef WebnnOperand (*WebnnProcModelBuilderSoftmax)(WebnnModelBuilder modelBuilder, WebnnOperand input);
typedef WebnnOperand (*WebnnProcModelBuilderTranspose)(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnTransposeOptions const * options);
typedef void (*WebnnProcModelBuilderReference)(WebnnModelBuilder modelBuilder);
typedef void (*WebnnProcModelBuilderRelease)(WebnnModelBuilder modelBuilder);

// Procs of NamedInputs
typedef void (*WebnnProcNamedInputsSet)(WebnnNamedInputs namedInputs, char const * name, WebnnInput const * input);
typedef void (*WebnnProcNamedInputsReference)(WebnnNamedInputs namedInputs);
typedef void (*WebnnProcNamedInputsRelease)(WebnnNamedInputs namedInputs);

// Procs of NamedOperands
typedef void (*WebnnProcNamedOperandsSet)(WebnnNamedOperands namedOperands, char const * name, WebnnOperand operand);
typedef void (*WebnnProcNamedOperandsReference)(WebnnNamedOperands namedOperands);
typedef void (*WebnnProcNamedOperandsRelease)(WebnnNamedOperands namedOperands);

// Procs of NamedOutputs
typedef void (*WebnnProcNamedOutputsSet)(WebnnNamedOutputs namedOutputs, char const * name, WebnnOutput const * output);
typedef void (*WebnnProcNamedOutputsReference)(WebnnNamedOutputs namedOutputs);
typedef void (*WebnnProcNamedOutputsRelease)(WebnnNamedOutputs namedOutputs);

// Procs of NamedResults
typedef WebnnResult (*WebnnProcNamedResultsGet)(WebnnNamedResults namedResults, char const * name);
typedef void (*WebnnProcNamedResultsReference)(WebnnNamedResults namedResults);
typedef void (*WebnnProcNamedResultsRelease)(WebnnNamedResults namedResults);

// Procs of NeuralNetworkContext
typedef WebnnModelBuilder (*WebnnProcNeuralNetworkContextCreateModelBuilder)(WebnnNeuralNetworkContext neuralNetworkContext);
typedef bool (*WebnnProcNeuralNetworkContextPopErrorScope)(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata);
typedef void (*WebnnProcNeuralNetworkContextPushErrorScope)(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorFilter filter);
typedef void (*WebnnProcNeuralNetworkContextSetUncapturedErrorCallback)(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata);
typedef void (*WebnnProcNeuralNetworkContextReference)(WebnnNeuralNetworkContext neuralNetworkContext);
typedef void (*WebnnProcNeuralNetworkContextRelease)(WebnnNeuralNetworkContext neuralNetworkContext);

// Procs of Operand
typedef void (*WebnnProcOperandReference)(WebnnOperand operand);
typedef void (*WebnnProcOperandRelease)(WebnnOperand operand);

// Procs of Result
typedef const void* (*WebnnProcResultBuffer)(WebnnResult result);
typedef uint32_t (*WebnnProcResultBufferSize)(WebnnResult result);
typedef const int32_t* (*WebnnProcResultDimensions)(WebnnResult result);
typedef uint32_t (*WebnnProcResultDimensionsSize)(WebnnResult result);
typedef void (*WebnnProcResultReference)(WebnnResult result);
typedef void (*WebnnProcResultRelease)(WebnnResult result);

#endif  // !defined(WEBNN_SKIP_PROCS)

#if !defined(WEBNN_SKIP_DECLARATIONS)

WEBNN_EXPORT WebnnNamedInputs webnnCreateNamedInputs();
WEBNN_EXPORT WebnnNamedOperands webnnCreateNamedOperands();
WEBNN_EXPORT WebnnNamedOutputs webnnCreateNamedOutputs();

// Methods of Compilation
WEBNN_EXPORT void webnnCompilationCompute(WebnnCompilation compilation, WebnnNamedInputs inputs, WebnnComputeCallback callback, void * userdata, WebnnNamedOutputs outputs);
WEBNN_EXPORT void webnnCompilationReference(WebnnCompilation compilation);
WEBNN_EXPORT void webnnCompilationRelease(WebnnCompilation compilation);

// Methods of Model
WEBNN_EXPORT void webnnModelCompile(WebnnModel model, WebnnCompileCallback callback, void * userdata, WebnnCompilationOptions const * options);
WEBNN_EXPORT void webnnModelReference(WebnnModel model);
WEBNN_EXPORT void webnnModelRelease(WebnnModel model);

// Methods of ModelBuilder
WEBNN_EXPORT WebnnOperand webnnModelBuilderAdd(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b);
WEBNN_EXPORT WebnnOperand webnnModelBuilderAveragePool2d(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options);
WEBNN_EXPORT WebnnOperand webnnModelBuilderBatchNorm(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand mean, WebnnOperand variance, WebnnBatchNormOptions const * options);
WEBNN_EXPORT WebnnOperand webnnModelBuilderClamp(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnClampOptions const * options);
WEBNN_EXPORT WebnnOperand webnnModelBuilderConcat(WebnnModelBuilder modelBuilder, uint32_t inputsCount, WebnnOperand const * inputs, uint32_t axis);
WEBNN_EXPORT WebnnOperand webnnModelBuilderConstant(WebnnModelBuilder modelBuilder, WebnnOperandDescriptor const * desc, void const * value, size_t size);
WEBNN_EXPORT WebnnOperand webnnModelBuilderConv2d(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand filter, WebnnConv2dOptions const * options);
WEBNN_EXPORT WebnnModel webnnModelBuilderCreateModel(WebnnModelBuilder modelBuilder, WebnnNamedOperands namedOperands);
WEBNN_EXPORT WebnnOperand webnnModelBuilderGemm(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b, WebnnGemmOptions const * options);
WEBNN_EXPORT WebnnOperand webnnModelBuilderInput(WebnnModelBuilder modelBuilder, char const * name, WebnnOperandDescriptor const * desc);
WEBNN_EXPORT WebnnOperand webnnModelBuilderLeakyRelu(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnLeakyReluOptions const * options);
WEBNN_EXPORT WebnnOperand webnnModelBuilderMatmul(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b);
WEBNN_EXPORT WebnnOperand webnnModelBuilderMaxPool2d(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options);
WEBNN_EXPORT WebnnOperand webnnModelBuilderMul(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b);
WEBNN_EXPORT WebnnOperand webnnModelBuilderRelu(WebnnModelBuilder modelBuilder, WebnnOperand input);
WEBNN_EXPORT WebnnOperand webnnModelBuilderReshape(WebnnModelBuilder modelBuilder, WebnnOperand input, int32_t const * newShape, uint32_t newShapeCount);
WEBNN_EXPORT WebnnOperand webnnModelBuilderSoftmax(WebnnModelBuilder modelBuilder, WebnnOperand input);
WEBNN_EXPORT WebnnOperand webnnModelBuilderTranspose(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnTransposeOptions const * options);
WEBNN_EXPORT void webnnModelBuilderReference(WebnnModelBuilder modelBuilder);
WEBNN_EXPORT void webnnModelBuilderRelease(WebnnModelBuilder modelBuilder);

// Methods of NamedInputs
WEBNN_EXPORT void webnnNamedInputsSet(WebnnNamedInputs namedInputs, char const * name, WebnnInput const * input);
WEBNN_EXPORT void webnnNamedInputsReference(WebnnNamedInputs namedInputs);
WEBNN_EXPORT void webnnNamedInputsRelease(WebnnNamedInputs namedInputs);

// Methods of NamedOperands
WEBNN_EXPORT void webnnNamedOperandsSet(WebnnNamedOperands namedOperands, char const * name, WebnnOperand operand);
WEBNN_EXPORT void webnnNamedOperandsReference(WebnnNamedOperands namedOperands);
WEBNN_EXPORT void webnnNamedOperandsRelease(WebnnNamedOperands namedOperands);

// Methods of NamedOutputs
WEBNN_EXPORT void webnnNamedOutputsSet(WebnnNamedOutputs namedOutputs, char const * name, WebnnOutput const * output);
WEBNN_EXPORT void webnnNamedOutputsReference(WebnnNamedOutputs namedOutputs);
WEBNN_EXPORT void webnnNamedOutputsRelease(WebnnNamedOutputs namedOutputs);

// Methods of NamedResults
WEBNN_EXPORT WebnnResult webnnNamedResultsGet(WebnnNamedResults namedResults, char const * name);
WEBNN_EXPORT void webnnNamedResultsReference(WebnnNamedResults namedResults);
WEBNN_EXPORT void webnnNamedResultsRelease(WebnnNamedResults namedResults);

// Methods of NeuralNetworkContext
WEBNN_EXPORT WebnnModelBuilder webnnNeuralNetworkContextCreateModelBuilder(WebnnNeuralNetworkContext neuralNetworkContext);
WEBNN_EXPORT bool webnnNeuralNetworkContextPopErrorScope(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata);
WEBNN_EXPORT void webnnNeuralNetworkContextPushErrorScope(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorFilter filter);
WEBNN_EXPORT void webnnNeuralNetworkContextSetUncapturedErrorCallback(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata);
WEBNN_EXPORT void webnnNeuralNetworkContextReference(WebnnNeuralNetworkContext neuralNetworkContext);
WEBNN_EXPORT void webnnNeuralNetworkContextRelease(WebnnNeuralNetworkContext neuralNetworkContext);

// Methods of Operand
WEBNN_EXPORT void webnnOperandReference(WebnnOperand operand);
WEBNN_EXPORT void webnnOperandRelease(WebnnOperand operand);

// Methods of Result
WEBNN_EXPORT const void* webnnResultBuffer(WebnnResult result);
WEBNN_EXPORT uint32_t webnnResultBufferSize(WebnnResult result);
WEBNN_EXPORT const int32_t* webnnResultDimensions(WebnnResult result);
WEBNN_EXPORT uint32_t webnnResultDimensionsSize(WebnnResult result);
WEBNN_EXPORT void webnnResultReference(WebnnResult result);
WEBNN_EXPORT void webnnResultRelease(WebnnResult result);

#endif  // !defined(WEBNN_SKIP_DECLARATIONS)

#ifdef __cplusplus
} // extern "C"
#endif

#endif // WEBNN_H_
