
#ifndef WEBNN_WEBNN_PROC_TABLE_H_
#define WEBNN_WEBNN_PROC_TABLE_H_

#include "webnn/webnn.h"

typedef struct WebnnProcTable {
    WebnnProcCreateNamedInputs createNamedInputs;
    WebnnProcCreateNamedOperands createNamedOperands;
    WebnnProcCreateNamedOutputs createNamedOutputs;

    WebnnProcCompilationCompute compilationCompute;
    WebnnProcCompilationReference compilationReference;
    WebnnProcCompilationRelease compilationRelease;

    WebnnProcModelCompile modelCompile;
    WebnnProcModelReference modelReference;
    WebnnProcModelRelease modelRelease;

    WebnnProcModelBuilderAdd modelBuilderAdd;
    WebnnProcModelBuilderAveragePool2d modelBuilderAveragePool2d;
    WebnnProcModelBuilderBatchNorm modelBuilderBatchNorm;
    WebnnProcModelBuilderClamp modelBuilderClamp;
    WebnnProcModelBuilderConcat modelBuilderConcat;
    WebnnProcModelBuilderConstant modelBuilderConstant;
    WebnnProcModelBuilderConv2d modelBuilderConv2d;
    WebnnProcModelBuilderCreateModel modelBuilderCreateModel;
    WebnnProcModelBuilderGemm modelBuilderGemm;
    WebnnProcModelBuilderInput modelBuilderInput;
    WebnnProcModelBuilderLeakyRelu modelBuilderLeakyRelu;
    WebnnProcModelBuilderMatmul modelBuilderMatmul;
    WebnnProcModelBuilderMaxPool2d modelBuilderMaxPool2d;
    WebnnProcModelBuilderMul modelBuilderMul;
    WebnnProcModelBuilderRelu modelBuilderRelu;
    WebnnProcModelBuilderReshape modelBuilderReshape;
    WebnnProcModelBuilderSoftmax modelBuilderSoftmax;
    WebnnProcModelBuilderTranspose modelBuilderTranspose;
    WebnnProcModelBuilderReference modelBuilderReference;
    WebnnProcModelBuilderRelease modelBuilderRelease;

    WebnnProcNamedInputsSet namedInputsSet;
    WebnnProcNamedInputsReference namedInputsReference;
    WebnnProcNamedInputsRelease namedInputsRelease;

    WebnnProcNamedOperandsSet namedOperandsSet;
    WebnnProcNamedOperandsReference namedOperandsReference;
    WebnnProcNamedOperandsRelease namedOperandsRelease;

    WebnnProcNamedOutputsSet namedOutputsSet;
    WebnnProcNamedOutputsReference namedOutputsReference;
    WebnnProcNamedOutputsRelease namedOutputsRelease;

    WebnnProcNamedResultsGet namedResultsGet;
    WebnnProcNamedResultsReference namedResultsReference;
    WebnnProcNamedResultsRelease namedResultsRelease;

    WebnnProcNeuralNetworkContextCreateModelBuilder neuralNetworkContextCreateModelBuilder;
    WebnnProcNeuralNetworkContextPopErrorScope neuralNetworkContextPopErrorScope;
    WebnnProcNeuralNetworkContextPushErrorScope neuralNetworkContextPushErrorScope;
    WebnnProcNeuralNetworkContextSetUncapturedErrorCallback neuralNetworkContextSetUncapturedErrorCallback;
    WebnnProcNeuralNetworkContextReference neuralNetworkContextReference;
    WebnnProcNeuralNetworkContextRelease neuralNetworkContextRelease;

    WebnnProcOperandReference operandReference;
    WebnnProcOperandRelease operandRelease;

    WebnnProcResultBuffer resultBuffer;
    WebnnProcResultBufferSize resultBufferSize;
    WebnnProcResultDimensions resultDimensions;
    WebnnProcResultDimensionsSize resultDimensionsSize;
    WebnnProcResultReference resultReference;
    WebnnProcResultRelease resultRelease;

} WebnnProcTable;

#endif  // WEBNN_WEBNN_PROC_TABLE_H_
