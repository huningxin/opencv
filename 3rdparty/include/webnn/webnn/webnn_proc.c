
#include "webnn/webnn_proc.h"

static WebnnProcTable procs;

static WebnnProcTable nullProcs;

void webnnProcSetProcs(const WebnnProcTable* procs_) {
    if (procs_) {
        procs = *procs_;
    } else {
        procs = nullProcs;
    }
}

WebnnNamedInputs webnnCreateNamedInputs() {
    return procs.createNamedInputs();
}

WebnnNamedOperands webnnCreateNamedOperands() {
    return procs.createNamedOperands();
}

WebnnNamedOutputs webnnCreateNamedOutputs() {
    return procs.createNamedOutputs();
}

void webnnCompilationCompute(WebnnCompilation compilation, WebnnNamedInputs inputs, WebnnComputeCallback callback, void * userdata, WebnnNamedOutputs outputs) {
    procs.compilationCompute(compilation, inputs, callback, userdata, outputs);
}
void webnnCompilationReference(WebnnCompilation compilation) {
    procs.compilationReference(compilation);
}
void webnnCompilationRelease(WebnnCompilation compilation) {
    procs.compilationRelease(compilation);
}

void webnnModelCompile(WebnnModel model, WebnnCompileCallback callback, void * userdata, WebnnCompilationOptions const * options) {
    procs.modelCompile(model, callback, userdata, options);
}
void webnnModelReference(WebnnModel model) {
    procs.modelReference(model);
}
void webnnModelRelease(WebnnModel model) {
    procs.modelRelease(model);
}

WebnnOperand webnnModelBuilderAdd(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b) {
return     procs.modelBuilderAdd(modelBuilder, a, b);
}
WebnnOperand webnnModelBuilderAveragePool2d(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options) {
return     procs.modelBuilderAveragePool2d(modelBuilder, input, options);
}
WebnnOperand webnnModelBuilderBatchNorm(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand mean, WebnnOperand variance, WebnnBatchNormOptions const * options) {
return     procs.modelBuilderBatchNorm(modelBuilder, input, mean, variance, options);
}
WebnnOperand webnnModelBuilderClamp(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnClampOptions const * options) {
return     procs.modelBuilderClamp(modelBuilder, input, options);
}
WebnnOperand webnnModelBuilderConcat(WebnnModelBuilder modelBuilder, uint32_t inputsCount, WebnnOperand const * inputs, uint32_t axis) {
return     procs.modelBuilderConcat(modelBuilder, inputsCount, inputs, axis);
}
WebnnOperand webnnModelBuilderConstant(WebnnModelBuilder modelBuilder, WebnnOperandDescriptor const * desc, void const * value, size_t size) {
return     procs.modelBuilderConstant(modelBuilder, desc, value, size);
}
WebnnOperand webnnModelBuilderConv2d(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand filter, WebnnConv2dOptions const * options) {
return     procs.modelBuilderConv2d(modelBuilder, input, filter, options);
}
WebnnModel webnnModelBuilderCreateModel(WebnnModelBuilder modelBuilder, WebnnNamedOperands namedOperands) {
return     procs.modelBuilderCreateModel(modelBuilder, namedOperands);
}
WebnnOperand webnnModelBuilderGemm(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b, WebnnGemmOptions const * options) {
return     procs.modelBuilderGemm(modelBuilder, a, b, options);
}
WebnnOperand webnnModelBuilderInput(WebnnModelBuilder modelBuilder, char const * name, WebnnOperandDescriptor const * desc) {
return     procs.modelBuilderInput(modelBuilder, name, desc);
}
WebnnOperand webnnModelBuilderLeakyRelu(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnLeakyReluOptions const * options) {
return     procs.modelBuilderLeakyRelu(modelBuilder, input, options);
}
WebnnOperand webnnModelBuilderMatmul(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b) {
return     procs.modelBuilderMatmul(modelBuilder, a, b);
}
WebnnOperand webnnModelBuilderMaxPool2d(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options) {
return     procs.modelBuilderMaxPool2d(modelBuilder, input, options);
}
WebnnOperand webnnModelBuilderMul(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b) {
return     procs.modelBuilderMul(modelBuilder, a, b);
}
WebnnOperand webnnModelBuilderRelu(WebnnModelBuilder modelBuilder, WebnnOperand input) {
return     procs.modelBuilderRelu(modelBuilder, input);
}
WebnnOperand webnnModelBuilderReshape(WebnnModelBuilder modelBuilder, WebnnOperand input, int32_t const * newShape, uint32_t newShapeCount) {
return     procs.modelBuilderReshape(modelBuilder, input, newShape, newShapeCount);
}
WebnnOperand webnnModelBuilderSoftmax(WebnnModelBuilder modelBuilder, WebnnOperand input) {
return     procs.modelBuilderSoftmax(modelBuilder, input);
}
WebnnOperand webnnModelBuilderTranspose(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnTransposeOptions const * options) {
return     procs.modelBuilderTranspose(modelBuilder, input, options);
}
void webnnModelBuilderReference(WebnnModelBuilder modelBuilder) {
    procs.modelBuilderReference(modelBuilder);
}
void webnnModelBuilderRelease(WebnnModelBuilder modelBuilder) {
    procs.modelBuilderRelease(modelBuilder);
}

void webnnNamedInputsSet(WebnnNamedInputs namedInputs, char const * name, WebnnInput const * input) {
    procs.namedInputsSet(namedInputs, name, input);
}
void webnnNamedInputsReference(WebnnNamedInputs namedInputs) {
    procs.namedInputsReference(namedInputs);
}
void webnnNamedInputsRelease(WebnnNamedInputs namedInputs) {
    procs.namedInputsRelease(namedInputs);
}

void webnnNamedOperandsSet(WebnnNamedOperands namedOperands, char const * name, WebnnOperand operand) {
    procs.namedOperandsSet(namedOperands, name, operand);
}
void webnnNamedOperandsReference(WebnnNamedOperands namedOperands) {
    procs.namedOperandsReference(namedOperands);
}
void webnnNamedOperandsRelease(WebnnNamedOperands namedOperands) {
    procs.namedOperandsRelease(namedOperands);
}

void webnnNamedOutputsSet(WebnnNamedOutputs namedOutputs, char const * name, WebnnOutput const * output) {
    procs.namedOutputsSet(namedOutputs, name, output);
}
void webnnNamedOutputsReference(WebnnNamedOutputs namedOutputs) {
    procs.namedOutputsReference(namedOutputs);
}
void webnnNamedOutputsRelease(WebnnNamedOutputs namedOutputs) {
    procs.namedOutputsRelease(namedOutputs);
}

WebnnResult webnnNamedResultsGet(WebnnNamedResults namedResults, char const * name) {
return     procs.namedResultsGet(namedResults, name);
}
void webnnNamedResultsReference(WebnnNamedResults namedResults) {
    procs.namedResultsReference(namedResults);
}
void webnnNamedResultsRelease(WebnnNamedResults namedResults) {
    procs.namedResultsRelease(namedResults);
}

WebnnModelBuilder webnnNeuralNetworkContextCreateModelBuilder(WebnnNeuralNetworkContext neuralNetworkContext) {
return     procs.neuralNetworkContextCreateModelBuilder(neuralNetworkContext);
}
bool webnnNeuralNetworkContextPopErrorScope(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata) {
return     procs.neuralNetworkContextPopErrorScope(neuralNetworkContext, callback, userdata);
}
void webnnNeuralNetworkContextPushErrorScope(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorFilter filter) {
    procs.neuralNetworkContextPushErrorScope(neuralNetworkContext, filter);
}
void webnnNeuralNetworkContextSetUncapturedErrorCallback(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata) {
    procs.neuralNetworkContextSetUncapturedErrorCallback(neuralNetworkContext, callback, userdata);
}
void webnnNeuralNetworkContextReference(WebnnNeuralNetworkContext neuralNetworkContext) {
    procs.neuralNetworkContextReference(neuralNetworkContext);
}
void webnnNeuralNetworkContextRelease(WebnnNeuralNetworkContext neuralNetworkContext) {
    procs.neuralNetworkContextRelease(neuralNetworkContext);
}

void webnnOperandReference(WebnnOperand operand) {
    procs.operandReference(operand);
}
void webnnOperandRelease(WebnnOperand operand) {
    procs.operandRelease(operand);
}

const void* webnnResultBuffer(WebnnResult result) {
return     procs.resultBuffer(result);
}
uint32_t webnnResultBufferSize(WebnnResult result) {
return     procs.resultBufferSize(result);
}
const int32_t* webnnResultDimensions(WebnnResult result) {
return     procs.resultDimensions(result);
}
uint32_t webnnResultDimensionsSize(WebnnResult result) {
return     procs.resultDimensionsSize(result);
}
void webnnResultReference(WebnnResult result) {
    procs.resultReference(result);
}
void webnnResultRelease(WebnnResult result) {
    procs.resultRelease(result);
}

