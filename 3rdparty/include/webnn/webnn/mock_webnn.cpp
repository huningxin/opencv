
#include "mock_webnn.h"

using namespace testing;

namespace {
    void ForwardCompilationCompute(WebnnCompilation self, WebnnNamedInputs inputs, WebnnComputeCallback callback, void * userdata, WebnnNamedOutputs outputs) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->CompilationCompute(self, inputs, callback, userdata, outputs);
    }
    void ForwardCompilationReference(WebnnCompilation self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->CompilationReference(self);
    }
    void ForwardCompilationRelease(WebnnCompilation self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->CompilationRelease(self);
    }

    void ForwardModelCompile(WebnnModel self, WebnnCompileCallback callback, void * userdata, WebnnCompilationOptions const * options) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelCompile(self, callback, userdata, options);
    }
    void ForwardModelReference(WebnnModel self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelReference(self);
    }
    void ForwardModelRelease(WebnnModel self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelRelease(self);
    }

    WebnnOperand ForwardModelBuilderAdd(WebnnModelBuilder self, WebnnOperand a, WebnnOperand b) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderAdd(self, a, b);
    }
    WebnnOperand ForwardModelBuilderAveragePool2d(WebnnModelBuilder self, WebnnOperand input, WebnnPool2dOptions const * options) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderAveragePool2d(self, input, options);
    }
    WebnnOperand ForwardModelBuilderBatchNorm(WebnnModelBuilder self, WebnnOperand input, WebnnOperand mean, WebnnOperand variance, WebnnBatchNormOptions const * options) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderBatchNorm(self, input, mean, variance, options);
    }
    WebnnOperand ForwardModelBuilderClamp(WebnnModelBuilder self, WebnnOperand input, WebnnClampOptions const * options) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderClamp(self, input, options);
    }
    WebnnOperand ForwardModelBuilderConcat(WebnnModelBuilder self, uint32_t inputsCount, WebnnOperand const * inputs, uint32_t axis) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderConcat(self, inputsCount, inputs, axis);
    }
    WebnnOperand ForwardModelBuilderConstant(WebnnModelBuilder self, WebnnOperandDescriptor const * desc, void const * value, size_t size) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderConstant(self, desc, value, size);
    }
    WebnnOperand ForwardModelBuilderConv2d(WebnnModelBuilder self, WebnnOperand input, WebnnOperand filter, WebnnConv2dOptions const * options) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderConv2d(self, input, filter, options);
    }
    WebnnModel ForwardModelBuilderCreateModel(WebnnModelBuilder self, WebnnNamedOperands namedOperands) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderCreateModel(self, namedOperands);
    }
    WebnnOperand ForwardModelBuilderGemm(WebnnModelBuilder self, WebnnOperand a, WebnnOperand b, WebnnGemmOptions const * options) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderGemm(self, a, b, options);
    }
    WebnnOperand ForwardModelBuilderInput(WebnnModelBuilder self, char const * name, WebnnOperandDescriptor const * desc) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderInput(self, name, desc);
    }
    WebnnOperand ForwardModelBuilderLeakyRelu(WebnnModelBuilder self, WebnnOperand input, WebnnLeakyReluOptions const * options) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderLeakyRelu(self, input, options);
    }
    WebnnOperand ForwardModelBuilderMatmul(WebnnModelBuilder self, WebnnOperand a, WebnnOperand b) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderMatmul(self, a, b);
    }
    WebnnOperand ForwardModelBuilderMaxPool2d(WebnnModelBuilder self, WebnnOperand input, WebnnPool2dOptions const * options) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderMaxPool2d(self, input, options);
    }
    WebnnOperand ForwardModelBuilderMul(WebnnModelBuilder self, WebnnOperand a, WebnnOperand b) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderMul(self, a, b);
    }
    WebnnOperand ForwardModelBuilderRelu(WebnnModelBuilder self, WebnnOperand input) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderRelu(self, input);
    }
    WebnnOperand ForwardModelBuilderReshape(WebnnModelBuilder self, WebnnOperand input, int32_t const * newShape, uint32_t newShapeCount) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderReshape(self, input, newShape, newShapeCount);
    }
    WebnnOperand ForwardModelBuilderSoftmax(WebnnModelBuilder self, WebnnOperand input) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderSoftmax(self, input);
    }
    WebnnOperand ForwardModelBuilderTranspose(WebnnModelBuilder self, WebnnOperand input, WebnnTransposeOptions const * options) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderTranspose(self, input, options);
    }
    void ForwardModelBuilderReference(WebnnModelBuilder self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderReference(self);
    }
    void ForwardModelBuilderRelease(WebnnModelBuilder self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ModelBuilderRelease(self);
    }

    void ForwardNamedInputsSet(WebnnNamedInputs self, char const * name, WebnnInput const * input) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedInputsSet(self, name, input);
    }
    void ForwardNamedInputsReference(WebnnNamedInputs self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedInputsReference(self);
    }
    void ForwardNamedInputsRelease(WebnnNamedInputs self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedInputsRelease(self);
    }

    void ForwardNamedOperandsSet(WebnnNamedOperands self, char const * name, WebnnOperand operand) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedOperandsSet(self, name, operand);
    }
    void ForwardNamedOperandsReference(WebnnNamedOperands self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedOperandsReference(self);
    }
    void ForwardNamedOperandsRelease(WebnnNamedOperands self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedOperandsRelease(self);
    }

    void ForwardNamedOutputsSet(WebnnNamedOutputs self, char const * name, WebnnOutput const * output) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedOutputsSet(self, name, output);
    }
    void ForwardNamedOutputsReference(WebnnNamedOutputs self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedOutputsReference(self);
    }
    void ForwardNamedOutputsRelease(WebnnNamedOutputs self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedOutputsRelease(self);
    }

    WebnnResult ForwardNamedResultsGet(WebnnNamedResults self, char const * name) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedResultsGet(self, name);
    }
    void ForwardNamedResultsReference(WebnnNamedResults self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedResultsReference(self);
    }
    void ForwardNamedResultsRelease(WebnnNamedResults self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NamedResultsRelease(self);
    }

    WebnnModelBuilder ForwardNeuralNetworkContextCreateModelBuilder(WebnnNeuralNetworkContext self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NeuralNetworkContextCreateModelBuilder(self);
    }
    bool ForwardNeuralNetworkContextPopErrorScope(WebnnNeuralNetworkContext self, WebnnErrorCallback callback, void * userdata) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NeuralNetworkContextPopErrorScope(self, callback, userdata);
    }
    void ForwardNeuralNetworkContextPushErrorScope(WebnnNeuralNetworkContext self, WebnnErrorFilter filter) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NeuralNetworkContextPushErrorScope(self, filter);
    }
    void ForwardNeuralNetworkContextSetUncapturedErrorCallback(WebnnNeuralNetworkContext self, WebnnErrorCallback callback, void * userdata) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NeuralNetworkContextSetUncapturedErrorCallback(self, callback, userdata);
    }
    void ForwardNeuralNetworkContextReference(WebnnNeuralNetworkContext self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NeuralNetworkContextReference(self);
    }
    void ForwardNeuralNetworkContextRelease(WebnnNeuralNetworkContext self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->NeuralNetworkContextRelease(self);
    }

    void ForwardOperandReference(WebnnOperand self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->OperandReference(self);
    }
    void ForwardOperandRelease(WebnnOperand self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->OperandRelease(self);
    }

    const void* ForwardResultBuffer(WebnnResult self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ResultBuffer(self);
    }
    uint32_t ForwardResultBufferSize(WebnnResult self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ResultBufferSize(self);
    }
    const int32_t* ForwardResultDimensions(WebnnResult self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ResultDimensions(self);
    }
    uint32_t ForwardResultDimensionsSize(WebnnResult self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ResultDimensionsSize(self);
    }
    void ForwardResultReference(WebnnResult self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ResultReference(self);
    }
    void ForwardResultRelease(WebnnResult self) {
        auto object = reinterpret_cast<ProcTableAsClass::Object*>(self);
        return object->procs->ResultRelease(self);
    }

}

ProcTableAsClass::~ProcTableAsClass() {
}

void ProcTableAsClass::GetProcTableAndDevice(WebnnProcTable* table) {
    // *device = GetNewDevice();

    table->compilationCompute = reinterpret_cast<WebnnProcCompilationCompute>(ForwardCompilationCompute);
    table->compilationReference = reinterpret_cast<WebnnProcCompilationReference>(ForwardCompilationReference);
    table->compilationRelease = reinterpret_cast<WebnnProcCompilationRelease>(ForwardCompilationRelease);
    table->modelCompile = reinterpret_cast<WebnnProcModelCompile>(ForwardModelCompile);
    table->modelReference = reinterpret_cast<WebnnProcModelReference>(ForwardModelReference);
    table->modelRelease = reinterpret_cast<WebnnProcModelRelease>(ForwardModelRelease);
    table->modelBuilderAdd = reinterpret_cast<WebnnProcModelBuilderAdd>(ForwardModelBuilderAdd);
    table->modelBuilderAveragePool2d = reinterpret_cast<WebnnProcModelBuilderAveragePool2d>(ForwardModelBuilderAveragePool2d);
    table->modelBuilderBatchNorm = reinterpret_cast<WebnnProcModelBuilderBatchNorm>(ForwardModelBuilderBatchNorm);
    table->modelBuilderClamp = reinterpret_cast<WebnnProcModelBuilderClamp>(ForwardModelBuilderClamp);
    table->modelBuilderConcat = reinterpret_cast<WebnnProcModelBuilderConcat>(ForwardModelBuilderConcat);
    table->modelBuilderConstant = reinterpret_cast<WebnnProcModelBuilderConstant>(ForwardModelBuilderConstant);
    table->modelBuilderConv2d = reinterpret_cast<WebnnProcModelBuilderConv2d>(ForwardModelBuilderConv2d);
    table->modelBuilderCreateModel = reinterpret_cast<WebnnProcModelBuilderCreateModel>(ForwardModelBuilderCreateModel);
    table->modelBuilderGemm = reinterpret_cast<WebnnProcModelBuilderGemm>(ForwardModelBuilderGemm);
    table->modelBuilderInput = reinterpret_cast<WebnnProcModelBuilderInput>(ForwardModelBuilderInput);
    table->modelBuilderLeakyRelu = reinterpret_cast<WebnnProcModelBuilderLeakyRelu>(ForwardModelBuilderLeakyRelu);
    table->modelBuilderMatmul = reinterpret_cast<WebnnProcModelBuilderMatmul>(ForwardModelBuilderMatmul);
    table->modelBuilderMaxPool2d = reinterpret_cast<WebnnProcModelBuilderMaxPool2d>(ForwardModelBuilderMaxPool2d);
    table->modelBuilderMul = reinterpret_cast<WebnnProcModelBuilderMul>(ForwardModelBuilderMul);
    table->modelBuilderRelu = reinterpret_cast<WebnnProcModelBuilderRelu>(ForwardModelBuilderRelu);
    table->modelBuilderReshape = reinterpret_cast<WebnnProcModelBuilderReshape>(ForwardModelBuilderReshape);
    table->modelBuilderSoftmax = reinterpret_cast<WebnnProcModelBuilderSoftmax>(ForwardModelBuilderSoftmax);
    table->modelBuilderTranspose = reinterpret_cast<WebnnProcModelBuilderTranspose>(ForwardModelBuilderTranspose);
    table->modelBuilderReference = reinterpret_cast<WebnnProcModelBuilderReference>(ForwardModelBuilderReference);
    table->modelBuilderRelease = reinterpret_cast<WebnnProcModelBuilderRelease>(ForwardModelBuilderRelease);
    table->namedInputsSet = reinterpret_cast<WebnnProcNamedInputsSet>(ForwardNamedInputsSet);
    table->namedInputsReference = reinterpret_cast<WebnnProcNamedInputsReference>(ForwardNamedInputsReference);
    table->namedInputsRelease = reinterpret_cast<WebnnProcNamedInputsRelease>(ForwardNamedInputsRelease);
    table->namedOperandsSet = reinterpret_cast<WebnnProcNamedOperandsSet>(ForwardNamedOperandsSet);
    table->namedOperandsReference = reinterpret_cast<WebnnProcNamedOperandsReference>(ForwardNamedOperandsReference);
    table->namedOperandsRelease = reinterpret_cast<WebnnProcNamedOperandsRelease>(ForwardNamedOperandsRelease);
    table->namedOutputsSet = reinterpret_cast<WebnnProcNamedOutputsSet>(ForwardNamedOutputsSet);
    table->namedOutputsReference = reinterpret_cast<WebnnProcNamedOutputsReference>(ForwardNamedOutputsReference);
    table->namedOutputsRelease = reinterpret_cast<WebnnProcNamedOutputsRelease>(ForwardNamedOutputsRelease);
    table->namedResultsGet = reinterpret_cast<WebnnProcNamedResultsGet>(ForwardNamedResultsGet);
    table->namedResultsReference = reinterpret_cast<WebnnProcNamedResultsReference>(ForwardNamedResultsReference);
    table->namedResultsRelease = reinterpret_cast<WebnnProcNamedResultsRelease>(ForwardNamedResultsRelease);
    table->neuralNetworkContextCreateModelBuilder = reinterpret_cast<WebnnProcNeuralNetworkContextCreateModelBuilder>(ForwardNeuralNetworkContextCreateModelBuilder);
    table->neuralNetworkContextPopErrorScope = reinterpret_cast<WebnnProcNeuralNetworkContextPopErrorScope>(ForwardNeuralNetworkContextPopErrorScope);
    table->neuralNetworkContextPushErrorScope = reinterpret_cast<WebnnProcNeuralNetworkContextPushErrorScope>(ForwardNeuralNetworkContextPushErrorScope);
    table->neuralNetworkContextSetUncapturedErrorCallback = reinterpret_cast<WebnnProcNeuralNetworkContextSetUncapturedErrorCallback>(ForwardNeuralNetworkContextSetUncapturedErrorCallback);
    table->neuralNetworkContextReference = reinterpret_cast<WebnnProcNeuralNetworkContextReference>(ForwardNeuralNetworkContextReference);
    table->neuralNetworkContextRelease = reinterpret_cast<WebnnProcNeuralNetworkContextRelease>(ForwardNeuralNetworkContextRelease);
    table->operandReference = reinterpret_cast<WebnnProcOperandReference>(ForwardOperandReference);
    table->operandRelease = reinterpret_cast<WebnnProcOperandRelease>(ForwardOperandRelease);
    table->resultBuffer = reinterpret_cast<WebnnProcResultBuffer>(ForwardResultBuffer);
    table->resultBufferSize = reinterpret_cast<WebnnProcResultBufferSize>(ForwardResultBufferSize);
    table->resultDimensions = reinterpret_cast<WebnnProcResultDimensions>(ForwardResultDimensions);
    table->resultDimensionsSize = reinterpret_cast<WebnnProcResultDimensionsSize>(ForwardResultDimensionsSize);
    table->resultReference = reinterpret_cast<WebnnProcResultReference>(ForwardResultReference);
    table->resultRelease = reinterpret_cast<WebnnProcResultRelease>(ForwardResultRelease);
}


void ProcTableAsClass::CompilationCompute(WebnnCompilation compilation, WebnnNamedInputs inputs, WebnnComputeCallback callback, void * userdata, WebnnNamedOutputs outputs) {
    ProcTableAsClass::Object* object = reinterpret_cast<ProcTableAsClass::Object*>(compilation);
    object->mCompilationComputeCallback = callback;
    object->userdata = userdata;
    return OnCompilationCompute(compilation, inputs, callback, userdata, outputs);
}

void ProcTableAsClass::CallCompilationComputeCallback(WebnnCompilation compilation, WebnnComputeStatus status, WebnnNamedResults outputs, char const * message) {
    ProcTableAsClass::Object* object = reinterpret_cast<ProcTableAsClass::Object*>(compilation);
    object->mCompilationComputeCallback(status, outputs, message, object->userdata);
}

void ProcTableAsClass::ModelCompile(WebnnModel model, WebnnCompileCallback callback, void * userdata, WebnnCompilationOptions const * options) {
    ProcTableAsClass::Object* object = reinterpret_cast<ProcTableAsClass::Object*>(model);
    object->mModelCompileCallback = callback;
    object->userdata = userdata;
    return OnModelCompile(model, callback, userdata, options);
}

void ProcTableAsClass::CallModelCompileCallback(WebnnModel model, WebnnCompileStatus status, WebnnCompilation compilation, char const * message) {
    ProcTableAsClass::Object* object = reinterpret_cast<ProcTableAsClass::Object*>(model);
    object->mModelCompileCallback(status, compilation, message, object->userdata);
}

bool ProcTableAsClass::NeuralNetworkContextPopErrorScope(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata) {
    ProcTableAsClass::Object* object = reinterpret_cast<ProcTableAsClass::Object*>(neuralNetworkContext);
    object->mNeuralNetworkContextPopErrorScopeCallback = callback;
    object->userdata = userdata;
    return OnNeuralNetworkContextPopErrorScope(neuralNetworkContext, callback, userdata);
}

void ProcTableAsClass::CallNeuralNetworkContextPopErrorScopeCallback(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorType type, char const * message) {
    ProcTableAsClass::Object* object = reinterpret_cast<ProcTableAsClass::Object*>(neuralNetworkContext);
    object->mNeuralNetworkContextPopErrorScopeCallback(type, message, object->userdata);
}

void ProcTableAsClass::NeuralNetworkContextSetUncapturedErrorCallback(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata) {
    ProcTableAsClass::Object* object = reinterpret_cast<ProcTableAsClass::Object*>(neuralNetworkContext);
    object->mNeuralNetworkContextSetUncapturedErrorCallbackCallback = callback;
    object->userdata = userdata;
    return OnNeuralNetworkContextSetUncapturedErrorCallback(neuralNetworkContext, callback, userdata);
}

void ProcTableAsClass::CallNeuralNetworkContextSetUncapturedErrorCallbackCallback(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorType type, char const * message) {
    ProcTableAsClass::Object* object = reinterpret_cast<ProcTableAsClass::Object*>(neuralNetworkContext);
    object->mNeuralNetworkContextSetUncapturedErrorCallbackCallback(type, message, object->userdata);
}

WebnnCompilation ProcTableAsClass::GetNewCompilation() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnCompilation>(mObjects.back().get());
}
WebnnModel ProcTableAsClass::GetNewModel() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnModel>(mObjects.back().get());
}
WebnnModelBuilder ProcTableAsClass::GetNewModelBuilder() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnModelBuilder>(mObjects.back().get());
}
WebnnNamedInputs ProcTableAsClass::GetNewNamedInputs() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnNamedInputs>(mObjects.back().get());
}
WebnnNamedOperands ProcTableAsClass::GetNewNamedOperands() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnNamedOperands>(mObjects.back().get());
}
WebnnNamedOutputs ProcTableAsClass::GetNewNamedOutputs() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnNamedOutputs>(mObjects.back().get());
}
WebnnNamedResults ProcTableAsClass::GetNewNamedResults() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnNamedResults>(mObjects.back().get());
}
WebnnNeuralNetworkContext ProcTableAsClass::GetNewNeuralNetworkContext() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnNeuralNetworkContext>(mObjects.back().get());
}
WebnnOperand ProcTableAsClass::GetNewOperand() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnOperand>(mObjects.back().get());
}
WebnnResult ProcTableAsClass::GetNewResult() {
    mObjects.emplace_back(new Object);
    mObjects.back()->procs = this;
    return reinterpret_cast<WebnnResult>(mObjects.back().get());
}

MockProcTable::MockProcTable() = default;

MockProcTable::~MockProcTable() = default;

void MockProcTable::IgnoreAllReleaseCalls() {
    EXPECT_CALL(*this, CompilationRelease(_)).Times(AnyNumber());
    EXPECT_CALL(*this, ModelRelease(_)).Times(AnyNumber());
    EXPECT_CALL(*this, ModelBuilderRelease(_)).Times(AnyNumber());
    EXPECT_CALL(*this, NamedInputsRelease(_)).Times(AnyNumber());
    EXPECT_CALL(*this, NamedOperandsRelease(_)).Times(AnyNumber());
    EXPECT_CALL(*this, NamedOutputsRelease(_)).Times(AnyNumber());
    EXPECT_CALL(*this, NamedResultsRelease(_)).Times(AnyNumber());
    EXPECT_CALL(*this, NeuralNetworkContextRelease(_)).Times(AnyNumber());
    EXPECT_CALL(*this, OperandRelease(_)).Times(AnyNumber());
    EXPECT_CALL(*this, ResultRelease(_)).Times(AnyNumber());
}
