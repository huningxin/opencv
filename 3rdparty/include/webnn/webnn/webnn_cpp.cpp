#include "webnn/webnn_cpp.h"

namespace webnn {

    // CompileStatus

    static_assert(sizeof(CompileStatus) == sizeof(WebnnCompileStatus), "sizeof mismatch for CompileStatus");
    static_assert(alignof(CompileStatus) == alignof(WebnnCompileStatus), "alignof mismatch for CompileStatus");

    static_assert(static_cast<uint32_t>(CompileStatus::Success) == WebnnCompileStatus_Success, "value mismatch for CompileStatus::Success");
    static_assert(static_cast<uint32_t>(CompileStatus::Error) == WebnnCompileStatus_Error, "value mismatch for CompileStatus::Error");
    static_assert(static_cast<uint32_t>(CompileStatus::ContextLost) == WebnnCompileStatus_ContextLost, "value mismatch for CompileStatus::ContextLost");
    static_assert(static_cast<uint32_t>(CompileStatus::Unknown) == WebnnCompileStatus_Unknown, "value mismatch for CompileStatus::Unknown");

    // ComputeStatus

    static_assert(sizeof(ComputeStatus) == sizeof(WebnnComputeStatus), "sizeof mismatch for ComputeStatus");
    static_assert(alignof(ComputeStatus) == alignof(WebnnComputeStatus), "alignof mismatch for ComputeStatus");

    static_assert(static_cast<uint32_t>(ComputeStatus::Success) == WebnnComputeStatus_Success, "value mismatch for ComputeStatus::Success");
    static_assert(static_cast<uint32_t>(ComputeStatus::Error) == WebnnComputeStatus_Error, "value mismatch for ComputeStatus::Error");
    static_assert(static_cast<uint32_t>(ComputeStatus::ContextLost) == WebnnComputeStatus_ContextLost, "value mismatch for ComputeStatus::ContextLost");
    static_assert(static_cast<uint32_t>(ComputeStatus::Unknown) == WebnnComputeStatus_Unknown, "value mismatch for ComputeStatus::Unknown");

    // ErrorFilter

    static_assert(sizeof(ErrorFilter) == sizeof(WebnnErrorFilter), "sizeof mismatch for ErrorFilter");
    static_assert(alignof(ErrorFilter) == alignof(WebnnErrorFilter), "alignof mismatch for ErrorFilter");

    static_assert(static_cast<uint32_t>(ErrorFilter::None) == WebnnErrorFilter_None, "value mismatch for ErrorFilter::None");
    static_assert(static_cast<uint32_t>(ErrorFilter::Validation) == WebnnErrorFilter_Validation, "value mismatch for ErrorFilter::Validation");
    static_assert(static_cast<uint32_t>(ErrorFilter::OutOfMemory) == WebnnErrorFilter_OutOfMemory, "value mismatch for ErrorFilter::OutOfMemory");

    // ErrorType

    static_assert(sizeof(ErrorType) == sizeof(WebnnErrorType), "sizeof mismatch for ErrorType");
    static_assert(alignof(ErrorType) == alignof(WebnnErrorType), "alignof mismatch for ErrorType");

    static_assert(static_cast<uint32_t>(ErrorType::NoError) == WebnnErrorType_NoError, "value mismatch for ErrorType::NoError");
    static_assert(static_cast<uint32_t>(ErrorType::Validation) == WebnnErrorType_Validation, "value mismatch for ErrorType::Validation");
    static_assert(static_cast<uint32_t>(ErrorType::OutOfMemory) == WebnnErrorType_OutOfMemory, "value mismatch for ErrorType::OutOfMemory");
    static_assert(static_cast<uint32_t>(ErrorType::Unknown) == WebnnErrorType_Unknown, "value mismatch for ErrorType::Unknown");
    static_assert(static_cast<uint32_t>(ErrorType::DeviceLost) == WebnnErrorType_DeviceLost, "value mismatch for ErrorType::DeviceLost");

    // OperandLayout

    static_assert(sizeof(OperandLayout) == sizeof(WebnnOperandLayout), "sizeof mismatch for OperandLayout");
    static_assert(alignof(OperandLayout) == alignof(WebnnOperandLayout), "alignof mismatch for OperandLayout");

    static_assert(static_cast<uint32_t>(OperandLayout::Nchw) == WebnnOperandLayout_Nchw, "value mismatch for OperandLayout::Nchw");
    static_assert(static_cast<uint32_t>(OperandLayout::Nhwc) == WebnnOperandLayout_Nhwc, "value mismatch for OperandLayout::Nhwc");

    // OperandType

    static_assert(sizeof(OperandType) == sizeof(WebnnOperandType), "sizeof mismatch for OperandType");
    static_assert(alignof(OperandType) == alignof(WebnnOperandType), "alignof mismatch for OperandType");

    static_assert(static_cast<uint32_t>(OperandType::Float32) == WebnnOperandType_Float32, "value mismatch for OperandType::Float32");
    static_assert(static_cast<uint32_t>(OperandType::Float16) == WebnnOperandType_Float16, "value mismatch for OperandType::Float16");
    static_assert(static_cast<uint32_t>(OperandType::Int32) == WebnnOperandType_Int32, "value mismatch for OperandType::Int32");
    static_assert(static_cast<uint32_t>(OperandType::Uint32) == WebnnOperandType_Uint32, "value mismatch for OperandType::Uint32");

    // PowerPreference

    static_assert(sizeof(PowerPreference) == sizeof(WebnnPowerPreference), "sizeof mismatch for PowerPreference");
    static_assert(alignof(PowerPreference) == alignof(WebnnPowerPreference), "alignof mismatch for PowerPreference");

    static_assert(static_cast<uint32_t>(PowerPreference::Default) == WebnnPowerPreference_Default, "value mismatch for PowerPreference::Default");
    static_assert(static_cast<uint32_t>(PowerPreference::Low_power) == WebnnPowerPreference_Low_power, "value mismatch for PowerPreference::Low_power");
    static_assert(static_cast<uint32_t>(PowerPreference::High_performance) == WebnnPowerPreference_High_performance, "value mismatch for PowerPreference::High_performance");

    // ChainedStruct


    // BatchNormOptions

    static_assert(sizeof(BatchNormOptions) == sizeof(WebnnBatchNormOptions), "sizeof mismatch for BatchNormOptions");
    static_assert(alignof(BatchNormOptions) == alignof(WebnnBatchNormOptions), "alignof mismatch for BatchNormOptions");

    static_assert(offsetof(BatchNormOptions, scale) == offsetof(WebnnBatchNormOptions, scale),
            "offsetof mismatch for BatchNormOptions::scale");
    static_assert(offsetof(BatchNormOptions, bias) == offsetof(WebnnBatchNormOptions, bias),
            "offsetof mismatch for BatchNormOptions::bias");
    static_assert(offsetof(BatchNormOptions, axis) == offsetof(WebnnBatchNormOptions, axis),
            "offsetof mismatch for BatchNormOptions::axis");
    static_assert(offsetof(BatchNormOptions, epsilon) == offsetof(WebnnBatchNormOptions, epsilon),
            "offsetof mismatch for BatchNormOptions::epsilon");

    // ClampOptions

    static_assert(sizeof(ClampOptions) == sizeof(WebnnClampOptions), "sizeof mismatch for ClampOptions");
    static_assert(alignof(ClampOptions) == alignof(WebnnClampOptions), "alignof mismatch for ClampOptions");

    static_assert(offsetof(ClampOptions, minValue) == offsetof(WebnnClampOptions, minValue),
            "offsetof mismatch for ClampOptions::minValue");
    static_assert(offsetof(ClampOptions, maxValue) == offsetof(WebnnClampOptions, maxValue),
            "offsetof mismatch for ClampOptions::maxValue");

    // CompilationOptions

    static_assert(sizeof(CompilationOptions) == sizeof(WebnnCompilationOptions), "sizeof mismatch for CompilationOptions");
    static_assert(alignof(CompilationOptions) == alignof(WebnnCompilationOptions), "alignof mismatch for CompilationOptions");

    static_assert(offsetof(CompilationOptions, powerPreference) == offsetof(WebnnCompilationOptions, powerPreference),
            "offsetof mismatch for CompilationOptions::powerPreference");

    // Conv2dOptions

    static_assert(sizeof(Conv2dOptions) == sizeof(WebnnConv2dOptions), "sizeof mismatch for Conv2dOptions");
    static_assert(alignof(Conv2dOptions) == alignof(WebnnConv2dOptions), "alignof mismatch for Conv2dOptions");

    static_assert(offsetof(Conv2dOptions, paddingCount) == offsetof(WebnnConv2dOptions, paddingCount),
            "offsetof mismatch for Conv2dOptions::paddingCount");
    static_assert(offsetof(Conv2dOptions, padding) == offsetof(WebnnConv2dOptions, padding),
            "offsetof mismatch for Conv2dOptions::padding");
    static_assert(offsetof(Conv2dOptions, stridesCount) == offsetof(WebnnConv2dOptions, stridesCount),
            "offsetof mismatch for Conv2dOptions::stridesCount");
    static_assert(offsetof(Conv2dOptions, strides) == offsetof(WebnnConv2dOptions, strides),
            "offsetof mismatch for Conv2dOptions::strides");
    static_assert(offsetof(Conv2dOptions, dilationsCount) == offsetof(WebnnConv2dOptions, dilationsCount),
            "offsetof mismatch for Conv2dOptions::dilationsCount");
    static_assert(offsetof(Conv2dOptions, dilations) == offsetof(WebnnConv2dOptions, dilations),
            "offsetof mismatch for Conv2dOptions::dilations");
    static_assert(offsetof(Conv2dOptions, groups) == offsetof(WebnnConv2dOptions, groups),
            "offsetof mismatch for Conv2dOptions::groups");
    static_assert(offsetof(Conv2dOptions, layout) == offsetof(WebnnConv2dOptions, layout),
            "offsetof mismatch for Conv2dOptions::layout");

    // GemmOptions

    static_assert(sizeof(GemmOptions) == sizeof(WebnnGemmOptions), "sizeof mismatch for GemmOptions");
    static_assert(alignof(GemmOptions) == alignof(WebnnGemmOptions), "alignof mismatch for GemmOptions");

    static_assert(offsetof(GemmOptions, c) == offsetof(WebnnGemmOptions, c),
            "offsetof mismatch for GemmOptions::c");
    static_assert(offsetof(GemmOptions, alpha) == offsetof(WebnnGemmOptions, alpha),
            "offsetof mismatch for GemmOptions::alpha");
    static_assert(offsetof(GemmOptions, beta) == offsetof(WebnnGemmOptions, beta),
            "offsetof mismatch for GemmOptions::beta");
    static_assert(offsetof(GemmOptions, aTranspose) == offsetof(WebnnGemmOptions, aTranspose),
            "offsetof mismatch for GemmOptions::aTranspose");
    static_assert(offsetof(GemmOptions, bTranspose) == offsetof(WebnnGemmOptions, bTranspose),
            "offsetof mismatch for GemmOptions::bTranspose");

    // Input

    static_assert(sizeof(Input) == sizeof(WebnnInput), "sizeof mismatch for Input");
    static_assert(alignof(Input) == alignof(WebnnInput), "alignof mismatch for Input");

    static_assert(offsetof(Input, buffer) == offsetof(WebnnInput, buffer),
            "offsetof mismatch for Input::buffer");
    static_assert(offsetof(Input, size) == offsetof(WebnnInput, size),
            "offsetof mismatch for Input::size");
    static_assert(offsetof(Input, dimensions) == offsetof(WebnnInput, dimensions),
            "offsetof mismatch for Input::dimensions");
    static_assert(offsetof(Input, dimensionsCount) == offsetof(WebnnInput, dimensionsCount),
            "offsetof mismatch for Input::dimensionsCount");

    // LeakyReluOptions

    static_assert(sizeof(LeakyReluOptions) == sizeof(WebnnLeakyReluOptions), "sizeof mismatch for LeakyReluOptions");
    static_assert(alignof(LeakyReluOptions) == alignof(WebnnLeakyReluOptions), "alignof mismatch for LeakyReluOptions");

    static_assert(offsetof(LeakyReluOptions, alpha) == offsetof(WebnnLeakyReluOptions, alpha),
            "offsetof mismatch for LeakyReluOptions::alpha");

    // OperandDescriptor

    static_assert(sizeof(OperandDescriptor) == sizeof(WebnnOperandDescriptor), "sizeof mismatch for OperandDescriptor");
    static_assert(alignof(OperandDescriptor) == alignof(WebnnOperandDescriptor), "alignof mismatch for OperandDescriptor");

    static_assert(offsetof(OperandDescriptor, type) == offsetof(WebnnOperandDescriptor, type),
            "offsetof mismatch for OperandDescriptor::type");
    static_assert(offsetof(OperandDescriptor, dimensions) == offsetof(WebnnOperandDescriptor, dimensions),
            "offsetof mismatch for OperandDescriptor::dimensions");
    static_assert(offsetof(OperandDescriptor, dimensionsCount) == offsetof(WebnnOperandDescriptor, dimensionsCount),
            "offsetof mismatch for OperandDescriptor::dimensionsCount");

    // Output

    static_assert(sizeof(Output) == sizeof(WebnnOutput), "sizeof mismatch for Output");
    static_assert(alignof(Output) == alignof(WebnnOutput), "alignof mismatch for Output");

    static_assert(offsetof(Output, buffer) == offsetof(WebnnOutput, buffer),
            "offsetof mismatch for Output::buffer");
    static_assert(offsetof(Output, size) == offsetof(WebnnOutput, size),
            "offsetof mismatch for Output::size");
    static_assert(offsetof(Output, dimensions) == offsetof(WebnnOutput, dimensions),
            "offsetof mismatch for Output::dimensions");
    static_assert(offsetof(Output, dimensionsCount) == offsetof(WebnnOutput, dimensionsCount),
            "offsetof mismatch for Output::dimensionsCount");

    // Pool2dOptions

    static_assert(sizeof(Pool2dOptions) == sizeof(WebnnPool2dOptions), "sizeof mismatch for Pool2dOptions");
    static_assert(alignof(Pool2dOptions) == alignof(WebnnPool2dOptions), "alignof mismatch for Pool2dOptions");

    static_assert(offsetof(Pool2dOptions, windowDimensionsCount) == offsetof(WebnnPool2dOptions, windowDimensionsCount),
            "offsetof mismatch for Pool2dOptions::windowDimensionsCount");
    static_assert(offsetof(Pool2dOptions, windowDimensions) == offsetof(WebnnPool2dOptions, windowDimensions),
            "offsetof mismatch for Pool2dOptions::windowDimensions");
    static_assert(offsetof(Pool2dOptions, paddingCount) == offsetof(WebnnPool2dOptions, paddingCount),
            "offsetof mismatch for Pool2dOptions::paddingCount");
    static_assert(offsetof(Pool2dOptions, padding) == offsetof(WebnnPool2dOptions, padding),
            "offsetof mismatch for Pool2dOptions::padding");
    static_assert(offsetof(Pool2dOptions, stridesCount) == offsetof(WebnnPool2dOptions, stridesCount),
            "offsetof mismatch for Pool2dOptions::stridesCount");
    static_assert(offsetof(Pool2dOptions, strides) == offsetof(WebnnPool2dOptions, strides),
            "offsetof mismatch for Pool2dOptions::strides");
    static_assert(offsetof(Pool2dOptions, dilationsCount) == offsetof(WebnnPool2dOptions, dilationsCount),
            "offsetof mismatch for Pool2dOptions::dilationsCount");
    static_assert(offsetof(Pool2dOptions, dilations) == offsetof(WebnnPool2dOptions, dilations),
            "offsetof mismatch for Pool2dOptions::dilations");
    static_assert(offsetof(Pool2dOptions, layout) == offsetof(WebnnPool2dOptions, layout),
            "offsetof mismatch for Pool2dOptions::layout");

    // TransposeOptions

    static_assert(sizeof(TransposeOptions) == sizeof(WebnnTransposeOptions), "sizeof mismatch for TransposeOptions");
    static_assert(alignof(TransposeOptions) == alignof(WebnnTransposeOptions), "alignof mismatch for TransposeOptions");

    static_assert(offsetof(TransposeOptions, permutationCount) == offsetof(WebnnTransposeOptions, permutationCount),
            "offsetof mismatch for TransposeOptions::permutationCount");
    static_assert(offsetof(TransposeOptions, permutation) == offsetof(WebnnTransposeOptions, permutation),
            "offsetof mismatch for TransposeOptions::permutation");

    // Compilation

    static_assert(sizeof(Compilation) == sizeof(WebnnCompilation), "sizeof mismatch for Compilation");
    static_assert(alignof(Compilation) == alignof(WebnnCompilation), "alignof mismatch for Compilation");

    void Compilation::Compute(NamedInputs const& inputs, ComputeCallback callback, void * userdata, NamedOutputs const& outputs) const {
        webnnCompilationCompute(GetHandle(), inputs.GetHandle(), callback, reinterpret_cast<void * >(userdata), outputs.GetHandle());
    }
    void Compilation::WebnnReference(WebnnCompilation handle) {
        if (handle != nullptr) {
            webnnCompilationReference(handle);
        }
    }
    void Compilation::WebnnRelease(WebnnCompilation handle) {
        if (handle != nullptr) {
            webnnCompilationRelease(handle);
        }
    }

    // Model

    static_assert(sizeof(Model) == sizeof(WebnnModel), "sizeof mismatch for Model");
    static_assert(alignof(Model) == alignof(WebnnModel), "alignof mismatch for Model");

    void Model::Compile(CompileCallback callback, void * userdata, CompilationOptions const * options) const {
        webnnModelCompile(GetHandle(), callback, reinterpret_cast<void * >(userdata), reinterpret_cast<WebnnCompilationOptions const * >(options));
    }
    void Model::WebnnReference(WebnnModel handle) {
        if (handle != nullptr) {
            webnnModelReference(handle);
        }
    }
    void Model::WebnnRelease(WebnnModel handle) {
        if (handle != nullptr) {
            webnnModelRelease(handle);
        }
    }

    // ModelBuilder

    static_assert(sizeof(ModelBuilder) == sizeof(WebnnModelBuilder), "sizeof mismatch for ModelBuilder");
    static_assert(alignof(ModelBuilder) == alignof(WebnnModelBuilder), "alignof mismatch for ModelBuilder");

    Operand ModelBuilder::Add(Operand const& a, Operand const& b) const {
        auto result = webnnModelBuilderAdd(GetHandle(), a.GetHandle(), b.GetHandle());
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::AveragePool2d(Operand const& input, Pool2dOptions const * options) const {
        auto result = webnnModelBuilderAveragePool2d(GetHandle(), input.GetHandle(), reinterpret_cast<WebnnPool2dOptions const * >(options));
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::BatchNorm(Operand const& input, Operand const& mean, Operand const& variance, BatchNormOptions const * options) const {
        auto result = webnnModelBuilderBatchNorm(GetHandle(), input.GetHandle(), mean.GetHandle(), variance.GetHandle(), reinterpret_cast<WebnnBatchNormOptions const * >(options));
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Clamp(Operand const& input, ClampOptions const * options) const {
        auto result = webnnModelBuilderClamp(GetHandle(), input.GetHandle(), reinterpret_cast<WebnnClampOptions const * >(options));
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Concat(uint32_t inputsCount, Operand const * inputs, uint32_t axis) const {
        auto result = webnnModelBuilderConcat(GetHandle(), inputsCount, reinterpret_cast<WebnnOperand const * >(inputs), axis);
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Constant(OperandDescriptor const * desc, void const * value, size_t size) const {
        auto result = webnnModelBuilderConstant(GetHandle(), reinterpret_cast<WebnnOperandDescriptor const * >(desc), reinterpret_cast<void const * >(value), size);
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Conv2d(Operand const& input, Operand const& filter, Conv2dOptions const * options) const {
        auto result = webnnModelBuilderConv2d(GetHandle(), input.GetHandle(), filter.GetHandle(), reinterpret_cast<WebnnConv2dOptions const * >(options));
        return Operand::Acquire(result);
    }
    Model ModelBuilder::CreateModel(NamedOperands const& namedOperands) const {
        auto result = webnnModelBuilderCreateModel(GetHandle(), namedOperands.GetHandle());
        return Model::Acquire(result);
    }
    Operand ModelBuilder::Gemm(Operand const& a, Operand const& b, GemmOptions const * options) const {
        auto result = webnnModelBuilderGemm(GetHandle(), a.GetHandle(), b.GetHandle(), reinterpret_cast<WebnnGemmOptions const * >(options));
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Input(char const * name, OperandDescriptor const * desc) const {
        auto result = webnnModelBuilderInput(GetHandle(), reinterpret_cast<char const * >(name), reinterpret_cast<WebnnOperandDescriptor const * >(desc));
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::LeakyRelu(Operand const& input, LeakyReluOptions const * options) const {
        auto result = webnnModelBuilderLeakyRelu(GetHandle(), input.GetHandle(), reinterpret_cast<WebnnLeakyReluOptions const * >(options));
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Matmul(Operand const& a, Operand const& b) const {
        auto result = webnnModelBuilderMatmul(GetHandle(), a.GetHandle(), b.GetHandle());
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::MaxPool2d(Operand const& input, Pool2dOptions const * options) const {
        auto result = webnnModelBuilderMaxPool2d(GetHandle(), input.GetHandle(), reinterpret_cast<WebnnPool2dOptions const * >(options));
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Mul(Operand const& a, Operand const& b) const {
        auto result = webnnModelBuilderMul(GetHandle(), a.GetHandle(), b.GetHandle());
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Relu(Operand const& input) const {
        auto result = webnnModelBuilderRelu(GetHandle(), input.GetHandle());
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Reshape(Operand const& input, int32_t const * newShape, uint32_t newShapeCount) const {
        auto result = webnnModelBuilderReshape(GetHandle(), input.GetHandle(), reinterpret_cast<int32_t const * >(newShape), newShapeCount);
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Softmax(Operand const& input) const {
        auto result = webnnModelBuilderSoftmax(GetHandle(), input.GetHandle());
        return Operand::Acquire(result);
    }
    Operand ModelBuilder::Transpose(Operand const& input, TransposeOptions const * options) const {
        auto result = webnnModelBuilderTranspose(GetHandle(), input.GetHandle(), reinterpret_cast<WebnnTransposeOptions const * >(options));
        return Operand::Acquire(result);
    }
    void ModelBuilder::WebnnReference(WebnnModelBuilder handle) {
        if (handle != nullptr) {
            webnnModelBuilderReference(handle);
        }
    }
    void ModelBuilder::WebnnRelease(WebnnModelBuilder handle) {
        if (handle != nullptr) {
            webnnModelBuilderRelease(handle);
        }
    }

    // NamedInputs

    static_assert(sizeof(NamedInputs) == sizeof(WebnnNamedInputs), "sizeof mismatch for NamedInputs");
    static_assert(alignof(NamedInputs) == alignof(WebnnNamedInputs), "alignof mismatch for NamedInputs");

    void NamedInputs::Set(char const * name, Input const * input) const {
        webnnNamedInputsSet(GetHandle(), reinterpret_cast<char const * >(name), reinterpret_cast<WebnnInput const * >(input));
    }
    void NamedInputs::WebnnReference(WebnnNamedInputs handle) {
        if (handle != nullptr) {
            webnnNamedInputsReference(handle);
        }
    }
    void NamedInputs::WebnnRelease(WebnnNamedInputs handle) {
        if (handle != nullptr) {
            webnnNamedInputsRelease(handle);
        }
    }

    // NamedOperands

    static_assert(sizeof(NamedOperands) == sizeof(WebnnNamedOperands), "sizeof mismatch for NamedOperands");
    static_assert(alignof(NamedOperands) == alignof(WebnnNamedOperands), "alignof mismatch for NamedOperands");

    void NamedOperands::Set(char const * name, Operand const& operand) const {
        webnnNamedOperandsSet(GetHandle(), reinterpret_cast<char const * >(name), operand.GetHandle());
    }
    void NamedOperands::WebnnReference(WebnnNamedOperands handle) {
        if (handle != nullptr) {
            webnnNamedOperandsReference(handle);
        }
    }
    void NamedOperands::WebnnRelease(WebnnNamedOperands handle) {
        if (handle != nullptr) {
            webnnNamedOperandsRelease(handle);
        }
    }

    // NamedOutputs

    static_assert(sizeof(NamedOutputs) == sizeof(WebnnNamedOutputs), "sizeof mismatch for NamedOutputs");
    static_assert(alignof(NamedOutputs) == alignof(WebnnNamedOutputs), "alignof mismatch for NamedOutputs");

    void NamedOutputs::Set(char const * name, Output const * output) const {
        webnnNamedOutputsSet(GetHandle(), reinterpret_cast<char const * >(name), reinterpret_cast<WebnnOutput const * >(output));
    }
    void NamedOutputs::WebnnReference(WebnnNamedOutputs handle) {
        if (handle != nullptr) {
            webnnNamedOutputsReference(handle);
        }
    }
    void NamedOutputs::WebnnRelease(WebnnNamedOutputs handle) {
        if (handle != nullptr) {
            webnnNamedOutputsRelease(handle);
        }
    }

    // NamedResults

    static_assert(sizeof(NamedResults) == sizeof(WebnnNamedResults), "sizeof mismatch for NamedResults");
    static_assert(alignof(NamedResults) == alignof(WebnnNamedResults), "alignof mismatch for NamedResults");

    Result NamedResults::Get(char const * name) const {
        auto result = webnnNamedResultsGet(GetHandle(), reinterpret_cast<char const * >(name));
        return Result::Acquire(result);
    }
    void NamedResults::WebnnReference(WebnnNamedResults handle) {
        if (handle != nullptr) {
            webnnNamedResultsReference(handle);
        }
    }
    void NamedResults::WebnnRelease(WebnnNamedResults handle) {
        if (handle != nullptr) {
            webnnNamedResultsRelease(handle);
        }
    }

    // NeuralNetworkContext

    static_assert(sizeof(NeuralNetworkContext) == sizeof(WebnnNeuralNetworkContext), "sizeof mismatch for NeuralNetworkContext");
    static_assert(alignof(NeuralNetworkContext) == alignof(WebnnNeuralNetworkContext), "alignof mismatch for NeuralNetworkContext");

    ModelBuilder NeuralNetworkContext::CreateModelBuilder() const {
        auto result = webnnNeuralNetworkContextCreateModelBuilder(GetHandle());
        return ModelBuilder::Acquire(result);
    }
    bool NeuralNetworkContext::PopErrorScope(ErrorCallback callback, void * userdata) const {
        auto result = webnnNeuralNetworkContextPopErrorScope(GetHandle(), callback, reinterpret_cast<void * >(userdata));
        return result;
    }
    void NeuralNetworkContext::PushErrorScope(ErrorFilter filter) const {
        webnnNeuralNetworkContextPushErrorScope(GetHandle(), static_cast<WebnnErrorFilter>(filter));
    }
    void NeuralNetworkContext::SetUncapturedErrorCallback(ErrorCallback callback, void * userdata) const {
        webnnNeuralNetworkContextSetUncapturedErrorCallback(GetHandle(), callback, reinterpret_cast<void * >(userdata));
    }
    void NeuralNetworkContext::WebnnReference(WebnnNeuralNetworkContext handle) {
        if (handle != nullptr) {
            webnnNeuralNetworkContextReference(handle);
        }
    }
    void NeuralNetworkContext::WebnnRelease(WebnnNeuralNetworkContext handle) {
        if (handle != nullptr) {
            webnnNeuralNetworkContextRelease(handle);
        }
    }

    // Operand

    static_assert(sizeof(Operand) == sizeof(WebnnOperand), "sizeof mismatch for Operand");
    static_assert(alignof(Operand) == alignof(WebnnOperand), "alignof mismatch for Operand");

    void Operand::WebnnReference(WebnnOperand handle) {
        if (handle != nullptr) {
            webnnOperandReference(handle);
        }
    }
    void Operand::WebnnRelease(WebnnOperand handle) {
        if (handle != nullptr) {
            webnnOperandRelease(handle);
        }
    }

    // Result

    static_assert(sizeof(Result) == sizeof(WebnnResult), "sizeof mismatch for Result");
    static_assert(alignof(Result) == alignof(WebnnResult), "alignof mismatch for Result");

    const void* Result::Buffer() const {
        auto result = webnnResultBuffer(GetHandle());
        return result;
    }
    uint32_t Result::BufferSize() const {
        auto result = webnnResultBufferSize(GetHandle());
        return result;
    }
    const int32_t* Result::Dimensions() const {
        auto result = webnnResultDimensions(GetHandle());
        return result;
    }
    uint32_t Result::DimensionsSize() const {
        auto result = webnnResultDimensionsSize(GetHandle());
        return result;
    }
    void Result::WebnnReference(WebnnResult handle) {
        if (handle != nullptr) {
            webnnResultReference(handle);
        }
    }
    void Result::WebnnRelease(WebnnResult handle) {
        if (handle != nullptr) {
            webnnResultRelease(handle);
        }
    }

    NamedInputs CreateNamedInputs() {
        return NamedInputs::Acquire(webnnCreateNamedInputs());
    }

    NamedOperands CreateNamedOperands() {
        return NamedOperands::Acquire(webnnCreateNamedOperands());
    }

    NamedOutputs CreateNamedOutputs() {
        return NamedOutputs::Acquire(webnnCreateNamedOutputs());
    }

}
