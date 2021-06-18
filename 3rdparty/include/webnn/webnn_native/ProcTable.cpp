
#include "webnn_native/webnn_platform.h"
#include "webnn_native/WebnnNative.h"

#include <algorithm>
#include <vector>

#include "webnn_native/Compilation.h"
#include "webnn_native/Model.h"
#include "webnn_native/ModelBuilder.h"
#include "webnn_native/NamedInputs.h"
#include "webnn_native/NamedOperands.h"
#include "webnn_native/NamedOutputs.h"
#include "webnn_native/NamedResults.h"
#include "webnn_native/NeuralNetworkContext.h"
#include "webnn_native/Operand.h"
#include "webnn_native/Result.h"

namespace webnn_native {

    namespace {


        void NativeCompilationCompute(WebnnCompilation cSelf, WebnnNamedInputs inputs, WebnnComputeCallback callback, void * userdata, WebnnNamedOutputs outputs) {
            auto self = reinterpret_cast<CompilationBase*>(cSelf);

            auto inputs_ = reinterpret_cast<NamedInputsBase* >(inputs);
            auto callback_ = callback;
            auto userdata_ = reinterpret_cast<void * >(userdata);
            auto outputs_ = reinterpret_cast<NamedOutputsBase* >(outputs);
            self->Compute(inputs_, callback_, userdata_, outputs_);
        }

        void NativeCompilationReference(WebnnCompilation cSelf) {
            auto self = reinterpret_cast<CompilationBase*>(cSelf);

            self->Reference();
        }

        void NativeCompilationRelease(WebnnCompilation cSelf) {
            auto self = reinterpret_cast<CompilationBase*>(cSelf);

            self->Release();
        }

        void NativeModelCompile(WebnnModel cSelf, WebnnCompileCallback callback, void * userdata, WebnnCompilationOptions const * options) {
            auto self = reinterpret_cast<ModelBase*>(cSelf);

            auto callback_ = callback;
            auto userdata_ = reinterpret_cast<void * >(userdata);
            auto options_ = reinterpret_cast<CompilationOptions const * >(options);
            self->Compile(callback_, userdata_, options_);
        }

        void NativeModelReference(WebnnModel cSelf) {
            auto self = reinterpret_cast<ModelBase*>(cSelf);

            self->Reference();
        }

        void NativeModelRelease(WebnnModel cSelf) {
            auto self = reinterpret_cast<ModelBase*>(cSelf);

            self->Release();
        }

        WebnnOperand NativeModelBuilderAdd(WebnnModelBuilder cSelf, WebnnOperand a, WebnnOperand b) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Add(a_, b_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderAveragePool2d(WebnnModelBuilder cSelf, WebnnOperand input, WebnnPool2dOptions const * options) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<Pool2dOptions const * >(options);
            auto result =            self->AveragePool2d(input_, options_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderBatchNorm(WebnnModelBuilder cSelf, WebnnOperand input, WebnnOperand mean, WebnnOperand variance, WebnnBatchNormOptions const * options) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto mean_ = reinterpret_cast<OperandBase* >(mean);
            auto variance_ = reinterpret_cast<OperandBase* >(variance);
            auto options_ = reinterpret_cast<BatchNormOptions const * >(options);
            auto result =            self->BatchNorm(input_, mean_, variance_, options_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderClamp(WebnnModelBuilder cSelf, WebnnOperand input, WebnnClampOptions const * options) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<ClampOptions const * >(options);
            auto result =            self->Clamp(input_, options_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderConcat(WebnnModelBuilder cSelf, uint32_t inputsCount, WebnnOperand const * inputs, uint32_t axis) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto inputsCount_ = inputsCount;
            auto inputs_ = reinterpret_cast<OperandBase* const * >(inputs);
            auto axis_ = axis;
            auto result =            self->Concat(inputsCount_, inputs_, axis_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderConstant(WebnnModelBuilder cSelf, WebnnOperandDescriptor const * desc, void const * value, size_t size) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto desc_ = reinterpret_cast<OperandDescriptor const * >(desc);
            auto value_ = reinterpret_cast<void const * >(value);
            auto size_ = size;
            auto result =            self->Constant(desc_, value_, size_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderConv2d(WebnnModelBuilder cSelf, WebnnOperand input, WebnnOperand filter, WebnnConv2dOptions const * options) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto filter_ = reinterpret_cast<OperandBase* >(filter);
            auto options_ = reinterpret_cast<Conv2dOptions const * >(options);
            auto result =            self->Conv2d(input_, filter_, options_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnModel NativeModelBuilderCreateModel(WebnnModelBuilder cSelf, WebnnNamedOperands namedOperands) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto namedOperands_ = reinterpret_cast<NamedOperandsBase* >(namedOperands);
            auto result =            self->CreateModel(namedOperands_);
            return reinterpret_cast<WebnnModel>(result);
        }

        WebnnOperand NativeModelBuilderGemm(WebnnModelBuilder cSelf, WebnnOperand a, WebnnOperand b, WebnnGemmOptions const * options) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto options_ = reinterpret_cast<GemmOptions const * >(options);
            auto result =            self->Gemm(a_, b_, options_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderInput(WebnnModelBuilder cSelf, char const * name, WebnnOperandDescriptor const * desc) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto name_ = reinterpret_cast<char const * >(name);
            auto desc_ = reinterpret_cast<OperandDescriptor const * >(desc);
            auto result =            self->Input(name_, desc_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderLeakyRelu(WebnnModelBuilder cSelf, WebnnOperand input, WebnnLeakyReluOptions const * options) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<LeakyReluOptions const * >(options);
            auto result =            self->LeakyRelu(input_, options_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderMatmul(WebnnModelBuilder cSelf, WebnnOperand a, WebnnOperand b) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Matmul(a_, b_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderMaxPool2d(WebnnModelBuilder cSelf, WebnnOperand input, WebnnPool2dOptions const * options) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<Pool2dOptions const * >(options);
            auto result =            self->MaxPool2d(input_, options_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderMul(WebnnModelBuilder cSelf, WebnnOperand a, WebnnOperand b) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto a_ = reinterpret_cast<OperandBase* >(a);
            auto b_ = reinterpret_cast<OperandBase* >(b);
            auto result =            self->Mul(a_, b_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderRelu(WebnnModelBuilder cSelf, WebnnOperand input) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Relu(input_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderReshape(WebnnModelBuilder cSelf, WebnnOperand input, int32_t const * newShape, uint32_t newShapeCount) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto newShape_ = reinterpret_cast<int32_t const * >(newShape);
            auto newShapeCount_ = newShapeCount;
            auto result =            self->Reshape(input_, newShape_, newShapeCount_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderSoftmax(WebnnModelBuilder cSelf, WebnnOperand input) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto result =            self->Softmax(input_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        WebnnOperand NativeModelBuilderTranspose(WebnnModelBuilder cSelf, WebnnOperand input, WebnnTransposeOptions const * options) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            auto input_ = reinterpret_cast<OperandBase* >(input);
            auto options_ = reinterpret_cast<TransposeOptions const * >(options);
            auto result =            self->Transpose(input_, options_);
            return reinterpret_cast<WebnnOperand>(result);
        }

        void NativeModelBuilderReference(WebnnModelBuilder cSelf) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            self->Reference();
        }

        void NativeModelBuilderRelease(WebnnModelBuilder cSelf) {
            auto self = reinterpret_cast<ModelBuilderBase*>(cSelf);

            self->Release();
        }

        void NativeNamedInputsSet(WebnnNamedInputs cSelf, char const * name, WebnnInput const * input) {
            auto self = reinterpret_cast<NamedInputsBase*>(cSelf);

            auto name_ = reinterpret_cast<char const * >(name);
            auto input_ = reinterpret_cast<Input const * >(input);
            self->Set(name_, input_);
        }

        void NativeNamedInputsReference(WebnnNamedInputs cSelf) {
            auto self = reinterpret_cast<NamedInputsBase*>(cSelf);

            self->Reference();
        }

        void NativeNamedInputsRelease(WebnnNamedInputs cSelf) {
            auto self = reinterpret_cast<NamedInputsBase*>(cSelf);

            self->Release();
        }

        void NativeNamedOperandsSet(WebnnNamedOperands cSelf, char const * name, WebnnOperand operand) {
            auto self = reinterpret_cast<NamedOperandsBase*>(cSelf);

            auto name_ = reinterpret_cast<char const * >(name);
            auto operand_ = reinterpret_cast<OperandBase* >(operand);
            self->Set(name_, operand_);
        }

        void NativeNamedOperandsReference(WebnnNamedOperands cSelf) {
            auto self = reinterpret_cast<NamedOperandsBase*>(cSelf);

            self->Reference();
        }

        void NativeNamedOperandsRelease(WebnnNamedOperands cSelf) {
            auto self = reinterpret_cast<NamedOperandsBase*>(cSelf);

            self->Release();
        }

        void NativeNamedOutputsSet(WebnnNamedOutputs cSelf, char const * name, WebnnOutput const * output) {
            auto self = reinterpret_cast<NamedOutputsBase*>(cSelf);

            auto name_ = reinterpret_cast<char const * >(name);
            auto output_ = reinterpret_cast<Output const * >(output);
            self->Set(name_, output_);
        }

        void NativeNamedOutputsReference(WebnnNamedOutputs cSelf) {
            auto self = reinterpret_cast<NamedOutputsBase*>(cSelf);

            self->Reference();
        }

        void NativeNamedOutputsRelease(WebnnNamedOutputs cSelf) {
            auto self = reinterpret_cast<NamedOutputsBase*>(cSelf);

            self->Release();
        }

        WebnnResult NativeNamedResultsGet(WebnnNamedResults cSelf, char const * name) {
            auto self = reinterpret_cast<NamedResultsBase*>(cSelf);

            auto name_ = reinterpret_cast<char const * >(name);
            auto result =            self->Get(name_);
            return reinterpret_cast<WebnnResult>(result);
        }

        void NativeNamedResultsReference(WebnnNamedResults cSelf) {
            auto self = reinterpret_cast<NamedResultsBase*>(cSelf);

            self->Reference();
        }

        void NativeNamedResultsRelease(WebnnNamedResults cSelf) {
            auto self = reinterpret_cast<NamedResultsBase*>(cSelf);

            self->Release();
        }

        WebnnModelBuilder NativeNeuralNetworkContextCreateModelBuilder(WebnnNeuralNetworkContext cSelf) {
            auto self = reinterpret_cast<NeuralNetworkContextBase*>(cSelf);

            auto result =            self->CreateModelBuilder();
            return reinterpret_cast<WebnnModelBuilder>(result);
        }

        bool NativeNeuralNetworkContextPopErrorScope(WebnnNeuralNetworkContext cSelf, WebnnErrorCallback callback, void * userdata) {
            auto self = reinterpret_cast<NeuralNetworkContextBase*>(cSelf);

            auto callback_ = callback;
            auto userdata_ = reinterpret_cast<void * >(userdata);
            auto result =            self->PopErrorScope(callback_, userdata_);
            return result;
        }

        void NativeNeuralNetworkContextPushErrorScope(WebnnNeuralNetworkContext cSelf, WebnnErrorFilter filter) {
            auto self = reinterpret_cast<NeuralNetworkContextBase*>(cSelf);

            auto filter_ = static_cast<webnn::ErrorFilter>(filter);
            self->PushErrorScope(filter_);
        }

        void NativeNeuralNetworkContextSetUncapturedErrorCallback(WebnnNeuralNetworkContext cSelf, WebnnErrorCallback callback, void * userdata) {
            auto self = reinterpret_cast<NeuralNetworkContextBase*>(cSelf);

            auto callback_ = callback;
            auto userdata_ = reinterpret_cast<void * >(userdata);
            self->SetUncapturedErrorCallback(callback_, userdata_);
        }

        void NativeNeuralNetworkContextReference(WebnnNeuralNetworkContext cSelf) {
            auto self = reinterpret_cast<NeuralNetworkContextBase*>(cSelf);

            self->Reference();
        }

        void NativeNeuralNetworkContextRelease(WebnnNeuralNetworkContext cSelf) {
            auto self = reinterpret_cast<NeuralNetworkContextBase*>(cSelf);

            self->Release();
        }

        void NativeOperandReference(WebnnOperand cSelf) {
            auto self = reinterpret_cast<OperandBase*>(cSelf);

            self->Reference();
        }

        void NativeOperandRelease(WebnnOperand cSelf) {
            auto self = reinterpret_cast<OperandBase*>(cSelf);

            self->Release();
        }

        const void* NativeResultBuffer(WebnnResult cSelf) {
            auto self = reinterpret_cast<ResultBase*>(cSelf);

            auto result =            self->Buffer();
            return result;
        }

        uint32_t NativeResultBufferSize(WebnnResult cSelf) {
            auto self = reinterpret_cast<ResultBase*>(cSelf);

            auto result =            self->BufferSize();
            return result;
        }

        const int32_t* NativeResultDimensions(WebnnResult cSelf) {
            auto self = reinterpret_cast<ResultBase*>(cSelf);

            auto result =            self->Dimensions();
            return result;
        }

        uint32_t NativeResultDimensionsSize(WebnnResult cSelf) {
            auto self = reinterpret_cast<ResultBase*>(cSelf);

            auto result =            self->DimensionsSize();
            return result;
        }

        void NativeResultReference(WebnnResult cSelf) {
            auto self = reinterpret_cast<ResultBase*>(cSelf);

            self->Reference();
        }

        void NativeResultRelease(WebnnResult cSelf) {
            auto self = reinterpret_cast<ResultBase*>(cSelf);

            self->Release();
        }

        struct ProcEntry {
            WebnnProc proc;
            const char* name;
        };
        static const ProcEntry sProcMap[] = {
            { reinterpret_cast<WebnnProc>(NativeCompilationCompute), "webnnCompilationCompute" },
            { reinterpret_cast<WebnnProc>(NativeCompilationReference), "webnnCompilationReference" },
            { reinterpret_cast<WebnnProc>(NativeCompilationRelease), "webnnCompilationRelease" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderAdd), "webnnModelBuilderAdd" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderAveragePool2d), "webnnModelBuilderAveragePool2d" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderBatchNorm), "webnnModelBuilderBatchNorm" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderClamp), "webnnModelBuilderClamp" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderConcat), "webnnModelBuilderConcat" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderConstant), "webnnModelBuilderConstant" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderConv2d), "webnnModelBuilderConv2d" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderCreateModel), "webnnModelBuilderCreateModel" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderGemm), "webnnModelBuilderGemm" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderInput), "webnnModelBuilderInput" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderLeakyRelu), "webnnModelBuilderLeakyRelu" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderMatmul), "webnnModelBuilderMatmul" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderMaxPool2d), "webnnModelBuilderMaxPool2d" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderMul), "webnnModelBuilderMul" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderReference), "webnnModelBuilderReference" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderRelease), "webnnModelBuilderRelease" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderRelu), "webnnModelBuilderRelu" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderReshape), "webnnModelBuilderReshape" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderSoftmax), "webnnModelBuilderSoftmax" },
            { reinterpret_cast<WebnnProc>(NativeModelBuilderTranspose), "webnnModelBuilderTranspose" },
            { reinterpret_cast<WebnnProc>(NativeModelCompile), "webnnModelCompile" },
            { reinterpret_cast<WebnnProc>(NativeModelReference), "webnnModelReference" },
            { reinterpret_cast<WebnnProc>(NativeModelRelease), "webnnModelRelease" },
            { reinterpret_cast<WebnnProc>(NativeNamedInputsReference), "webnnNamedInputsReference" },
            { reinterpret_cast<WebnnProc>(NativeNamedInputsRelease), "webnnNamedInputsRelease" },
            { reinterpret_cast<WebnnProc>(NativeNamedInputsSet), "webnnNamedInputsSet" },
            { reinterpret_cast<WebnnProc>(NativeNamedOperandsReference), "webnnNamedOperandsReference" },
            { reinterpret_cast<WebnnProc>(NativeNamedOperandsRelease), "webnnNamedOperandsRelease" },
            { reinterpret_cast<WebnnProc>(NativeNamedOperandsSet), "webnnNamedOperandsSet" },
            { reinterpret_cast<WebnnProc>(NativeNamedOutputsReference), "webnnNamedOutputsReference" },
            { reinterpret_cast<WebnnProc>(NativeNamedOutputsRelease), "webnnNamedOutputsRelease" },
            { reinterpret_cast<WebnnProc>(NativeNamedOutputsSet), "webnnNamedOutputsSet" },
            { reinterpret_cast<WebnnProc>(NativeNamedResultsGet), "webnnNamedResultsGet" },
            { reinterpret_cast<WebnnProc>(NativeNamedResultsReference), "webnnNamedResultsReference" },
            { reinterpret_cast<WebnnProc>(NativeNamedResultsRelease), "webnnNamedResultsRelease" },
            { reinterpret_cast<WebnnProc>(NativeNeuralNetworkContextCreateModelBuilder), "webnnNeuralNetworkContextCreateModelBuilder" },
            { reinterpret_cast<WebnnProc>(NativeNeuralNetworkContextPopErrorScope), "webnnNeuralNetworkContextPopErrorScope" },
            { reinterpret_cast<WebnnProc>(NativeNeuralNetworkContextPushErrorScope), "webnnNeuralNetworkContextPushErrorScope" },
            { reinterpret_cast<WebnnProc>(NativeNeuralNetworkContextReference), "webnnNeuralNetworkContextReference" },
            { reinterpret_cast<WebnnProc>(NativeNeuralNetworkContextRelease), "webnnNeuralNetworkContextRelease" },
            { reinterpret_cast<WebnnProc>(NativeNeuralNetworkContextSetUncapturedErrorCallback), "webnnNeuralNetworkContextSetUncapturedErrorCallback" },
            { reinterpret_cast<WebnnProc>(NativeOperandReference), "webnnOperandReference" },
            { reinterpret_cast<WebnnProc>(NativeOperandRelease), "webnnOperandRelease" },
            { reinterpret_cast<WebnnProc>(NativeResultBuffer), "webnnResultBuffer" },
            { reinterpret_cast<WebnnProc>(NativeResultBufferSize), "webnnResultBufferSize" },
            { reinterpret_cast<WebnnProc>(NativeResultDimensions), "webnnResultDimensions" },
            { reinterpret_cast<WebnnProc>(NativeResultDimensionsSize), "webnnResultDimensionsSize" },
            { reinterpret_cast<WebnnProc>(NativeResultReference), "webnnResultReference" },
            { reinterpret_cast<WebnnProc>(NativeResultRelease), "webnnResultRelease" },
        };
        static constexpr size_t sProcMapSize = sizeof(sProcMap) / sizeof(sProcMap[0]);
    }

    std::vector<const char*> GetProcMapNamesForTestingInternal() {
        std::vector<const char*> result;
        result.reserve(sProcMapSize);
        for (const ProcEntry& entry : sProcMap) {
            result.push_back(entry.name);
        }
        return result;
    }

    WebnnNamedInputs NativeCreateNamedInputs() {
        return reinterpret_cast<WebnnNamedInputs>(new NamedInputsBase());
    }

    WebnnNamedOperands NativeCreateNamedOperands() {
         return reinterpret_cast<WebnnNamedOperands>(new NamedOperandsBase());
    }

    WebnnNamedOutputs NativeCreateNamedOutputs() {
         return reinterpret_cast<WebnnNamedOutputs>(new NamedOutputsBase());
    }

    static WebnnProcTable gProcTable = {
        NativeCreateNamedInputs,
        NativeCreateNamedOperands,
        NativeCreateNamedOutputs,
        NativeCompilationCompute,
        NativeCompilationReference,
        NativeCompilationRelease,
        NativeModelCompile,
        NativeModelReference,
        NativeModelRelease,
        NativeModelBuilderAdd,
        NativeModelBuilderAveragePool2d,
        NativeModelBuilderBatchNorm,
        NativeModelBuilderClamp,
        NativeModelBuilderConcat,
        NativeModelBuilderConstant,
        NativeModelBuilderConv2d,
        NativeModelBuilderCreateModel,
        NativeModelBuilderGemm,
        NativeModelBuilderInput,
        NativeModelBuilderLeakyRelu,
        NativeModelBuilderMatmul,
        NativeModelBuilderMaxPool2d,
        NativeModelBuilderMul,
        NativeModelBuilderRelu,
        NativeModelBuilderReshape,
        NativeModelBuilderSoftmax,
        NativeModelBuilderTranspose,
        NativeModelBuilderReference,
        NativeModelBuilderRelease,
        NativeNamedInputsSet,
        NativeNamedInputsReference,
        NativeNamedInputsRelease,
        NativeNamedOperandsSet,
        NativeNamedOperandsReference,
        NativeNamedOperandsRelease,
        NativeNamedOutputsSet,
        NativeNamedOutputsReference,
        NativeNamedOutputsRelease,
        NativeNamedResultsGet,
        NativeNamedResultsReference,
        NativeNamedResultsRelease,
        NativeNeuralNetworkContextCreateModelBuilder,
        NativeNeuralNetworkContextPopErrorScope,
        NativeNeuralNetworkContextPushErrorScope,
        NativeNeuralNetworkContextSetUncapturedErrorCallback,
        NativeNeuralNetworkContextReference,
        NativeNeuralNetworkContextRelease,
        NativeOperandReference,
        NativeOperandRelease,
        NativeResultBuffer,
        NativeResultBufferSize,
        NativeResultDimensions,
        NativeResultDimensionsSize,
        NativeResultReference,
        NativeResultRelease,
    };

    const WebnnProcTable& GetProcsAutogen() {
        return gProcTable;
    }

}
