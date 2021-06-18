
#ifndef MOCK_WEBNN_H
#define MOCK_WEBNN_H

#include <webnn/webnn_proc_table.h>
#include <webnn/webnn.h>
#include <gmock/gmock.h>

#include <memory>

// An abstract base class representing a proc table so that API calls can be mocked. Most API calls
// are directly represented by a delete virtual method but others need minimal state tracking to be
// useful as mocks.
class ProcTableAsClass {
    public:
        virtual ~ProcTableAsClass();

        void GetProcTableAndDevice(WebnnProcTable* table);

        // Creates an object that can be returned by a mocked call as in WillOnce(Return(foo)).
        // It returns an object of the write type that isn't equal to any previously returned object.
        // Otherwise some mock expectation could be triggered by two different objects having the same
        // value.
        WebnnCompilation GetNewCompilation();
        WebnnModel GetNewModel();
        WebnnModelBuilder GetNewModelBuilder();
        WebnnNamedInputs GetNewNamedInputs();
        WebnnNamedOperands GetNewNamedOperands();
        WebnnNamedOutputs GetNewNamedOutputs();
        WebnnNamedResults GetNewNamedResults();
        WebnnNeuralNetworkContext GetNewNeuralNetworkContext();
        WebnnOperand GetNewOperand();
        WebnnResult GetNewResult();


        virtual void CompilationReference(WebnnCompilation self) = 0;
        virtual void CompilationRelease(WebnnCompilation self) = 0;

        void CompilationCompute(WebnnCompilation compilation, WebnnNamedInputs inputs, WebnnComputeCallback callback, void * userdata, WebnnNamedOutputs outputs);
        virtual void OnCompilationCompute(WebnnCompilation compilation, WebnnNamedInputs inputs, WebnnComputeCallback callback, void * userdata, WebnnNamedOutputs outputs) = 0;

        void CallCompilationComputeCallback(WebnnCompilation compilation, WebnnComputeStatus status, WebnnNamedResults outputs, char const * message);

        virtual void ModelReference(WebnnModel self) = 0;
        virtual void ModelRelease(WebnnModel self) = 0;

        void ModelCompile(WebnnModel model, WebnnCompileCallback callback, void * userdata, WebnnCompilationOptions const * options);
        virtual void OnModelCompile(WebnnModel model, WebnnCompileCallback callback, void * userdata, WebnnCompilationOptions const * options) = 0;

        void CallModelCompileCallback(WebnnModel model, WebnnCompileStatus status, WebnnCompilation compilation, char const * message);
        virtual WebnnOperand ModelBuilderAdd(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b) = 0;
        virtual WebnnOperand ModelBuilderAveragePool2d(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options) = 0;
        virtual WebnnOperand ModelBuilderBatchNorm(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand mean, WebnnOperand variance, WebnnBatchNormOptions const * options) = 0;
        virtual WebnnOperand ModelBuilderClamp(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnClampOptions const * options) = 0;
        virtual WebnnOperand ModelBuilderConcat(WebnnModelBuilder modelBuilder, uint32_t inputsCount, WebnnOperand const * inputs, uint32_t axis) = 0;
        virtual WebnnOperand ModelBuilderConstant(WebnnModelBuilder modelBuilder, WebnnOperandDescriptor const * desc, void const * value, size_t size) = 0;
        virtual WebnnOperand ModelBuilderConv2d(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand filter, WebnnConv2dOptions const * options) = 0;
        virtual WebnnModel ModelBuilderCreateModel(WebnnModelBuilder modelBuilder, WebnnNamedOperands namedOperands) = 0;
        virtual WebnnOperand ModelBuilderGemm(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b, WebnnGemmOptions const * options) = 0;
        virtual WebnnOperand ModelBuilderInput(WebnnModelBuilder modelBuilder, char const * name, WebnnOperandDescriptor const * desc) = 0;
        virtual WebnnOperand ModelBuilderLeakyRelu(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnLeakyReluOptions const * options) = 0;
        virtual WebnnOperand ModelBuilderMatmul(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b) = 0;
        virtual WebnnOperand ModelBuilderMaxPool2d(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options) = 0;
        virtual WebnnOperand ModelBuilderMul(WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b) = 0;
        virtual WebnnOperand ModelBuilderRelu(WebnnModelBuilder modelBuilder, WebnnOperand input) = 0;
        virtual WebnnOperand ModelBuilderReshape(WebnnModelBuilder modelBuilder, WebnnOperand input, int32_t const * newShape, uint32_t newShapeCount) = 0;
        virtual WebnnOperand ModelBuilderSoftmax(WebnnModelBuilder modelBuilder, WebnnOperand input) = 0;
        virtual WebnnOperand ModelBuilderTranspose(WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnTransposeOptions const * options) = 0;

        virtual void ModelBuilderReference(WebnnModelBuilder self) = 0;
        virtual void ModelBuilderRelease(WebnnModelBuilder self) = 0;

        virtual void NamedInputsSet(WebnnNamedInputs namedInputs, char const * name, WebnnInput const * input) = 0;

        virtual void NamedInputsReference(WebnnNamedInputs self) = 0;
        virtual void NamedInputsRelease(WebnnNamedInputs self) = 0;

        virtual void NamedOperandsSet(WebnnNamedOperands namedOperands, char const * name, WebnnOperand operand) = 0;

        virtual void NamedOperandsReference(WebnnNamedOperands self) = 0;
        virtual void NamedOperandsRelease(WebnnNamedOperands self) = 0;

        virtual void NamedOutputsSet(WebnnNamedOutputs namedOutputs, char const * name, WebnnOutput const * output) = 0;

        virtual void NamedOutputsReference(WebnnNamedOutputs self) = 0;
        virtual void NamedOutputsRelease(WebnnNamedOutputs self) = 0;

        virtual WebnnResult NamedResultsGet(WebnnNamedResults namedResults, char const * name) = 0;

        virtual void NamedResultsReference(WebnnNamedResults self) = 0;
        virtual void NamedResultsRelease(WebnnNamedResults self) = 0;

        virtual WebnnModelBuilder NeuralNetworkContextCreateModelBuilder(WebnnNeuralNetworkContext neuralNetworkContext) = 0;
        virtual void NeuralNetworkContextPushErrorScope(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorFilter filter) = 0;

        virtual void NeuralNetworkContextReference(WebnnNeuralNetworkContext self) = 0;
        virtual void NeuralNetworkContextRelease(WebnnNeuralNetworkContext self) = 0;

        bool NeuralNetworkContextPopErrorScope(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata);
        virtual bool OnNeuralNetworkContextPopErrorScope(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata) = 0;

        void CallNeuralNetworkContextPopErrorScopeCallback(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorType type, char const * message);
        void NeuralNetworkContextSetUncapturedErrorCallback(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata);
        virtual void OnNeuralNetworkContextSetUncapturedErrorCallback(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata) = 0;

        void CallNeuralNetworkContextSetUncapturedErrorCallbackCallback(WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorType type, char const * message);

        virtual void OperandReference(WebnnOperand self) = 0;
        virtual void OperandRelease(WebnnOperand self) = 0;

        virtual const void* ResultBuffer(WebnnResult result) = 0;
        virtual uint32_t ResultBufferSize(WebnnResult result) = 0;
        virtual const int32_t* ResultDimensions(WebnnResult result) = 0;
        virtual uint32_t ResultDimensionsSize(WebnnResult result) = 0;

        virtual void ResultReference(WebnnResult self) = 0;
        virtual void ResultRelease(WebnnResult self) = 0;


        struct Object {
            ProcTableAsClass* procs = nullptr;
            WebnnComputeCallback mCompilationComputeCallback = nullptr;
            WebnnCompileCallback mModelCompileCallback = nullptr;
            WebnnErrorCallback mNeuralNetworkContextPopErrorScopeCallback = nullptr;
            WebnnErrorCallback mNeuralNetworkContextSetUncapturedErrorCallbackCallback = nullptr;
            void* userdata = 0;
        };

    private:
        // Remembers the values returned by GetNew* so they can be freed.
        std::vector<std::unique_ptr<Object>> mObjects;
};

class MockProcTable : public ProcTableAsClass {
    public:
        MockProcTable();
        ~MockProcTable() override;

        void IgnoreAllReleaseCalls();


        MOCK_METHOD(void, CompilationReference, (WebnnCompilation self), (override));
        MOCK_METHOD(void, CompilationRelease, (WebnnCompilation self), (override));

        MOCK_METHOD(void, OnCompilationCompute, (WebnnCompilation compilation, WebnnNamedInputs inputs, WebnnComputeCallback callback, void * userdata, WebnnNamedOutputs outputs), (override));

        MOCK_METHOD(void, ModelReference, (WebnnModel self), (override));
        MOCK_METHOD(void, ModelRelease, (WebnnModel self), (override));

        MOCK_METHOD(void, OnModelCompile, (WebnnModel model, WebnnCompileCallback callback, void * userdata, WebnnCompilationOptions const * options), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderAdd, (WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderAveragePool2d, (WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderBatchNorm, (WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand mean, WebnnOperand variance, WebnnBatchNormOptions const * options), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderClamp, (WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnClampOptions const * options), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderConcat, (WebnnModelBuilder modelBuilder, uint32_t inputsCount, WebnnOperand const * inputs, uint32_t axis), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderConstant, (WebnnModelBuilder modelBuilder, WebnnOperandDescriptor const * desc, void const * value, size_t size), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderConv2d, (WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnOperand filter, WebnnConv2dOptions const * options), (override));
        MOCK_METHOD(WebnnModel, ModelBuilderCreateModel, (WebnnModelBuilder modelBuilder, WebnnNamedOperands namedOperands), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderGemm, (WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b, WebnnGemmOptions const * options), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderInput, (WebnnModelBuilder modelBuilder, char const * name, WebnnOperandDescriptor const * desc), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderLeakyRelu, (WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnLeakyReluOptions const * options), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderMatmul, (WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderMaxPool2d, (WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnPool2dOptions const * options), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderMul, (WebnnModelBuilder modelBuilder, WebnnOperand a, WebnnOperand b), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderRelu, (WebnnModelBuilder modelBuilder, WebnnOperand input), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderReshape, (WebnnModelBuilder modelBuilder, WebnnOperand input, int32_t const * newShape, uint32_t newShapeCount), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderSoftmax, (WebnnModelBuilder modelBuilder, WebnnOperand input), (override));
        MOCK_METHOD(WebnnOperand, ModelBuilderTranspose, (WebnnModelBuilder modelBuilder, WebnnOperand input, WebnnTransposeOptions const * options), (override));

        MOCK_METHOD(void, ModelBuilderReference, (WebnnModelBuilder self), (override));
        MOCK_METHOD(void, ModelBuilderRelease, (WebnnModelBuilder self), (override));

        MOCK_METHOD(void, NamedInputsSet, (WebnnNamedInputs namedInputs, char const * name, WebnnInput const * input), (override));

        MOCK_METHOD(void, NamedInputsReference, (WebnnNamedInputs self), (override));
        MOCK_METHOD(void, NamedInputsRelease, (WebnnNamedInputs self), (override));

        MOCK_METHOD(void, NamedOperandsSet, (WebnnNamedOperands namedOperands, char const * name, WebnnOperand operand), (override));

        MOCK_METHOD(void, NamedOperandsReference, (WebnnNamedOperands self), (override));
        MOCK_METHOD(void, NamedOperandsRelease, (WebnnNamedOperands self), (override));

        MOCK_METHOD(void, NamedOutputsSet, (WebnnNamedOutputs namedOutputs, char const * name, WebnnOutput const * output), (override));

        MOCK_METHOD(void, NamedOutputsReference, (WebnnNamedOutputs self), (override));
        MOCK_METHOD(void, NamedOutputsRelease, (WebnnNamedOutputs self), (override));

        MOCK_METHOD(WebnnResult, NamedResultsGet, (WebnnNamedResults namedResults, char const * name), (override));

        MOCK_METHOD(void, NamedResultsReference, (WebnnNamedResults self), (override));
        MOCK_METHOD(void, NamedResultsRelease, (WebnnNamedResults self), (override));

        MOCK_METHOD(WebnnModelBuilder, NeuralNetworkContextCreateModelBuilder, (WebnnNeuralNetworkContext neuralNetworkContext), (override));
        MOCK_METHOD(void, NeuralNetworkContextPushErrorScope, (WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorFilter filter), (override));

        MOCK_METHOD(void, NeuralNetworkContextReference, (WebnnNeuralNetworkContext self), (override));
        MOCK_METHOD(void, NeuralNetworkContextRelease, (WebnnNeuralNetworkContext self), (override));

        MOCK_METHOD(bool, OnNeuralNetworkContextPopErrorScope, (WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata), (override));
        MOCK_METHOD(void, OnNeuralNetworkContextSetUncapturedErrorCallback, (WebnnNeuralNetworkContext neuralNetworkContext, WebnnErrorCallback callback, void * userdata), (override));

        MOCK_METHOD(void, OperandReference, (WebnnOperand self), (override));
        MOCK_METHOD(void, OperandRelease, (WebnnOperand self), (override));

        MOCK_METHOD(const void*, ResultBuffer, (WebnnResult result), (override));
        MOCK_METHOD(uint32_t, ResultBufferSize, (WebnnResult result), (override));
        MOCK_METHOD(const int32_t*, ResultDimensions, (WebnnResult result), (override));
        MOCK_METHOD(uint32_t, ResultDimensionsSize, (WebnnResult result), (override));

        MOCK_METHOD(void, ResultReference, (WebnnResult self), (override));
        MOCK_METHOD(void, ResultRelease, (WebnnResult self), (override));

};

#endif  // MOCK_WEBNN_H
