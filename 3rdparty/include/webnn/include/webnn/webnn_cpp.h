#ifndef WEBNN_CPP_H_
#define WEBNN_CPP_H_

#include "webnn/webnn.h"
#include "webnn/EnumClassBitmasks.h"

namespace webnn {

    enum class CompileStatus : uint32_t {
        Success = 0x00000000,
        Error = 0x00000001,
        ContextLost = 0x00000002,
        Unknown = 0x00000003,
    };

    enum class ComputeStatus : uint32_t {
        Success = 0x00000000,
        Error = 0x00000001,
        ContextLost = 0x00000002,
        Unknown = 0x00000003,
    };

    enum class ErrorFilter : uint32_t {
        None = 0x00000000,
        Validation = 0x00000001,
        OutOfMemory = 0x00000002,
    };

    enum class ErrorType : uint32_t {
        NoError = 0x00000000,
        Validation = 0x00000001,
        OutOfMemory = 0x00000002,
        Unknown = 0x00000003,
        DeviceLost = 0x00000004,
    };

    enum class OperandLayout : uint32_t {
        Nchw = 0x00000000,
        Nhwc = 0x00000001,
    };

    enum class OperandType : uint32_t {
        Float32 = 0x00000000,
        Float16 = 0x00000001,
        Int32 = 0x00000002,
        Uint32 = 0x00000003,
    };

    enum class PowerPreference : uint32_t {
        Default = 0x00000000,
        Low_power = 0x00000001,
        High_performance = 0x00000002,
    };




    using Proc = WebnnProc;
    using CompileCallback = WebnnCompileCallback;
    using ComputeCallback = WebnnComputeCallback;
    using ErrorCallback = WebnnErrorCallback;

    class Compilation;
    class Model;
    class ModelBuilder;
    class NamedInputs;
    class NamedOperands;
    class NamedOutputs;
    class NamedResults;
    class NeuralNetworkContext;
    class Operand;
    class Result;

    struct BatchNormOptions;
    struct ClampOptions;
    struct CompilationOptions;
    struct Conv2dOptions;
    struct GemmOptions;
    struct Input;
    struct LeakyReluOptions;
    struct OperandDescriptor;
    struct Output;
    struct Pool2dOptions;
    struct TransposeOptions;

    template<typename Derived, typename CType>
    class ObjectBase {
      public:
        ObjectBase() = default;
        ObjectBase(CType handle): mHandle(handle) {
            if (mHandle) Derived::WebnnReference(mHandle);
        }
        ~ObjectBase() {
            if (mHandle) Derived::WebnnRelease(mHandle);
        }

        ObjectBase(ObjectBase const& other)
            : ObjectBase(other.GetHandle()) {
        }
        Derived& operator=(ObjectBase const& other) {
            if (&other != this) {
                if (mHandle) Derived::WebnnRelease(mHandle);
                mHandle = other.mHandle;
                if (mHandle) Derived::WebnnReference(mHandle);
            }

            return static_cast<Derived&>(*this);
        }

        ObjectBase(ObjectBase&& other) {
            mHandle = other.mHandle;
            other.mHandle = 0;
        }
        Derived& operator=(ObjectBase&& other) {
            if (&other != this) {
                if (mHandle) Derived::WebnnRelease(mHandle);
                mHandle = other.mHandle;
                other.mHandle = 0;
            }

            return static_cast<Derived&>(*this);
        }

        ObjectBase(std::nullptr_t) {}
        Derived& operator=(std::nullptr_t) {
            if (mHandle != nullptr) {
                Derived::WebnnRelease(mHandle);
                mHandle = nullptr;
            }
            return static_cast<Derived&>(*this);
        }

        bool operator==(std::nullptr_t) const {
            return mHandle == nullptr;
        }
        bool operator!=(std::nullptr_t) const {
            return mHandle != nullptr;
        }

        explicit operator bool() const {
            return mHandle != nullptr;
        }
        CType GetHandle() const {
            return mHandle;
        }
        CType Release() {
            CType result = mHandle;
            mHandle = 0;
            return result;
        }
        static Derived Acquire(CType handle) {
            Derived result;
            result.mHandle = handle;
            return result;
        }

      protected:
        CType mHandle = nullptr;
    };



    class Compilation : public ObjectBase<Compilation, WebnnCompilation> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;

        void Compute(NamedInputs const& inputs, ComputeCallback callback, void * userdata, NamedOutputs const& outputs) const;

      private:
        friend ObjectBase<Compilation, WebnnCompilation>;
        static void WebnnReference(WebnnCompilation handle);
        static void WebnnRelease(WebnnCompilation handle);
    };

    class Model : public ObjectBase<Model, WebnnModel> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;

        void Compile(CompileCallback callback, void * userdata, CompilationOptions const * options = nullptr) const;

      private:
        friend ObjectBase<Model, WebnnModel>;
        static void WebnnReference(WebnnModel handle);
        static void WebnnRelease(WebnnModel handle);
    };

    class ModelBuilder : public ObjectBase<ModelBuilder, WebnnModelBuilder> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;

        Operand Add(Operand const& a, Operand const& b) const;
        Operand AveragePool2d(Operand const& input, Pool2dOptions const * options = nullptr) const;
        Operand BatchNorm(Operand const& input, Operand const& mean, Operand const& variance, BatchNormOptions const * options = nullptr) const;
        Operand Clamp(Operand const& input, ClampOptions const * options = nullptr) const;
        Operand Concat(uint32_t inputsCount, Operand const * inputs, uint32_t axis) const;
        Operand Constant(OperandDescriptor const * desc, void const * value, size_t size) const;
        Operand Conv2d(Operand const& input, Operand const& filter, Conv2dOptions const * options = nullptr) const;
        Model CreateModel(NamedOperands const& namedOperands) const;
        Operand Gemm(Operand const& a, Operand const& b, GemmOptions const * options = nullptr) const;
        Operand Input(char const * name, OperandDescriptor const * desc) const;
        Operand LeakyRelu(Operand const& input, LeakyReluOptions const * options = nullptr) const;
        Operand Matmul(Operand const& a, Operand const& b) const;
        Operand MaxPool2d(Operand const& input, Pool2dOptions const * options = nullptr) const;
        Operand Mul(Operand const& a, Operand const& b) const;
        Operand Relu(Operand const& input) const;
        Operand Reshape(Operand const& input, int32_t const * newShape, uint32_t newShapeCount) const;
        Operand Softmax(Operand const& input) const;
        Operand Transpose(Operand const& input, TransposeOptions const * options = nullptr) const;

      private:
        friend ObjectBase<ModelBuilder, WebnnModelBuilder>;
        static void WebnnReference(WebnnModelBuilder handle);
        static void WebnnRelease(WebnnModelBuilder handle);
    };

    class NamedInputs : public ObjectBase<NamedInputs, WebnnNamedInputs> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;

        void Set(char const * name, Input const * input) const;

      private:
        friend ObjectBase<NamedInputs, WebnnNamedInputs>;
        static void WebnnReference(WebnnNamedInputs handle);
        static void WebnnRelease(WebnnNamedInputs handle);
    };

    class NamedOperands : public ObjectBase<NamedOperands, WebnnNamedOperands> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;

        void Set(char const * name, Operand const& operand) const;

      private:
        friend ObjectBase<NamedOperands, WebnnNamedOperands>;
        static void WebnnReference(WebnnNamedOperands handle);
        static void WebnnRelease(WebnnNamedOperands handle);
    };

    class NamedOutputs : public ObjectBase<NamedOutputs, WebnnNamedOutputs> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;

        void Set(char const * name, Output const * output) const;

      private:
        friend ObjectBase<NamedOutputs, WebnnNamedOutputs>;
        static void WebnnReference(WebnnNamedOutputs handle);
        static void WebnnRelease(WebnnNamedOutputs handle);
    };

    class NamedResults : public ObjectBase<NamedResults, WebnnNamedResults> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;

        Result Get(char const * name) const;

      private:
        friend ObjectBase<NamedResults, WebnnNamedResults>;
        static void WebnnReference(WebnnNamedResults handle);
        static void WebnnRelease(WebnnNamedResults handle);
    };

    class NeuralNetworkContext : public ObjectBase<NeuralNetworkContext, WebnnNeuralNetworkContext> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;

        ModelBuilder CreateModelBuilder() const;
        bool PopErrorScope(ErrorCallback callback, void * userdata) const;
        void PushErrorScope(ErrorFilter filter) const;
        void SetUncapturedErrorCallback(ErrorCallback callback, void * userdata) const;

      private:
        friend ObjectBase<NeuralNetworkContext, WebnnNeuralNetworkContext>;
        static void WebnnReference(WebnnNeuralNetworkContext handle);
        static void WebnnRelease(WebnnNeuralNetworkContext handle);
    };

    class Operand : public ObjectBase<Operand, WebnnOperand> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;


      private:
        friend ObjectBase<Operand, WebnnOperand>;
        static void WebnnReference(WebnnOperand handle);
        static void WebnnRelease(WebnnOperand handle);
    };

    class Result : public ObjectBase<Result, WebnnResult> {
      public:
        using ObjectBase::ObjectBase;
        using ObjectBase::operator=;

        const void* Buffer() const;
        uint32_t BufferSize() const;
        const int32_t* Dimensions() const;
        uint32_t DimensionsSize() const;

      private:
        friend ObjectBase<Result, WebnnResult>;
        static void WebnnReference(WebnnResult handle);
        static void WebnnRelease(WebnnResult handle);
    };


    struct ChainedStruct {
        ChainedStruct const * nextInChain = nullptr;
        // SType sType = SType::Invalid;
    };

    struct BatchNormOptions {
        Operand scale;
        Operand bias;
        uint32_t axis = 1;
        float epsilon = 1e-05;
    };

    struct ClampOptions {
        Operand minValue;
        Operand maxValue;
    };

    struct CompilationOptions {
        PowerPreference powerPreference;
    };

    struct Conv2dOptions {
        uint32_t paddingCount = 0;
        int32_t const * padding = nullptr;
        uint32_t stridesCount = 0;
        int32_t const * strides = nullptr;
        uint32_t dilationsCount = 0;
        int32_t const * dilations = nullptr;
        int32_t groups = 1;
        OperandLayout layout = OperandLayout::Nchw;
    };

    struct GemmOptions {
        Operand c;
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
        OperandType type;
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
        OperandLayout layout = OperandLayout::Nchw;
    };

    struct TransposeOptions {
        uint32_t permutationCount = 0;
        int32_t const * permutation = nullptr;
    };


    NamedInputs CreateNamedInputs();
    NamedOperands CreateNamedOperands();
    NamedOutputs CreateNamedOutputs();

}  // namespace webnn

#endif // WEBNN_CPP_H_
