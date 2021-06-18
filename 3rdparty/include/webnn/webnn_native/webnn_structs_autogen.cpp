
#include "webnn_native/webnn_structs_autogen.h"

namespace webnn_native {


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


    static_assert(sizeof(ClampOptions) == sizeof(WebnnClampOptions), "sizeof mismatch for ClampOptions");
    static_assert(alignof(ClampOptions) == alignof(WebnnClampOptions), "alignof mismatch for ClampOptions");

    static_assert(offsetof(ClampOptions, minValue) == offsetof(WebnnClampOptions, minValue),
            "offsetof mismatch for ClampOptions::minValue");
    static_assert(offsetof(ClampOptions, maxValue) == offsetof(WebnnClampOptions, maxValue),
            "offsetof mismatch for ClampOptions::maxValue");


    static_assert(sizeof(CompilationOptions) == sizeof(WebnnCompilationOptions), "sizeof mismatch for CompilationOptions");
    static_assert(alignof(CompilationOptions) == alignof(WebnnCompilationOptions), "alignof mismatch for CompilationOptions");

    static_assert(offsetof(CompilationOptions, powerPreference) == offsetof(WebnnCompilationOptions, powerPreference),
            "offsetof mismatch for CompilationOptions::powerPreference");


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


    static_assert(sizeof(LeakyReluOptions) == sizeof(WebnnLeakyReluOptions), "sizeof mismatch for LeakyReluOptions");
    static_assert(alignof(LeakyReluOptions) == alignof(WebnnLeakyReluOptions), "alignof mismatch for LeakyReluOptions");

    static_assert(offsetof(LeakyReluOptions, alpha) == offsetof(WebnnLeakyReluOptions, alpha),
            "offsetof mismatch for LeakyReluOptions::alpha");


    static_assert(sizeof(OperandDescriptor) == sizeof(WebnnOperandDescriptor), "sizeof mismatch for OperandDescriptor");
    static_assert(alignof(OperandDescriptor) == alignof(WebnnOperandDescriptor), "alignof mismatch for OperandDescriptor");

    static_assert(offsetof(OperandDescriptor, type) == offsetof(WebnnOperandDescriptor, type),
            "offsetof mismatch for OperandDescriptor::type");
    static_assert(offsetof(OperandDescriptor, dimensions) == offsetof(WebnnOperandDescriptor, dimensions),
            "offsetof mismatch for OperandDescriptor::dimensions");
    static_assert(offsetof(OperandDescriptor, dimensionsCount) == offsetof(WebnnOperandDescriptor, dimensionsCount),
            "offsetof mismatch for OperandDescriptor::dimensionsCount");


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


    static_assert(sizeof(TransposeOptions) == sizeof(WebnnTransposeOptions), "sizeof mismatch for TransposeOptions");
    static_assert(alignof(TransposeOptions) == alignof(WebnnTransposeOptions), "alignof mismatch for TransposeOptions");

    static_assert(offsetof(TransposeOptions, permutationCount) == offsetof(WebnnTransposeOptions, permutationCount),
            "offsetof mismatch for TransposeOptions::permutationCount");
    static_assert(offsetof(TransposeOptions, permutation) == offsetof(WebnnTransposeOptions, permutation),
            "offsetof mismatch for TransposeOptions::permutation");

}
