
#ifndef BACKEND_VALIDATIONUTILS_H_
#define BACKEND_VALIDATIONUTILS_H_

#include "webnn/webnn_cpp.h"

#include "webnn_native/Error.h"

namespace webnn_native {

    // Helper functions to check the value of enums and bitmasks
    MaybeError ValidateCompileStatus(webnn::CompileStatus value);
    MaybeError ValidateComputeStatus(webnn::ComputeStatus value);
    MaybeError ValidateErrorFilter(webnn::ErrorFilter value);
    MaybeError ValidateErrorType(webnn::ErrorType value);
    MaybeError ValidateOperandLayout(webnn::OperandLayout value);
    MaybeError ValidateOperandType(webnn::OperandType value);
    MaybeError ValidatePowerPreference(webnn::PowerPreference value);

} // namespace webnn_native

#endif  // BACKEND_VALIDATIONUTILS_H_
