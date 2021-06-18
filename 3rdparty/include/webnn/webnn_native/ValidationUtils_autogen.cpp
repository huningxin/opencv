
#include "webnn_native/ValidationUtils_autogen.h"

namespace webnn_native {

    MaybeError ValidateCompileStatus(webnn::CompileStatus value) {
        switch (value) {
            case webnn::CompileStatus::Success:
                return {};
            case webnn::CompileStatus::Error:
                return {};
            case webnn::CompileStatus::ContextLost:
                return {};
            case webnn::CompileStatus::Unknown:
                return {};
            default:
                return DAWN_VALIDATION_ERROR("Invalid value for WebnnCompileStatus");
        }
    }

    MaybeError ValidateComputeStatus(webnn::ComputeStatus value) {
        switch (value) {
            case webnn::ComputeStatus::Success:
                return {};
            case webnn::ComputeStatus::Error:
                return {};
            case webnn::ComputeStatus::ContextLost:
                return {};
            case webnn::ComputeStatus::Unknown:
                return {};
            default:
                return DAWN_VALIDATION_ERROR("Invalid value for WebnnComputeStatus");
        }
    }

    MaybeError ValidateErrorFilter(webnn::ErrorFilter value) {
        switch (value) {
            case webnn::ErrorFilter::None:
                return {};
            case webnn::ErrorFilter::Validation:
                return {};
            case webnn::ErrorFilter::OutOfMemory:
                return {};
            default:
                return DAWN_VALIDATION_ERROR("Invalid value for WebnnErrorFilter");
        }
    }

    MaybeError ValidateErrorType(webnn::ErrorType value) {
        switch (value) {
            case webnn::ErrorType::NoError:
                return {};
            case webnn::ErrorType::Validation:
                return {};
            case webnn::ErrorType::OutOfMemory:
                return {};
            case webnn::ErrorType::Unknown:
                return {};
            case webnn::ErrorType::DeviceLost:
                return {};
            default:
                return DAWN_VALIDATION_ERROR("Invalid value for WebnnErrorType");
        }
    }

    MaybeError ValidateOperandLayout(webnn::OperandLayout value) {
        switch (value) {
            case webnn::OperandLayout::Nchw:
                return {};
            case webnn::OperandLayout::Nhwc:
                return {};
            default:
                return DAWN_VALIDATION_ERROR("Invalid value for WebnnOperandLayout");
        }
    }

    MaybeError ValidateOperandType(webnn::OperandType value) {
        switch (value) {
            case webnn::OperandType::Float32:
                return {};
            case webnn::OperandType::Float16:
                return {};
            case webnn::OperandType::Int32:
                return {};
            case webnn::OperandType::Uint32:
                return {};
            default:
                return DAWN_VALIDATION_ERROR("Invalid value for WebnnOperandType");
        }
    }

    MaybeError ValidatePowerPreference(webnn::PowerPreference value) {
        switch (value) {
            case webnn::PowerPreference::Default:
                return {};
            case webnn::PowerPreference::Low_power:
                return {};
            case webnn::PowerPreference::High_performance:
                return {};
            default:
                return DAWN_VALIDATION_ERROR("Invalid value for WebnnPowerPreference");
        }
    }



} // namespace webnn_native
