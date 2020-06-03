if(HAVE_WEBGPU OR WITH_WEBGPU)
  set(WEBGPU_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/3rdparty/include/webgpu/include")
  set(WEBGPU_LIBRARIES "${PROJECT_SOURCE_DIR}/3rdparty/include/webgpu/lib")
endif()

if(HAVE_WEBGPU OR WITH_WEBGPU)
  include_directories(${WEBGPU_INCLUDE_DIRS})
  link_directories(${WEBGPU_LIBRARIES})
  link_libraries(dawn_proc dawn_native dawn_wire)

endif()

try_compile(VALID_WEBGPU
      "${OpenCV_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/webgpu.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${WEBGPU_INCLUDE_DIRS}"
      OUTPUT_VARIABLE TRY_OUT
      )
if(NOT ${VALID_WEBGPU})
  message(WARNING "Can't use WebGPU-Dawn")
  return()
endif()

set(HAVE_WEBGPU 1)