ocv_clear_vars(HAVE_WEBNN)
if(WITH_WEBNN)
  set(WEBNN_HEADER_DIRS "$ENV{WEBNN_NATIVE_DIR}/out/Release/gen/src/include/webnn")
  set(WEBNN_INCLUDE_DIRS "$ENV{WEBNN_NATIVE_DIR}/src/include")
  set(WEBNN_LIBRARIES "$ENV{WEBNN_NATIVE_DIR}/out/Release/gen/src/webnn")
endif()

try_compile(VALID_WEBNN
      "${OpenCV_BINARY_DIR}"
      "${OpenCV_SOURCE_DIR}/cmake/checks/webnn.cpp"
      CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${WEBNN_INCLUDE_DIRS}\;${WEBNN_HEADER_DIRS}"
                  "-DLINK_LIBRARIES:STRING=${WEBNN_LIBRARIES}"
      OUTPUT_VARIABLE TRY_OUT
      )
if(NOT ${VALID_WEBNN})
  message(WARNING "Can't use WebNN-native")
  return()
endif()
message(AUTHOR_WARNING "Use WebNN-native")

set(HAVE_WEBNN 1)

if(NOT EMSCRIPTEN AND HAVE_WEBGPU)
  include_directories(${WEBNN_INCLUDE_DIRS} ${WEBNN_HEADER_DIRS})
  link_directories(${WEBNN_LIBRARIES})
endif()