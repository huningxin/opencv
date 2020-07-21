cmake_minimum_required(VERSION 3.10)
project(WEBGPU VERSION 1.0)
set(OPENCV_SOURCE_DIR "/home/webml/GSoC/Forked/opencv")

set(WEBGPU_INCLUDE_DIRS "${OPENCV_SOURCE_DIR}/3rdparty/include/webgpu/include")
set(WEBGPU_LIBRARIES "${OPENCV_SOURCE_DIR}/3rdparty/include/webgpu/lib")

add_executable(WEBGPU helloCompute.cpp ${WEBGPU_LIBRARIES}/webgpu_cpp.cpp)
set_property(TARGET WEBGPU PROPERTY CXX_STANDARD 14)
include_directories(${WEBGPU_INCLUDE_DIRS} )
target_link_directories(WEBGPU PUBLIC ${WEBGPU_LIBRARIES})
target_link_libraries(WEBGPU PUBLIC  dawn_wire dawn_proc dawn_native )