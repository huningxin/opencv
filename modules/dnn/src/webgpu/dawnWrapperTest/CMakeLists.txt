cmake_minimum_required(VERSION 3.10)

project(softmaxTest  VERSION 1.0)

find_package(OpenCV REQUIRED)
set(OPENCV_SOURCE_DIR "/home/webml/GSoC/Forked/opencv")
set(WEBGPU_INCLUDE_DIRS "${OPENCV_SOURCE_DIR}/3rdparty/include/webgpu/include")
set(WEBGPU_LIBRARIES "${OPENCV_SOURCE_DIR}/3rdparty/include/webgpu/lib")

file(GLOB SOURCES
    ${OPENCV_SOURCE_DIR}/modules/dnn/src/webgpu/include/*.h
    ${OPENCV_SOURCE_DIR}/modules/dnn/src/webgpu/src/*.cpp
)
file(GLOB Dawn
    ${OPENCV_SOURCE_DIR}/modules/dnn/src/webgpu/dawn/*.h
    ${OPENCV_SOURCE_DIR}/modules/dnn/src/webgpu/dawn/*.cpp
)
file(GLOB DNN
    ${OPENCV_SOURCE_DIR}/modules/dnn/include/opencv2/dnn/dnn.hpp
    ${OPENCV_SOURCE_DIR}/modules/dnn/src/webgpu/dawn/*.cpp
)

add_executable(softmaxTest op_softmax_test.cpp  
                ${SOURCES} 
                ${Dawn} 
                ${DNN}
                ${OPENCV_SOURCE_DIR}/modules/dnn/src/webgpu/shader/softmax_spv.cpp )
set_property(TARGET softmaxTest PROPERTY CXX_STANDARD 11)

include_directories(${WEBGPU_INCLUDE_DIRS} ${OPENCV_SOURCE_DIR}/modules/dnn/src/webgpu/include ${OPENCV_SOURCE_DIR}/modules/dnn/src/webgpu/dawn)
target_link_directories(softmaxTest PUBLIC ${WEBGPU_LIBRARIES}) 
target_link_libraries(softmaxTest PUBLIC  dawn_wire dawn_proc dawn_native ${OpenCV_LIBS})
