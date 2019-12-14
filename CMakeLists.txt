cmake_minimum_required(VERSION 2.8)

project(pyretinanetcpp)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(DENABLE_PYTHON3 ON)
set(static_pbcvt ON)

option(PYTHON_VERSION "3.6")               # for ubuntu 16.04 == 3.5
option(BOOST_PYTHON_VERSION "python-py3.6") # for ubuntu 18.04 == python-py36

# retinanet
# ===========================================
find_package(CUDA REQUIRED)
enable_language(CUDA)

if(DEFINED TensorRT_DIR)
  include_directories("${TensorRT_DIR}/include")
  link_directories("${TensorRT_DIR}/lib")
endif(DEFINED TensorRT_DIR)
include_directories(${CUDA_INCLUDE_DIRS})
# ===========================================

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(PythonLibs ${PYTHON_VERSION} REQUIRED)
include_directories(${PythonLibs_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(Boost REQUIRED COMPONENTS system ${BOOST_PYTHON_VERSION})
include_directories(${Boost_INCLUDE_DIRS})

include_directories(libs/pyboostcvconverter/include/)
add_subdirectory(libs/pyboostcvconverter)
add_library(pyretinanetcpp SHARED src/pyretinanetcpp.cpp)

# retinanet
# ===========================================
add_library(retinanet
    src/tensorrt/cuda/decode.h
    src/tensorrt/cuda/decode.cu
    src/tensorrt/cuda/nms.h
    src/tensorrt/cuda/nms.cu
    src/tensorrt/cuda/utils.h
    src/tensorrt/engine.h
    src/tensorrt/engine.cpp
    src/tensorrt/calibrator.h
)
set_target_properties(retinanet PROPERTIES
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
target_link_libraries(retinanet PUBLIC nvinfer nvonnxparser ${OPENCV_LIBS})
# ===========================================


target_link_libraries(pyretinanetcpp PRIVATE
    rt
    cuda
    pbcvt
    retinanet
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
)

# Suppress prefix "lib" because Python does not allow this prefix
set_target_properties(pyretinanetcpp PROPERTIES PREFIX "")

#install(TARGETS pyboostcvconverter EXPORT pyboostcvconverter)
# install(TARGETS tensorrt_retinanet EXPORT tensorrt_retinanet DESTINATION lib/tsm/)
install(TARGETS pyretinanetcpp EXPORT pyretinanetcpp DESTINATION lib/retinanet)

install(CODE "execute_process(COMMAND /bin/bash -c \"cp -v ${CMAKE_CURRENT_BINARY_DIR}/libs/pyboostcvconverter/pbcvt.cpython-3* /usr/lib/ \")")
install(CODE "execute_process(COMMAND /bin/bash -c \"  cd ${CMAKE_CURRENT_SOURCE_DIR}/python/; python3 setup.py install; cd ${CMAKE_CURRENT_BINARY_DIR};\")")
