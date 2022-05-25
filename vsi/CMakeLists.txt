cmake_minimum_required (VERSION 3.14)
project(mlperf_runner LANGUAGES C CXX)

set(TARGET_NAME "mlperf_runner")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pthread")

aux_source_directory(./ SRC)

if(NOT TIM_VX_INSTALL OR NOT EXISTS ${TIM_VX_INSTALL})
  message(FATAL_ERROR "Set TIM_VX_INSTALL to tim-vx install folder.")
endif()

if(NOT LITE_DRIVER_INSTALL OR NOT EXISTS ${LITE_DRIVER_INSTALL})
  message(FATAL_ERROR "Set LITE_DRIVER_INSTALL to lite driver sdk install folder.")
endif()

if(NOT VX_DRIVER_INSTALL OR NOT EXISTS ${VX_DRIVER_INSTALL})
  message(FATAL_ERROR "Set VX_DRIVER_INSTALL to openvx driver sdk install folder.")
endif()

if(NOT LOADGEN_DIR OR NOT EXISTS ${LOADGEN_DIR})
  message(FATAL_ERROR "Set LOADGEN_LIB_DIR to libmlperf_loadgen.a folder.")
endif()

if(NOT OPENCV_INSTALL OR NOT EXISTS ${OPENCV_INSTALL})
  message(FATAL_ERROR "Set OPENCV_INSTALL to opencv install folder.")
endif()

set(OPENCV_INC ${OPENCV_INSTALL}/include/opencv4)
set(TIM_VX_INC ${TIM_VX_INSTALL}/include)
set(TIM_VX_LIB_DIR ${TIM_VX_INSTALL}/lib)
set(LITE_DRIVER_DIR ${LITE_DRIVER_INSTALL}/drivers)
set(VX_DRIVER_DIR ${VX_DRIVER_INSTALL}/lib)
set(OPENCV_LIB_DIR ${OPENCV_INSTALL}/lib)
set(LOADGEN_LIB_DIR ${LOADGEN_DIR}/build/)

link_directories(${TIM_VX_LIB_DIR} ${LITE_DRIVER_DIR} ${VX_DRIVER_DIR} ${OPENCV_LIB_DIR} ${LOADGEN_LIB_DIR})

add_executable(${TARGET_NAME} ${SRC})
target_link_libraries(${TARGET_NAME} PRIVATE
                        VIPlite
                        OpenVX
                        tim-vx
                        opencv_imgcodecs opencv_core opencv_highgui opencv_imgproc
                        mlperf_loadgen)
target_include_directories(${TARGET_NAME} PRIVATE
    ${OPENCV_INC}
    ${TIM_VX_INC}
    ${LOADGEN_DIR}
)

