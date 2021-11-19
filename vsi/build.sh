#! /bin/bash

mkdir -p build
cd build && rm * -rf
cmake .. -DTIM_VX_INSTALL=/root/yzw/mlperf/vsi/TIM-VX/build/install -DLITE_DRIVER_INSTALL=/root/yzw/mlperf/vsi/TIM-VX/lite_vim3_sdk -DVX_DRIVER_INSTALL=/root/yzw/mlperf/vsi/TIM-VX/build/aarch64_A311D_6.4.8 -DLOADGEN_LIB_DIR=/root/yzw/mlperf/vsi -DOPENCV_INSTALL=/root/yzw/mlperf/vsi/opencv_install_vim3 -DLOADGEN_DIR=/root/yzw/mlperf/loadgen

make mlperf_runner
cp mlperf_runner ../ && cd ..

