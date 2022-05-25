#! /bin/bash

export LD_LIBRARY_PATH=/root/yzw/mlperf/vsi/TIM-VX/build/aarch64_A311D_6.4.8/lib:/root/yzw/mlperf/vsi/TIM-VX/lite_vim3_sdk/drivers:/root/yzw/mlperf/vsi/TIM-VX/build/install/lib:/root/yzw/mlperf/vsi/opencv_install_vim3/lib

./mlperf_runner --image_dir dataset/imagenet_val/ --nbg_file mobilenetv3-large_224_1.0_uint8  --enable_trace --num_samples 100 --scenario SingleStream --mode SubmissionRun

