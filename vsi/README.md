# VSI MLPerf Runner 

## Prepare Environment

To build mlperf runner, we need prebuild the following software on specific platform:

* [TIM-VX](https://github.com/VeriSilicon/TIM-VX)
* VSI OpenVX Driver
* VSI Lite Driver
* OPenCV
* LoadGen (We can build LoadGen referring to https://github.com/mlcommons/inference/blob/master/loadgen/README_BUILD.md)

## Build MLPerf Runner

```bash
mkdir -p build
cd build && rm * -rf
cmake .. -DTIM_VX_INSTALL=/root/yzw/mlperf/vsi/TIM-VX/build/install -DLITE_DRIVER_INSTALL=/root/yzw/mlperf/vsi/TIM-VX/lite_vim3_sdk -DVX_DRIVER_INSTALL=/root/yzw/mlperf/vsi/TIM-VX/build/aarch64_A311D_6.4.8 -DLOADGEN_LIB_DIR=/root/yzw/mlperf/vsi -DOPENCV_INSTALL=/root/yzw/mlperf/vsi/opencv_install_vim3 -DLOADGEN_DIR=/root/yzw/mlperf/loadgen

make mlperf_runner
cp mlperf_runner ../ && cd ..
```

## Test MLPerf Runner

We can specify the test scenario, test mode and samples number by setting the relative parameters. To learn about detailed usage, we can run:

```bash
./mlperf_runner --help
Usage: mlperf_run [options]
Options:
--image_dir         The directory to store imagenet validation images.
--nbg_file          The ngb file path.
--enable_trace      Enable log trace.
--num_samples       The number of samples.
--scenario          The test scenaroi, must be one of SingleStream, MultiStream, Server, Offline.
--mode              The test mode, must be one of SubmissionRun, AccuracyOnly, PerformanceOnly.
--help              Print help message and exit.
```

Here is a sample to run mlperf_run:
```bash
export LD_LIBRARY_PATH=/root/yzw/mlperf/vsi/TIM-VX/build/aarch64_A311D_6.4.8/lib:/root/yzw/mlperf/vsi/TIM-VX/lite_vim3_sdk/drivers:/root/yzw/mlperf/vsi/TIM-VX/build/install/lib:/root/yzw/mlperf/vsi/opencv_install_vim3/lib

./mlperf_runner --image_dir dataset/imagenet_val/ --nbg_file mobilenetv3-large_224_1.0_uint8  --enable_trace --num_samples 100 --scenario SingleStream --mode SubmissionRun
```

After mlperf_runner finished, the typical mlperf log files would be generated, and we can use the following command to calculate the accuracy.

```bash
python ../../vision/classification_and_detection/tools/accuracy-imagenet.py --imagenet-val-file=../dataset/val_map.txt --mlperf-accuracy-file=mlperf_log_accuracy.json | tee accuracy.txt
```