#include <getopt.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>

#include "loadgen.h"
#include "query_sample_library.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "tim/lite/execution.h"
#include "tim/lite/handle.h"

static unsigned int load_file(const char *name, void *dst) {
  FILE *fp = fopen(name, "rb");
  unsigned int size = 0;

  if (fp != NULL) {
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);

    fseek(fp, 0, SEEK_SET);
    size = fread(dst, size, 1, fp);

    fclose(fp);
  }

  return size;
}

static unsigned int get_file_size(const char *name) {
  FILE *fp = fopen(name, "rb");
  unsigned int size = 0;

  if (fp != NULL) {
    fseek(fp, 0, SEEK_END);
    size = ftell(fp);

    fclose(fp);
  } else {
    printf("Checking file %s failed.\n", name);
  }

  return size;
}

float Top1(const uint8_t *d, int size) {
  assert(d && size > 0);
  std::priority_queue<std::pair<float, int>> q;

  for (auto i = 0; i < size; ++i) {
    q.push(std::pair<float, int>(d[i], i));
  }

  return q.top().second;
}

#define MEM_ALIGN(x, align) (((x) + ((align)-1)) & ~((align)-1))
class InputDataGen {
 public:
  static InputDataGen &GetIml() {
    static InputDataGen data_gen;
    return data_gen;
  }

  bool Init(std::string image_dir, std::shared_ptr<tim::lite::Execution> exe) {
    vsi_exe_ = exe;
    std::string val_map_file = "dataset/val_map.txt";
    std::ifstream f;
    f.open(val_map_file.c_str());
    if (!f.is_open()) {
      std::cout << "Error: cannot open image list file " << val_map_file
                << std::endl;
      return false;
    }

    std::string image_file_name;
    uint32_t label;
    for (int i = 0; i < 5000 && !f.eof(); i++) {
      f >> image_file_name;
      f >> label;
      val_image_list_.push_back(
          std::make_pair(label, image_dir + image_file_name));
    }
    return true;
  }

  void Load(const mlperf::QuerySampleIndex &idx, uint32_t width,
            uint32_t height) {
    uint32_t size = width * height * 3;
    std::string file_name = val_image_list_[idx].second;
    uint8_t *input_ptr = (uint8_t *)aligned_alloc(64, MEM_ALIGN(size, 64));

    // const float mean[3] = {123.68, 116.78, 103.94};
    // const float scaleIn = 0.5;

    const float mean[3] = {127.5, 127.5, 127.5};
    const float scaleIn = 0.007843;

    std::cout << "QuerySampleIndex: " << idx << " image_file: " << file_name
              << " lable: " << val_image_list_[idx].first << std::endl;

    cv::Mat orig = cv::imread(file_name);
    cv::cvtColor(orig, orig, cv::COLOR_BGR2RGB);
    float scale = 0.0;
    if (orig.rows > orig.cols)
      scale = 256.0 / orig.cols;
    else
      scale = 256.0 / orig.rows;
    int new_h = round(orig.rows * scale);
    int new_w = round(orig.cols * scale);

    cv::Mat resizedImage = cv::Mat(new_h, new_w, CV_8SC3);
    cv::resize(orig, resizedImage, cv::Size(new_w, new_h), 0, 0,
               cv::INTER_AREA);

    /// Center Crop Image
    const int offsetW = (new_w - width) / 2;
    const int offsetH = (new_h - height) / 2;

    const cv::Rect roi(offsetW, offsetH, width, height);
    resizedImage = resizedImage(roi).clone();

    for (uint32_t h = 0; h < height; h++)
      for (uint32_t w = 0; w < width; w++)
        for (uint32_t c = 0; c < 3; c++) {
          input_ptr[0 + (3 * h * width) + (3 * w) + c] =
              (resizedImage.at<cv::Vec3b>(h, w)[c] - mean[c]) * scaleIn /
                  0.007812 +
              128;
        }
    input_data_[idx].first = input_ptr;
    input_data_[idx].second =
        vsi_exe_->CreateInputHandle(0, input_ptr, MEM_ALIGN(size, 64));
  }

  std::shared_ptr<tim::lite::Handle> &GetInputHandle(
      const mlperf::QuerySampleIndex &idx) {
    if (input_data_.find(idx) != input_data_.end()) {
      return input_data_[idx].second;
    }
    std::cout << "Input handle has not been prepared." << std::endl;
    assert(false);
  }

  void Unload(const mlperf::QuerySampleIndex &idx) {
    free(input_data_[idx].first);
  }

 private:
  std::vector<std::pair<uint32_t /* lable */, std::string /* file name */>>
      val_image_list_;
  std::shared_ptr<tim::lite::Execution> vsi_exe_ = nullptr;
  std::map<mlperf::QuerySampleIndex,
           std::pair<uint8_t *, std::shared_ptr<tim::lite::Handle>>>
      input_data_;
};

class VSISystemUnderTest : public mlperf::SystemUnderTest {
 public:
  VSISystemUnderTest(std::string nbg_file, std::string image_dir) {
    uint32_t nbg_size = get_file_size(nbg_file.c_str());
    nbg_data_ = malloc(nbg_size);
    load_file(nbg_file.c_str(), nbg_data_);
    vsi_exe_ = tim::lite::Execution::Create(nbg_data_, nbg_size);
    if (!vsi_exe_) {
      std::cout << "Load executable fail." << std::endl;
      assert(false);
    }
    InputDataGen &data_gen = InputDataGen::GetIml();
    data_gen.Init(image_dir, vsi_exe_);

    output_size_ = 1001 * sizeof(uint8_t);
    output_ = (uint8_t *)aligned_alloc(64, MEM_ALIGN(output_size_, 64));
    memset(output_, 0, output_size_);

    output_handle_ =
        vsi_exe_->CreateOutputHandle(0, output_, MEM_ALIGN(output_size_, 64));
    vsi_exe_->BindOutputs({output_handle_});
  }

  const std::string &Name() const override { return name_; }

  void IssueQuery(const std::vector<mlperf::QuerySample> &samples) override {
    InputDataGen &data_gen = InputDataGen::GetIml();
    result_response_.resize(samples.size());
    top1_.resize(samples.size());

    for (uint32_t i = 0; i < samples.size(); i++) {
      auto input_handle = data_gen.GetInputHandle(samples[i].index);
      input_handle->Flush();
      vsi_exe_->BindInputs({input_handle});
      vsi_exe_->Trigger();
      output_handle_->Invalidate();
      top1_[i] = Top1(output_, 1001) - 1;
      result_response_[i].id = samples[i].id;
      result_response_[i].data = (uintptr_t)(&(top1_[i]));
      result_response_[i].size = sizeof(top1_[i]);
      std::cout << "SampleIndex: " << samples[i].index << " TOP1: " << top1_[i]
                << std::endl;
      vsi_exe_->UnBindInput(input_handle);
    }
    mlperf::QuerySamplesComplete(result_response_.data(),
                                 result_response_.size());
  }

  void FlushQueries() override {}

  void ReportLatencyResults(
      const std::vector<mlperf::QuerySampleLatency> &latencies_ns) override {}

  ~VSISystemUnderTest() {
    free(nbg_data_);
    free(output_);
  }

 private:
  std::string name_ = "vsi-npu";
  void *nbg_data_ = nullptr;
  std::shared_ptr<tim::lite::Execution> vsi_exe_ = nullptr;
  uint8_t *output_ = nullptr;
  uint32_t output_size_ = 0;
  std::shared_ptr<tim::lite::Handle> output_handle_ = nullptr;
  std::vector<mlperf::QuerySampleResponse> result_response_;
  std::vector<float> top1_;
};

class VSIQuerySampleLibrary : public mlperf::QuerySampleLibrary {
 public:
  VSIQuerySampleLibrary(uint32_t sample_num, uint32_t heigh, uint32_t width)
      : sample_num_(sample_num), height_(heigh), width_(width) {}

  ~VSIQuerySampleLibrary() {}

  const std::string &Name() const override { return name_; }

  size_t TotalSampleCount() override { return sample_num_; }

  size_t PerformanceSampleCount() override {
    return sample_num_ < 5000 ? sample_num_ : 5000;
  }

  void LoadSamplesToRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override {
    std::cout << "LoadSamplesToRam" << std::endl;
    InputDataGen &data_gen = InputDataGen::GetIml();
    for (const auto &s : samples) {
      data_gen.Load(s, width_, height_);
    }
  }

  void UnloadSamplesFromRam(
      const std::vector<mlperf::QuerySampleIndex> &samples) override {
    std::cout << "UnloadSamplesFromRam" << std::endl;
    InputDataGen &data_gen = InputDataGen::GetIml();
    for (const auto &s : samples) {
      data_gen.Unload(s);
    }
  }

 private:
  std::string name_ = "vsi-npu";
  uint32_t sample_num_;
  uint32_t height_;
  uint32_t width_;
};

void ShowUsage() {
  std::cout << "Usage: mlperf_run [options]" << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "--image_dir         The directory to store imagenet validation "
               "images."
            << std::endl;
  std::cout << "--nbg_file          The ngb file path." << std::endl;
  std::cout << "--enable_trace      Enable log trace." << std::endl;
  std::cout << "--num_samples       The number of samples." << std::endl;
  std::cout << "--scenario          The test scenaroi, must be one of "
               "SingleStream, MultiStream, Server, Offline."
            << std::endl;
  std::cout << "--mode              The test mode, must be one of "
               "SubmissionRun, AccuracyOnly, PerformanceOnly."
            << std::endl;
  std::cout << "--help              Print help message and exit." << std::endl;
}

int main(int argc, char **argv) {
  std::string image_dir;
  std::string nbg_file;
  bool enbale_trace = false;
  uint32_t num_samples = 0;
  std::string scenario = "SingleStream";
  std::string mode = "AccuracyOnly";
  int32_t val;
  for (;;) {
    struct option long_options[] = {{"image_dir", required_argument, &val, 1},
                                    {"nbg_file", required_argument, &val, 2},
                                    {"enable_trace", no_argument, 0, 3},
                                    {"num_samples", required_argument, &val, 4},
                                    {"scenario", required_argument, &val, 5},
                                    {"mode", required_argument, &val, 6},
                                    {"help", no_argument, 0, 7},
                                    {0, 0, 0, 0}};
    /* getopt_long stores the option index here. */
    int32_t option_index = 0;
    int c = getopt_long(argc, argv, "", long_options, &option_index);
    if (c == 3) {
      enbale_trace = true;
    } else if (c == 0) {
      switch (val) {
        case 1:
          image_dir = std::string(optarg);
          break;
        case 2:
          nbg_file = std::string(optarg);
          break;
        case 4:
          num_samples = std::stoi(std::string(optarg));
          break;
        case 5:
          scenario = std::string(optarg);
          break;
        case 6:
          mode = std::string(optarg);
          break;
      }
    } else if (c == 7) {
      ShowUsage();
      return 0;
    } else {
      break;
    }
  }
  auto test_setting = mlperf::TestSettings();
  auto log_setting = mlperf::LogSettings();
  if (scenario == "SingleStream") {
    test_setting.scenario = mlperf::TestScenario::SingleStream;
  } else if (scenario == "MultiStream") {
    test_setting.scenario = mlperf::TestScenario::MultiStream;
  } else if (scenario == "Offline") {
    test_setting.scenario = mlperf::TestScenario::Offline;
  } else if (scenario == "Server") {
    test_setting.scenario = mlperf::TestScenario::Server;
  } else {
    std::cout << "Unkown TestScenario." << std::endl;
    return -1;
  }

  if (mode == "AccuracyOnly") {
    test_setting.mode = mlperf::TestMode::AccuracyOnly;
  } else if (mode == "PerformanceOnly") {
    test_setting.mode = mlperf::TestMode::PerformanceOnly;
  } else if (mode == "SubmissionRun") {
    test_setting.mode = mlperf::TestMode::SubmissionRun;
  } else {
    std::cout << "Unkown TestMode";
    return -1;
  }

  if (enbale_trace) {
    log_setting.enable_trace = true;
  }

  auto sut = std::make_unique<VSISystemUnderTest>(nbg_file, image_dir);
  auto qsl = std::make_unique<VSIQuerySampleLibrary>(num_samples, 224, 224);
  mlperf::StartTest(sut.get(), qsl.get(), test_setting, log_setting);
  return 0;
}