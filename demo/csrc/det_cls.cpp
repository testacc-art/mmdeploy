
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/logger.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/core/value.h"
#include "mmdeploy/experimental/module_adapter.h"
#include "opencv2/imgcodecs.hpp"
#include "mmdeploy/pipeline.h"

const auto config_json = R"(
{
  "type": "Pipeline",
  "input": "img",
  "output": ["dets", "labels"],
  "tasks": [
    {
      "type": "Inference",
      "input": "img",
      "output": "dets",
      "params": { "model": "../_detection_tmp_model" }
    },
    {
      "type": "Pipeline",
      "input": ["boxes=*dets", "imgs=+img"],
      "tasks": [
        {
          "type": "Task",
          "module": "CropBox",
          "input": ["imgs", "boxes"],
          "output": "patches"
        },
        {
          "type": "Inference",
          "input": "patches",
          "output": "labels",
          "params": { "model": "../_mmcls_tmp_model" }
        }
      ],
      "output": "*labels"
    }
  ]
}
)"_json;

using namespace mmdeploy;

class CropBox {
 public:
  Result<Value> operator()(const Value& img, const Value& dets) {
    auto patch = img["ori_img"].get<Mat>();
    if (dets.is_object() && dets.contains("bbox")) {
      auto _box = from_value<std::vector<float>>(dets["bbox"]);
      cv::Rect rect(cv::Rect2f(cv::Point2f(_box[0], _box[1]), cv::Point2f(_box[2], _box[3])));
      patch = crop(patch, rect);
    }
    return Value{{"ori_img", patch}};
  }

 private:
  static Mat crop(const Mat& img, cv::Rect rect) {
    cv::Mat mat(img.height(), img.width(), CV_8UC(img.channel()), img.data<void>());
    rect &= cv::Rect(cv::Point(0, 0), mat.size());
    mat = mat(rect).clone();
    std::shared_ptr<void> data(mat.data, [mat = mat](void*) {});
    return Mat{mat.rows, mat.cols, img.pixel_format(), img.type(), std::move(data)};
  }
};

class CropBoxCreator : public Creator<Module> {
 public:
  const char* GetName() const override { return "CropBox"; }
  std::unique_ptr<Module> Create(const Value& value) override { return CreateTask(CropBox{}); }
};

REGISTER_MODULE(Module, CropBoxCreator);

int main() {
  auto config = from_json<Value>(config_json);

  mmdeploy_pipeline_t pipeline{};
  if (auto ec =
          mmdeploy_pipeline_create((mmdeploy_value_t)&config, "cuda", 0, nullptr, &pipeline)) {
    MMDEPLOY_ERROR("failed to create pipeline: {}", ec);
    return -1;
  }

  cv::Mat mat = cv::imread("../demo.jpg");
  mmdeploy::Mat img(mat.rows, mat.cols, PixelFormat::kBGR, DataType::kINT8, mat.data, Device(0));

  Value input = Value::Array{Value::Array{Value::Object{{"ori_img", img}}}};

  mmdeploy_value_t tmp{};
  mmdeploy_pipeline_apply(pipeline, (mmdeploy_value_t)&input, &tmp);

  auto output = std::move(*(Value*)tmp);
  mmdeploy_value_destroy(tmp);

  MMDEPLOY_INFO("{}", output);

  mmdeploy_pipeline_destroy(pipeline);
}
