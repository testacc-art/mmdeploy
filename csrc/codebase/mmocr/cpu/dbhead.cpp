// Copyright (c) OpenMMLab. All rights reserved.

#include "codebase/mmocr/dbnet.h"
#include "core/utils/device_utils.h"
#include "opencv2/imgcodecs.hpp"

namespace mmdeploy::mmocr {

class DbHeadCpuImpl : public DbHeadImpl {
 public:
  void Init(const DbHeadParams& params, const Stream& stream) override {
    DbHeadImpl::Init(params, stream);
    device_ = Device("cpu");
  }

  Result<void> Process(Tensor prob, std::vector<std::vector<cv::Point>>& points,
                       std::vector<float>& scores) override {
    OUTCOME_TRY(auto conf, MakeAvailableOnDevice(prob, device_, stream_));
    OUTCOME_TRY(stream_.Wait());

    auto h = conf.shape(1);
    auto w = conf.shape(2);
    auto data = conf.data<float>();

    cv::Mat score_map((int)h, (int)w, CV_32F, data);

    // cv::imwrite("conf.png", score_map * 255.);

    cv::Mat text_mask = score_map >= params_.mask_thr;
    // cv::imwrite("text_mask.png", text_mask);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(text_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    if (contours.size() > params_.max_candidates) {
      contours.resize(params_.max_candidates);
    }

    for (auto& poly : contours) {
      auto epsilon = 0.01 * cv::arcLength(poly, true);
      std::vector<cv::Point> approx;
      cv::approxPolyDP(poly, approx, epsilon, true);
      if (approx.size() < 4) {
        continue;
      }
      auto score = box_score_fast(score_map, approx);

      points.push_back(approx);
      scores.push_back(score);
    }

    return success();
  }

  static float box_score_fast(const cv::Mat& bitmap, const std::vector<cv::Point>& box) noexcept {
    auto rect = cv::boundingRect(box) & cv::Rect({}, bitmap.size());

    cv::Mat mask(rect.size(), CV_8U, cv::Scalar(0));

    cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{box}, 1, cv::LINE_8, 0, -rect.tl());
    auto mean = cv::mean(bitmap(rect), mask)[0];
    return static_cast<float>(mean);
  }

  Device device_;
};

class DbHeadCpuImplCreator : public ::mmdeploy::Creator<DbHeadImpl> {
 public:
  const char* GetName() const override { return "cpu"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<DbHeadImpl> Create(const Value&) override {
    return std::make_unique<DbHeadCpuImpl>();
  }
};

REGISTER_MODULE(DbHeadImpl, DbHeadCpuImplCreator);

}  // namespace mmdeploy::mmocr
