// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_MMDEPLOY_POSE_DETECTOR_HPP_
#define MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_MMDEPLOY_POSE_DETECTOR_HPP_

#include "mmdeploy/common.hpp"
#include "mmdeploy/pose_detector.h"

namespace mmdeploy {

using PoseDetection = mmdeploy_pose_detection_t;

class PoseDetector : public NonMovable {
 public:
  PoseDetector(const Model& model, const Device& device) {
    auto ec = mmdeploy_pose_detector_create(model, device.name(), device.index(), &detector_);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }
  }

  ~PoseDetector() {
    if (detector_) {
      mmdeploy_pose_detector_destroy(detector_);
      detector_ = {};
    }
  }

  using Result = Result_<PoseDetection>;

  std::vector<Result> Apply(Span<const Mat> images, Span<const Rect> bboxes,
                            Span<const int> bbox_count) {
    if (images.empty()) {
      return {};
    }
    auto mats = GetMats(images);

    const mmdeploy_rect_t* p_bboxes{};
    const int* p_bbox_count{};

    if (!bboxes.empty()) {
      p_bboxes = bboxes.data();
      p_bbox_count = bbox_count.data();
    }

    PoseDetection* results{};
    auto ec = mmdeploy_pose_detector_apply_bbox(
        detector_, mats.data(), static_cast<int>(mats.size()), p_bboxes, p_bbox_count, &results);
    if (ec != MMDEPLOY_SUCCESS) {
      throw_exception(static_cast<ErrorCode>(ec));
    }

    std::shared_ptr<PoseDetection> data(results, [count = mats.size()](auto p) {
      mmdeploy_pose_detector_release_result(p, count);
    });

    std::vector<Result> rets;
    rets.reserve(images.size());

    size_t offset = 0;
    for (size_t i = 0; i < mats.size(); ++i) {
      offset += rets.emplace_back(offset, bboxes.empty() ? 1 : bbox_count[i], data).size();
    }

    return rets;
  }

  Result Apply(const Mat& image, Span<const Rect> bboxes = {}) {
    return Apply(Span{image}, bboxes, {static_cast<int>(bboxes.size())})[0];
  }

 private:
  mmdeploy_pose_detector_t detector_{};
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_MMDEPLOY_APIS_CXX_MMDEPLOY_POSE_DETECTOR_HPP_
