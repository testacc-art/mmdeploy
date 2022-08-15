// Copyright (c) OpenMMLab. All rights reserved.

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/net.h"
#include "mmdeploy/core/utils/formatter.h"
#include "rknn_api.h"

namespace mmdeploy {

static inline Result<void> _m(int rk_err, SourceLocation loc = SourceLocation::current()) {
  if (rk_err == RKNN_SUCC) {
    return success();
  } else {
    return Status(eFail, loc);
  }
}

static inline Result<DataType> FromRknnTensorType(rknn_tensor_type type) {
  switch (type) {
    case RKNN_TENSOR_FLOAT32:
      return DataType::kFLOAT;
    case RKNN_TENSOR_FLOAT16:
      return DataType::kHALF;
    case RKNN_TENSOR_INT8:
      return DataType::kINT8;
    case RKNN_TENSOR_INT32:
      return DataType::kINT32;
    case RKNN_TENSOR_INT64:
      return DataType::kINT64;
    default:
      return Status(eNotSupported);
  }
}

static inline Result<rknn_tensor_type> ToRknnTensorType(DataType type) {
  switch (type) {
    case DataType::kFLOAT:
      return RKNN_TENSOR_FLOAT32;
    case DataType::kHALF:
      return RKNN_TENSOR_FLOAT16;
    case DataType::kINT8:
      return RKNN_TENSOR_INT8;
    case DataType::kINT32:
      return RKNN_TENSOR_INT32;
    case DataType::kINT64:
      return RKNN_TENSOR_INT64;
    default:
      return Status(eNotSupported);
  }
}

static inline Result<TensorDesc> ToDesc(const rknn_tensor_attr& attr,
                                        SourceLocation loc = SourceLocation::current()) {
  OUTCOME_TRY(auto data_type, FromRknnTensorType(attr.type));
  TensorDesc desc{Device(0), data_type, TensorShape{&attr.dims[0], &attr.dims[0] + attr.n_dims},
                  attr.name};
  return desc;
}

class RknnNet : public Net {
 public:
  ~RknnNet() override {
    if (context_) {
      rknn_destroy(context_);
    }
  }

  Result<void> Init(const Value& args) override {
    auto& context = args["context"];
    device_ = context["device"].get<Device>();
    stream_ = context["stream"].get<Stream>();

    auto name = args["name"].get<std::string>();
    auto model = context["model"].get<Model>();

    OUTCOME_TRY(auto config, model.GetModelConfig(name));
    OUTCOME_TRY(auto bytes, model.ReadFile(config.net));

    OUTCOME_TRY(_m(rknn_init(&context_, bytes.data(), bytes.size(), 0, nullptr)));

    rknn_input_output_num io_num{};
    OUTCOME_TRY(_m(rknn_query(context_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num))));

    for (uint32_t i = 0; i < io_num.n_input; ++i) {
      rknn_tensor_attr attr{};
      attr.index = i;
      OUTCOME_TRY(_m(rknn_query(context_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr))));
      OUTCOME_TRY(auto desc, ToDesc(attr));
      input_tensor_.emplace_back(std::move(desc));
      input_attr_.push_back(attr);
    }

    for (uint32_t i = 0; i < io_num.n_output; ++i) {
      rknn_tensor_attr attr{};
      attr.index = i;
      OUTCOME_TRY(_m(rknn_query(context_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr))));
      OUTCOME_TRY(auto desc, ToDesc(attr));
      output_tensor_.emplace_back(std::move(desc));
      output_attr_.push_back(attr);
    }

    return success();
  }

  Result<void> Deinit() override { return success(); }

  Result<Span<Tensor>> GetInputTensors() override { return input_tensor_; }

  Result<Span<Tensor>> GetOutputTensors() override { return output_tensor_; }

  Result<void> Reshape(Span<TensorShape> input_shapes) override { return success(); }

  Result<void> Forward() override {
    std::vector<rknn_input> inputs;
    inputs.reserve(input_tensor_.size());
    for (int i = 0; i < input_tensor_.size(); ++i) {
      rknn_input input{};
      input.index = input_attr_[i].index;
      input.buf = input_tensor_[i].data();
      input.size = input_tensor_[i].byte_size();
      input.pass_through = true;
      inputs.push_back(input);
    }

    std::vector<rknn_output> outputs;
    outputs.reserve(output_tensor_.size());
    for (int i = 0; i < output_tensor_.size(); ++i) {
      rknn_output output{};
      output.want_float = true;
      output.is_prealloc = true;
      output.index = output_attr_[i].index;
      output.buf = output_tensor_[i].data();
      output.size = output_tensor_[i].byte_size();
      outputs.push_back(output);
    }

    OUTCOME_TRY(_m(rknn_inputs_set(context_, inputs.size(), inputs.data())));
    OUTCOME_TRY(_m(rknn_run(context_, nullptr)));
    OUTCOME_TRY(_m(rknn_outputs_get(context_, outputs.size(), outputs.data(), nullptr)));
    return success();
  }

  Result<void> ForwardAsync(Event* event) override { return Status(eNotSupported); };

 private:
  std::vector<rknn_tensor_attr> input_attr_;
  std::vector<rknn_tensor_attr> output_attr_;
  std::vector<Tensor> input_tensor_;
  std::vector<Tensor> output_tensor_;
  rknn_context context_{};
  Device device_;
  Stream stream_;
};

class RknnNetCreator : public Creator<Net> {
 public:
  const char* GetName() const override { return "rknn"; }
  std::unique_ptr<Net> Create(const Value& args) override {
    auto p = std::make_unique<RknnNet>();
    if (p->Init(args)) {
      return p;
    }
    return nullptr;
  }
};

REGISTER_MODULE(Net, RknnNetCreator);

}  // namespace mmdeploy
