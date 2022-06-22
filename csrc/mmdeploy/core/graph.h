// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_
#define MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_

#include "mmdeploy/core/model.h"
#include "mmdeploy/core/module.h"
#include "mmdeploy/core/mpl/span.h"
#include "mmdeploy/core/registry.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/execution/schedulers/registry.h"

namespace mmdeploy {

namespace graph {

using std::pair;
using std::string;
using std::unique_ptr;
using std::vector;

template <class... Ts>
using Sender = TypeErasedSender<Ts...>;

class MMDEPLOY_API Node {
 public:
  virtual ~Node() = default;
  virtual Sender<Value> Process(Sender<Value> input) = 0;

  struct process_t {
    Sender<Value> operator()(Sender<Value> sender, Node* node) const {
      return node->Process(std::move(sender));
    }
  };
  __closure::_BinderBack<process_t, Node*> Process() { return {{}, {}, {this}}; }
};

class Builder {
 public:
  virtual ~Builder() = default;

  virtual Result<void> SetInputs();
  virtual Result<void> SetOutputs();
  virtual Result<unique_ptr<Node>> Build() = 0;

  const vector<string>& inputs() const noexcept { return inputs_; }
  const vector<string>& outputs() const noexcept { return outputs_; }
  const string& name() const noexcept { return name_; }

 protected:
  explicit Builder(Value config);

 protected:
  Value config_;
  vector<string> inputs_;
  vector<string> outputs_;
  string name_;
};

}  // namespace graph

MMDEPLOY_DECLARE_REGISTRY(graph::Node);
MMDEPLOY_DECLARE_REGISTRY(graph::Builder);

}  // namespace mmdeploy

#endif  // MMDEPLOY_SRC_EXPERIMENTAL_PIPELINE_IR_H_
