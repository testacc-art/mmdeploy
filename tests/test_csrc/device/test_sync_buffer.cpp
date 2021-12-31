// Copyright (c) OpenMMLab. All rights reserved.

// clang-format off
#include "catch.hpp"
// clang-format on

#include <chrono>
#include <iostream>
#include <thread>

#include "core/device.h"

using namespace mmdeploy;

namespace mmdeploy::cuda {

void add(const float *a, const float *b, float *c, int n, void *stream);

}

// https://docs.nvidia.com/cuda/cuda-for-tegra-appnote/index.html

TEST_CASE("test sync-buffer", "[sync-buffer]") {
  static constexpr auto N = 1024;
  using Dtype = float;
  SyncBuffer buf_a(N * sizeof(Dtype));
  SyncBuffer buf_b(N * sizeof(Dtype));
  SyncBuffer buf_c(N * sizeof(Dtype));
  auto device = Device("cuda");
  auto stream = Stream(device);

  {
    HostAccessor a(buf_a, 0, buf_a.GetSize(), AccessMode::write_only);
    for (int i = 0; i < N; ++i) {
      a.ptr<Dtype>()[i] = (Dtype)i;
    }
  }

  {
    HostAccessor b(buf_b, 0, buf_b.GetSize(), AccessMode::write_only);
    for (int i = 0; i < N; ++i) {
      b.ptr<Dtype>()[i] = (Dtype)i * 2;
    }
  }

  {
    Accessor a(buf_a, 0, buf_a.GetSize(), AccessMode::read_only, stream);
    Accessor b(buf_b, 0, buf_b.GetSize(), AccessMode::read_only, stream);
    Accessor c(buf_c, 0, buf_c.GetSize(), AccessMode::write_only, stream);
    mmdeploy::cuda::add((float *)a.native(), (float *)b.native(), (float *)c.native(), N,
                        stream.GetNative());
  }

  {
    HostAccessor c(buf_c, 0, buf_c.GetSize(), AccessMode::read_only);
    for (int i = 0; i < N; ++i) {
      REQUIRE(c.ptr<Dtype>()[i] == i * 3);
    }
  }

  {
    Accessor c(buf_c, 0, buf_c.GetSize(), AccessMode::read_only, stream);
    Accessor b(buf_b, 0, buf_b.GetSize(), AccessMode::read_only, stream);
    Accessor a(buf_a, 0, buf_a.GetSize(), AccessMode::write_only, stream);
    mmdeploy::cuda::add((float *)c.native(), (float *)b.native(), (float *)a.native(), N,
                        stream.GetNative());
  }

  {
    HostAccessor a(buf_a, 0, buf_a.GetSize(), AccessMode::read_only);
    for (int i = 0; i < N; ++i) {
      REQUIRE(a.ptr<Dtype>()[i] == i * 5);
    }
  }
}
