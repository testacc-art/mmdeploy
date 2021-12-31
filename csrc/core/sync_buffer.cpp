// Copyright (c) OpenMMLab. All rights reserved.

#include "core/device.h"
#include "core/logger.h"

#include <list>

namespace mmdeploy {

struct SyncBuffer::Impl {
  explicit Impl(size_t size) : size_(size) {}

  struct Region {
    Buffer buffer;
    int is_valid{false};
    Stream stream;
  };

  Region* FindRegion(Device device) {
    for (auto& r : regions_) {
      if (r.buffer.GetDevice() == device) {
        return &r;
      }
    }
    return nullptr;
  }

  Region* FindFirstValidRegion() {
    for (auto& r : regions_) {
      if (r.is_valid) {
        return &r;
      }
    }
    return nullptr;
  }

  void InvalidateAll() {
    for (auto& r : regions_) {
      r.is_valid = false;
    }
  }

  Region* CreateRegion(Device device) {
    regions_.push_back({Buffer(device, size_), false});
    return &regions_.back();
  }

  size_t size_;
  std::list<Region> regions_;
};

SyncBuffer::SyncBuffer(size_t size, size_t page_size) : impl_(std::make_shared<Impl>(size)) {}

size_t SyncBuffer::GetSize() { return impl_->size_; }

HostAccessor::HostAccessor(SyncBuffer& sync_buffer, size_t offset, size_t size, AccessMode mode) {
  auto& impl = *sync_buffer.impl_;

  constexpr const auto device = Device{0};

  auto region = impl.FindRegion(device);
  if (!region) {
    INFO("malloc on host");
    region = impl.CreateRegion(device);
  }

  if (mode == AccessMode::read_only || mode == AccessMode::read_write) {
    // find an valid region
    if (!region->is_valid) {
      if (auto src = impl.FindFirstValidRegion()) {
        assert(src->stream);
        auto host_ptr = region->buffer.GetNative();
        INFO("copy to host");
        src->stream.Copy(src->buffer, host_ptr).value();
        src->stream.Wait().value();
      } else {
        WARN("reading uninitialized buffer");
      }
    }
  }

  if (mode == AccessMode::write_only || mode == AccessMode::read_write) {
    // invalidate other region
    impl.InvalidateAll();
    region->is_valid = true;
  }

  region_ = region;
  ptr_ = (uint8_t*)region->buffer.GetNative() + offset;
}

HostAccessor::~HostAccessor() {
  auto region = static_cast<SyncBuffer::Impl::Region*>(region_);
  region->stream = Stream();
}

Accessor::Accessor(SyncBuffer& sync_buffer, size_t offset, size_t size, AccessMode mode,
                   Stream& stream) {
  auto& impl = *sync_buffer.impl_;

  auto device = stream.GetDevice();

  auto region = impl.FindRegion(device);
  if (!region) {
    region = impl.CreateRegion(device);
  }

  if (mode == AccessMode::read_only || mode == AccessMode::read_write) {
    // find an valid region
    if (!region->is_valid) {
      if (auto src = impl.FindFirstValidRegion()) {
        if (src->stream != stream) {
          HostAccessor host_accessor(sync_buffer, offset, size, mode);
          auto host_ptr = host_accessor.ptr();
          INFO("copy to device");
          stream.Copy(host_ptr, region->buffer).value();
          region->is_valid = true;
        } else {
          assert(0);
        }
      }
    }
  }

  if (mode == AccessMode::write_only || mode == AccessMode::read_write) {
    // invalidate other region
    impl.InvalidateAll();
    region->is_valid = true;
    region->stream = stream;
  }

  region_ = region;
  mode_ = mode;
  native_ = region->buffer.GetNative();
}

Accessor::~Accessor() = default;

}  // namespace mmdeploy
