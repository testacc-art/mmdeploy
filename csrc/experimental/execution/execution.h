// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_
#define MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_

#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>
#include <type_traits>
#include <variant>

#include "core/mpl/detected.h"
#include "core/utils/formatter.h"
#include "intrusive_queue.h"

namespace mmdeploy {

template <class T, class E, class U = void>
using _decays_to = std::enable_if_t<std::is_same<std::decay_t<T>, E>::value, U>;

template <class... Ts>
using __decayed_tuple = std::tuple<std::decay_t<Ts>...>;

template <class Fun, class... As>
using __call_result_t = decltype(std::declval<Fun>()(std::declval<As>()...));

template <class F>
struct __conv {
  F f_;
  using type = __call_result_t<F>;
  operator type() && { return ((F &&) f_)(); }
};

template <class F>
__conv(F) -> __conv<F>;

template <class T, class = std::enable_if_t<std::is_destructible_v<T>>>
struct __conv_proxy {
  T v_;
  template <class F>
  explicit __conv_proxy(F&& f) : v_(((F &&) f)()) {}
  T& operator*() noexcept { return v_; }
};

template <class _Member, class _Self>
_Member _Self::*__memptr(const _Self&);

template <typename _Self, typename _Member>
using __member_t = decltype((std::declval<_Self>().*__memptr<_Member>(std::declval<_Self>())));

template <class From, class To>
using _copy_cvref_t = __member_t<From, To>;

template <class S, class R>
using connect_result_t = decltype(Connect(std::declval<S>(), std::declval<R>()));

template <class...>
struct _types
#if defined(__GNUC__) && !defined(__clang__)
{
}
#endif
;

template <class Sender>
using _get_completion_scheduler_t = decltype(GetCompletionScheduler(std::declval<Sender>()));

template <class Sender>
inline constexpr auto _has_completion_scheduler =
    detail::is_detected_v<_get_completion_scheduler_t, Sender>;

template <class T>
using _empty_if_void_t = std::conditional_t<std::is_same_v<std::tuple<void>, T>, std::tuple<>, T>;

template <class Sender, class SFINAE = void>
struct _completion_signature_for {};

template <class Sender>
struct _completion_signature_for<Sender, std::void_t<typename Sender::value_type>> {
  using type = typename Sender::value_type;
};

template <class Sender>
using completion_signature_for_t = typename _completion_signature_for<Sender>::type;

template <class Sender>
inline constexpr bool _is_sender = detail::is_detected_v<completion_signature_for_t, Sender>;

namespace __closure {

template <class D>
struct SenderAdaptorClosure;

}  // namespace __closure

using __closure::SenderAdaptorClosure;

namespace __closure {

template <class T0, class T1>
struct _Compose : SenderAdaptorClosure<_Compose<T0, T1>> {
  T0 t0_;
  T1 t1_;

  template <class Sender, class = std::enable_if_t<_is_sender<std::decay_t<Sender>>>>
  std::invoke_result_t<T1, std::invoke_result_t<T0, Sender>> operator()(Sender&& sender) && {
    return ((T1 &&) t1_)(((T0 &&) t0_)((Sender &&) sender));
  }

  template <class Sender, class = std::enable_if_t<_is_sender<std::decay_t<Sender>>>>
  std::invoke_result_t<T1, std::invoke_result_t<T0, Sender>> operator()(Sender&& sender) const& {
    return t1_(t0_((Sender &&) sender));
  }
};

template <class D>
struct SenderAdaptorClosure {};

template <class T0, class T1,
          class = std::enable_if_t<
              std::is_base_of_v<SenderAdaptorClosure<std::decay_t<T0>>, std::decay_t<T0>> &&
              std::is_base_of_v<SenderAdaptorClosure<std::decay_t<T1>>, std::decay_t<T1>>>>
_Compose<std::decay_t<T0>, std::decay_t<T1>> operator|(T0&& t0, T1&& t1) {
  return {(T0 &&) t0, (T1 &&) t1};
}

template <class Sender, class Closure,
          class = std::enable_if_t<_is_sender<std::decay_t<Sender>> &&
                                   std::is_base_of_v<SenderAdaptorClosure<std::decay_t<Closure>>,
                                                     std::decay_t<Closure>>>>
std::invoke_result_t<Closure, Sender> operator|(Sender&& sender, Closure&& closure) {
  return ((Closure &&) closure)((Sender &&) sender);
}

template <class Fun, class... As>
struct _BinderBack : SenderAdaptorClosure<_BinderBack<Fun, As...>> {
  Fun fun_;
  std::tuple<As...> as_;

  template <class Sender, class = std::enable_if_t<_is_sender<std::decay_t<Sender>>>>
  std::invoke_result_t<Fun, Sender, As...> operator()(Sender&& sender) && {
    return std::apply(
        [&sender, this](As&... as) { return ((Fun &&) fun_)((Sender &&) sender, (As &&) as...); },
        as_);
  }

  template <class Sender, class = std::enable_if_t<_is_sender<std::decay_t<Sender>>>>
  std::invoke_result_t<Fun, Sender, As...> operator()(Sender&& sender) const& {
    return std::apply([&sender, this](const As&... as) { return fun_((Sender &&) sender, as...); },
                      as_);
  }
};

}  // namespace __closure

using __closure::_BinderBack;

namespace __schedule {

struct schedule_t {
  template <class Scheduler>
  auto operator()(Scheduler&& scheduler) const
      -> decltype(mmdeploySchedule((Scheduler &&) scheduler)) {
    return mmdeploySchedule((Scheduler &&) scheduler);
  }
};

}  // namespace __schedule

using __schedule::schedule_t;
inline constexpr schedule_t Schedule{};

template <class Sched>
using schedule_result_t = decltype(Schedule(std::declval<Sched>()));


namespace __just {

template <class... Ts>
struct _Sender {
  using value_type = std::tuple<Ts...>;
  value_type vals_;

  template <class Receiver>
  struct _Operation {
    value_type vals_;
    Receiver rcvr_;
    friend void Start(_Operation& op_state) noexcept {
      std::apply([&](Ts&... ts) noexcept
                 -> void { SetValue((Receiver &&) op_state.rcvr_, (Ts &&) ts...); },
                 op_state.vals_);
    }
  };

  template <class Receiver>
  friend _Operation<std::decay_t<Receiver>> Connect(const _Sender& sndr, Receiver&& rcvr) {
    return {sndr.vals_, (Receiver &&) rcvr};
  }

  template <class Receiver>
  friend _Operation<std::decay_t<Receiver>> Connect(_Sender&& sndr, Receiver&& rcvr) {
    return {((_Sender &&) sndr).vals_, (Receiver &&) rcvr};
  }
};

struct just_t {
  template <class... Ts>
  _Sender<std::decay_t<Ts>...> operator()(Ts&&... ts) const {
    return {{(Ts &&) ts...}};
  }
};

}  // namespace __just

using __just::just_t;
inline constexpr just_t Just{};

namespace __on {

template <class Scheduler, class Sender, class Receiver>
struct _Operation;

template <class Scheduler, class Sender, class Receiver>
struct _ReceiverRef {
  _Operation<Scheduler, Sender, Receiver>* op_state_;
  template <class... Args>
  friend void SetValue(_ReceiverRef&& self, Args&&... args) {
    SetValue((Receiver &&) self.op_state_->rcvr_, ((Args &&) args)...);
  }
};

template <class Scheduler, class Sender, class Receiver>
struct _Receiver {
  _Operation<Scheduler, Sender, Receiver>* op_state_;
  using ReceiverRef = _ReceiverRef<Scheduler, Sender, Receiver>;
  friend void SetValue(_Receiver&& self) {
    auto op_state = self.op_state_;
    Start(op_state->data_.template emplace<1>(
        Connect((Sender &&) op_state->sndr_, ReceiverRef{op_state})));
  }
};

template <class Scheduler, class Sender, class Receiver>
struct _Operation {
  using __Receiver = _Receiver<Scheduler, Sender, Receiver>;
  using __ReceiverRef = _ReceiverRef<Scheduler, Sender, Receiver>;

  template <class Sender2, class Receiver2>
  _Operation(Scheduler sched, Sender2&& sndr, Receiver2&& rcvr)
      : data_(std::in_place_index<0>, Connect(Schedule(sched), __Receiver{this})),
        scheduler_(sched),
        sndr_((Sender2 &&) sndr),
        rcvr_((Receiver2 &&) rcvr) {}

  friend void Start(_Operation& self) { Start(std::get<0>(self.data_)); }

  std::variant<connect_result_t<schedule_result_t<Scheduler>, __Receiver>,
               connect_result_t<Sender, __ReceiverRef>>
      data_;
  Scheduler scheduler_;
  Sender sndr_;
  Receiver rcvr_;
};

template <class Scheduler, class Sender>
struct _Sender {
  using value_type = completion_signature_for_t<Sender>;
  Scheduler sched_;
  Sender sndr_;

  template <class Receiver>
  using _ReceiverRef = _ReceiverRef<Scheduler, Sender, Receiver>;
  template <class Receiver>
  using _Receiver = _Receiver<Scheduler, Sender, Receiver>;
  template <class Receiver>
  using _Operation = _Operation<Scheduler, Sender, Receiver>;

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr) -> _Operation<std::decay_t<Receiver>> {
    return {((Self &&) self).sched_, ((Self &&) self).sndr_, (Receiver &&) rcvr};
  }
};

struct on_t {
  template <class Scheduler, class Sender>
  _Sender<std::decay_t<Scheduler>, std::decay_t<Sender>> operator()(Scheduler&& scheduler,
                                                                    Sender&& sender) const {
    return {(Scheduler &&) scheduler, (Sender &&) sender};
  }
};

}  // namespace __on

using __on::on_t;
inline constexpr on_t On{};

namespace __schedule_from {

template <class Scheduler, class CvrefSender, class Receiver>
struct _Operation1;

template <class Scheduler, class CvrefSender, class Receiver>
struct _Receiver1;

template <class Scheduler, class CvrefSender, class Receiver>
struct _Receiver2 {
  _Operation1<Scheduler, CvrefSender, Receiver>* op_state_;

  friend void SetValue(_Receiver2&& self) noexcept {
    std::apply(
        [&](auto&&... vals) { SetValue(std::move(self.op_state_->rcvr_), std::move(vals)...); },
        std::move(*self.op_state_->data_));
  }
};

template <class Scheduler, class CvrefSender, class Receiver>
struct _Receiver1 {
  using Receiver2 = _Receiver2<Scheduler, CvrefSender, Receiver>;

  _Operation1<Scheduler, CvrefSender, Receiver>* op_state_;

  template <class... As>
  friend void SetValue(_Receiver1&& self, As&&... as) {
    self.op_state_->data_.emplace((As &&) as...);
    auto sndr = Schedule(self.op_state_->sched_);
    self.op_state_->state2_.emplace(
        __conv{[&] { return Connect(std::move(sndr), Receiver2{self.op_state_}); }});
    Start(*self.op_state_->state2_);
  }
};

template <class Scheduler, class CvrefSender, class Receiver>
struct _Operation1 {
  using Receiver1 = _Receiver1<Scheduler, CvrefSender, Receiver>;
  using Receiver2 = _Receiver2<Scheduler, CvrefSender, Receiver>;

  Scheduler sched_;
  Receiver rcvr_;
  std::optional<completion_signature_for_t<std::decay_t<CvrefSender>>> data_;
  connect_result_t<CvrefSender, Receiver1> state1_;
  std::optional<connect_result_t<schedule_result_t<Scheduler>, Receiver2>> state2_;

  template <class R>
  _Operation1(Scheduler sched, CvrefSender&& sndr, R&& rcvr)
      : sched_(sched),
        rcvr_((R &&) rcvr),
        state1_(Connect((CvrefSender &&) sndr, Receiver1{this})) {}

  _Operation1(const _Operation1&) = delete;
  _Operation1(_Operation1&&) noexcept = delete;
  _Operation1& operator=(const _Operation1&) = delete;
  _Operation1& operator=(_Operation1&&) noexcept = delete;

  friend void Start(_Operation1& op_state) noexcept { Start(op_state.state1_); }
};

template <class Scheduler, class Sender>
struct _Sender {
  using value_type = completion_signature_for_t<Sender>;

  Scheduler sched_;
  Sender sndr_;

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr)
      -> _Operation1<Scheduler, _copy_cvref_t<Self, Sender>, std::decay_t<Receiver>> {
    return {self.sched_, ((Self &&) self).sndr_, (Receiver &&) rcvr};
  }

  friend Scheduler GetCompletionScheduler(const _Sender& self) noexcept { return self.sched_; }
};

struct schedule_from_t {
  template <class Scheduler, class Sender>
  _Sender<std::decay_t<Scheduler>, std::decay_t<Sender>> operator()(Scheduler&& scheduler,
                                                                    Sender&& sender) const {
    return {(Scheduler &&) scheduler, (Sender &&) sender};
  }
};

}  // namespace __schedule_from

using __schedule_from::schedule_from_t;
inline constexpr schedule_from_t ScheduleFrom{};

namespace __transfer {

struct transfer_t {
  template <class Sender, class Scheduler>
  auto operator()(Sender&& sender, Scheduler&& scheduler) const {
    return ScheduleFrom((Scheduler &&) scheduler, (Sender &&) sender);
  }

  template <class Scheduler>
  _BinderBack<transfer_t, std::decay_t<Scheduler>> operator()(Scheduler&& scheduler) const {
    return {{}, {}, {(Scheduler &&) scheduler}};
  }
};

}  // namespace __transfer

using __transfer::transfer_t;
inline constexpr transfer_t Transfer{};

namespace __then {

template <class R, class F>
struct _Receiver {
  R r_;
  F f_;

  template <class... Args>
  friend void SetValue(_Receiver&& self, Args&&... args) {
    SetValue(std::move(self.r_), std::invoke((F &&) self.f_, (Args &&) args...));
  }
};

template <class S, class F>
struct _Sender {
  using value_type = _empty_if_void_t<std::tuple<decltype(std::apply(
      std::declval<F>(), std::declval<completion_signature_for_t<S>>()))>>;

  S s_;
  F f_;

  template <class Receiver>
  using receiver_t = _Receiver<std::decay_t<Receiver>, F>;

  template <class Self, class R, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, R r) {
    return Connect(((Self &&) self).s_, receiver_t<R>{(R &&) r, (F &&) self.f_});
  }

  template <class Sender = S>
  friend auto GetCompletionScheduler(const _Sender& self) noexcept
      -> decltype(GetCompletionScheduler(std::declval<Sender>())) {
    return GetCompletionScheduler(self.s_);
  }
};

struct then_t {
  template <class Sender, class Fun>
  _Sender<std::decay_t<Sender>, Fun> operator()(Sender&& sender, Fun fun) const {
    return {(Sender &&) sender, std::move(fun)};
  }
  template <class Fun>
  _BinderBack<then_t, Fun> operator()(Fun fun) const {
    return {{}, {}, {std::move(fun)}};
  }
};

}  // namespace __then

using __then::then_t;
inline constexpr then_t Then;

namespace __let_value {

template <class T>
using __decay_ref = std::decay_t<T>&;

template <class Fun, class... As>
using __result_sender_t = __call_result_t<Fun, __decay_ref<As>...>;

template <class Fun, class Tup>
struct __value_type {};

template <class Fun, class... As>
struct __value_type<Fun, std::tuple<As...>> {
  using type = __result_sender_t<Fun, As...>;
};

template <class Fun, class Tup>
using __value_type_t = typename __value_type<Fun, Tup>::type;

template <class CvrefSender, class Receiver, class Fun>
struct _Storage {
  using Sender = std::decay_t<CvrefSender>;
  using operation_t =
      connect_result_t<__value_type_t<Fun, completion_signature_for_t<Sender>>, Receiver>;
  std::optional<completion_signature_for_t<Sender>> args_;
  // workaround for MSVC v142 toolset, copy elision does not work here
  std::optional<__conv_proxy<operation_t>> proxy_;
};

template <class CvrefSender, class Receiver, class Fun>
struct _Operation;

template <class CvrefSender, class Receiver, class Fun>
struct _Receiver {
  _Operation<CvrefSender, Receiver, Fun>* op_state_;

  template <class... As>
  friend void SetValue(_Receiver&& self, As&&... as) noexcept {
    using operation_t = typename _Storage<CvrefSender, Receiver, Fun>::operation_t;
    auto* op_state = self.op_state_;
    auto& args = op_state->storage_.args_.emplace((As &&) as...);
    op_state->storage_.proxy_.emplace([&] {
      return Connect(std::apply(std::move(op_state->fun_), args), std::move(op_state->rcvr_));
    });
    Start(**op_state->storage_.proxy_);
  }
};

template <class CvrefSender, class Receiver, class Fun>
struct _Operation {
  using receiver_t = _Receiver<CvrefSender, Receiver, Fun>;

  friend void Start(_Operation& self) noexcept { Start(self.op_state2_); }

  template <class Receiver2>
  _Operation(CvrefSender&& sndr, Receiver2&& rcvr, Fun fun)
      : op_state2_(Connect((CvrefSender &&) sndr, receiver_t{this})),
        rcvr_((Receiver2 &&) rcvr),
        fun_((Fun &&) fun) {}

  connect_result_t<CvrefSender, receiver_t> op_state2_;
  Receiver rcvr_;
  Fun fun_;
  _Storage<CvrefSender, Receiver, Fun> storage_;
};

template <class Sender, class Fun>
struct _Sender {
  template <class Self, class Receiver>
  using operation_t = _Operation<_copy_cvref_t<Self, Sender>, std::decay_t<Receiver>, Fun>;
  template <class Self, class Receiver>
  using receiver_t = _Receiver<_copy_cvref_t<Self, Sender>, std::decay_t<Receiver>, Fun>;

  using value_type =
      completion_signature_for_t<__value_type_t<Fun, completion_signature_for_t<Sender>>>;

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr) -> operation_t<Self, Receiver> {
    return operation_t<Self, Receiver>{((Self &&) self).sndr_, (Receiver &&) rcvr,
                                       ((Self &&) self).fun_};
  }
  Sender sndr_;
  Fun fun_;
};

struct let_value_t {
  template <class Sender, class Fun>
  _Sender<std::decay_t<Sender>, Fun> operator()(Sender&& sender, Fun fun) const {
    return {(Sender &&) sender, std::move(fun)};
  }
  template <class Fun>
  _BinderBack<let_value_t, Fun> operator()(Fun fun) const {
    return {{}, {}, {std::move(fun)}};
  }
};

}  // namespace __let_value

using __let_value::let_value_t;
inline constexpr let_value_t LetValue{};

namespace __bulk {

template <class CvrefSender, class Shape, class Fun, class Receiver>
struct _Operation;

template <class Receiver, class Shape, class Fun>
struct _Receiver {
  Receiver rcvr_;
  Shape shape_;
  Fun fun_;

  template <class... As>
  friend void SetValue(_Receiver&& self, As&&... as) {
    for (Shape i = 0; i < self.shape_; ++i) {
      self.fun_(i, as...);
    }
    SetValue((Receiver &&) self.rcvr_, (As &&) as...);
  }
};

template <class CvrefSender, class Shape, class Fun, class Receiver>
struct _Operation {
  connect_result_t<CvrefSender, _Receiver<Receiver, Shape, Fun>> op_state2_;
  friend void Start(_Operation& op_state) { Start(op_state.op_state2_); }
};

template <class Sender, class Shape, class Fun>
struct _Sender {
  using value_type = completion_signature_for_t<Sender>;

  Sender sndr_;
  Shape shape_;
  Fun fun_;

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr)
      -> _Operation<_copy_cvref_t<Self, Sender>, Shape, Fun, Receiver> {
    return {Connect(((Self &&) self).sndr_,
                    _Receiver<Receiver, Shape, Fun>{(Receiver &&) rcvr, ((Self &&) self).shape_,
                                                    ((Self &&) self).fun_})};
  }
};

struct bulk_t {
  template <class Sender, class Shape, class Fun,
            class = std::enable_if_t<_is_sender<std::decay_t<Sender>>>>
  __bulk::_Sender<std::decay_t<Sender>, Shape, Fun> operator()(Sender&& sender, Shape shape,
                                                               Fun fun) const {
    return {(Sender &&) sender, shape, std::move(fun)};
  }
  template <class Shape, class Fun>
  _BinderBack<bulk_t, Shape, Fun> operator()(Shape shape, Fun fun) const {
    return {{}, {}, {shape, std::move(fun)}};
  }
};

}  // namespace __bulk

using __bulk::bulk_t;
inline constexpr bulk_t Bulk{};

namespace __split {

struct _OperationBase {
  _OperationBase* next_;
  void (*notify_)(_OperationBase*) noexcept;
};

template <class SharedState>
struct _Receiver {
  SharedState& shared_state_;

  template <class... As>
  friend void SetValue(_Receiver&& recvr, As&&... as) {
    auto& state = recvr.shared_state_;
    state.data_.emplace((As &&) as...);
    state._Notify();
  }
};

template <class Sender>
struct _SharedState {
  std::optional<completion_signature_for_t<Sender>> data_;

  using Receiver = _Receiver<_SharedState>;

  connect_result_t<Sender, Receiver> op_state2_;

  std::atomic<void*> head_{nullptr};

  explicit _SharedState(Sender& sndr) : op_state2_(Connect((Sender &&) sndr, Receiver{*this})) {}

  void _Notify() noexcept {
    void* const completion_state = static_cast<void*>(this);
    void* old = head_.exchange(completion_state, std::memory_order_acq_rel);
    auto* op_state = static_cast<_OperationBase*>(old);

    while (op_state != nullptr) {
      _OperationBase* next = op_state->next_;
      op_state->notify_(op_state);
      op_state = next;
    }
  }
};

template <class Sender, class Receiver>
struct _Operation : _OperationBase {
  Receiver recvr_;
  std::shared_ptr<_SharedState<Sender>> shared_state_;

  _Operation(Receiver&& r, std::shared_ptr<_SharedState<Sender>> shared_state)
      : _OperationBase{nullptr, _Notify},
        recvr_(std::move(r)),
        shared_state_(std::move(shared_state)) {}

  static void _Notify(_OperationBase* self) noexcept {
    auto op = static_cast<_Operation*>(self);
    std::apply([&](const auto&... args) { SetValue((Receiver &&) op->recvr_, args...); },
               op->shared_state_->data_.value());
  }

  friend void Start(_Operation& self) {
    auto shared_state = self.shared_state_.get();
    std::atomic<void*>& head = shared_state->head_;
    void* const completion_state = static_cast<void*>(shared_state);
    void* old = head.load(std::memory_order_acquire);

    do {
      if (old == completion_state) {
        self._Notify(&self);
        return;
      }
      self.next_ = static_cast<_OperationBase*>(old);
    } while (!head.compare_exchange_weak(old, static_cast<void*>(&self), std::memory_order_release,
                                         std::memory_order_acquire));

    if (old == nullptr) {
      Start(shared_state->op_state2_);
    }
  }
};

template <class Sender>
struct _Sender {
  using SharedState = _SharedState<Sender>;
  template <class Receiver>
  using Operation = _Operation<Sender, std::decay_t<Receiver>>;

  using value_type = completion_signature_for_t<Sender>;

  Sender sndr_;
  std::shared_ptr<SharedState> shared_state_;

  explicit _Sender(Sender sndr)
      : sndr_((Sender &&) sndr), shared_state_{std::make_shared<SharedState>(sndr_)} {}

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& recvr) -> Operation<Receiver> {
    return Operation<Receiver>((std::decay_t<Receiver> &&) recvr, self.shared_state_);
  }
};

struct split_t {
  template <class Sender>
  _Sender<std::decay_t<Sender>> operator()(Sender&& sender) const {
    return _Sender<std::decay_t<Sender>>{(Sender &&) sender};
  }
  _BinderBack<split_t> operator()() const { return {{}, {}, {}}; }
};

}  // namespace __split

using __split::split_t;
inline constexpr split_t Split{};

namespace __when_all {

template <class... Senders>
using __concat_t = decltype(std::tuple_cat(std::declval<completion_signature_for_t<Senders>>()...));

template <class... Senders>
struct _Sender {
  //  template <class... _Sndrs>
  //  explicit _Sender(_Sndrs&&... sndrs) : sndrs_((_Sndrs &&) sndrs...) {}

  template <class CvrefReceiver>
  struct _Operation;

  using value_type = __concat_t<Senders...>;

  template <class CvrefReceiver, size_t Index>
  struct _Receiver {
    using Receiver = std::decay_t<CvrefReceiver>;
    _Operation<CvrefReceiver>* op_state_;

    template <class... As>
    friend void SetValue(_Receiver&& self, As&&... as) noexcept {
      std::get<Index>(self.op_state_->vals_).emplace((As &&) as...);
      self.op_state_->_Arrive();
    }
  };

  template <class CvrefReceiver>
  struct _Operation {
    using _WhenAll = _copy_cvref_t<CvrefReceiver, _Sender>;
    using Receiver = std::decay_t<CvrefReceiver>;
    template <class Sender, size_t Index>
    using _ChildOpState =
        connect_result_t<_copy_cvref_t<_WhenAll, Sender>, _Receiver<CvrefReceiver, Index>>;

    using _Indices = std::index_sequence_for<Senders...>;

    // workaround for a bug in GCC7 that Is in a lambda is treated as unexpanded parameter pack
    template <class S, class R>
    static auto _Connect1(S&& s, R&& r) {
      return __conv{[&]() mutable { return Connect((S &&) s, (R &&) r); }};
    }

    template <size_t... Is>
    static auto _ConnectChildren(_Operation* self, _WhenAll&& when_all, std::index_sequence<Is...>)
        -> std::tuple<_ChildOpState<Senders, Is>...> {
      return {_Connect1(std::get<Is>(((_WhenAll &&) when_all).sndrs_),
                        _Receiver<CvrefReceiver, Is>{self})...};
    }

    using _ChildOpStatesTuple =
        decltype(_ConnectChildren(nullptr, std::declval<_WhenAll>(), _Indices{}));

    using _ChildValueTuple = std::tuple<std::optional<completion_signature_for_t<Senders>>...>;

    void _Arrive() noexcept {
      if (0 == --count_) {
        _Complete();
      }
    }

    void _Complete() noexcept {
      std::apply(
          [this](auto&... opt_vals) -> void {
            std::apply(
                [this](auto&... all_vals) -> void {
                  SetValue((Receiver &&) recvr_, std::move(all_vals)...);
                },
                std::tuple_cat(
                    std::apply([](auto&... vals) { return std::tie(vals...); }, *opt_vals)...));
          },
          vals_);
    }

    _Operation(_WhenAll&& when_all, Receiver rcvr)
        : child_states_{_ConnectChildren(this, (_WhenAll &&) when_all, _Indices{})},
          recvr_((Receiver &&) rcvr) {}

    friend void Start(_Operation& self) noexcept {
      std::apply([](auto&&... child_ops) noexcept -> void { (Start(child_ops), ...); },
                 self.child_states_);
    }

    _Operation(const _Operation&) = delete;
    _Operation(_Operation&&) = delete;
    _Operation& operator=(const _Operation&) = delete;
    _Operation& operator=(_Operation&&) = delete;

    _ChildOpStatesTuple child_states_;
    Receiver recvr_;
    std::atomic<size_t> count_{sizeof...(Senders)};
    _ChildValueTuple vals_;
  };

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr)
      -> _Operation<_copy_cvref_t<Self, std::decay_t<Receiver>>> {
    return {(Self &&) self, (Receiver &&) rcvr};
  }

  std::tuple<Senders...> sndrs_;
};

struct when_all_t {
  template <class... Senders, class = std::enable_if_t<(_is_sender<std::decay_t<Senders>> && ...)>>
  _Sender<std::decay_t<Senders>...> operator()(Senders&&... senders) const {
    return {{(Senders &&) senders...}};
  }
};

}  // namespace __when_all

using __when_all::when_all_t;
inline constexpr when_all_t WhenAll{};

namespace __ensure_started {

struct _OperationBase {
  void (*notify_)(_OperationBase*);
};

template <class SharedState>
struct _Receiver {
  std::shared_ptr<SharedState> shared_state_;

  template <class... As>
  friend void SetValue(_Receiver&& self, As&&... as) {
    assert(self.shared_state_);
    self.shared_state_->data_.emplace((As &&) as...);
    self.shared_state_->_Notify();
    self.shared_state_.reset();
  }
};

template <class Sender>
struct _SharedState {
  std::optional<completion_signature_for_t<Sender>> data_;
  std::optional<connect_result_t<Sender, _Receiver<_SharedState>>> op_state2_;
  std::atomic<void*> awaiting_{nullptr};

  void _Notify() noexcept {
    void* const completion_state = static_cast<void*>(this);
    void* old = awaiting_.exchange(completion_state, std::memory_order_acq_rel);
    auto* op_state = static_cast<_OperationBase*>(old);

    if (op_state != nullptr) {
      op_state->notify_(op_state);
    }
  }
};

template <class Sender, class Receiver>
struct _Operation : public _OperationBase {
  Receiver rcvr_;
  std::shared_ptr<_SharedState<Sender>> shared_state_;

  _Operation(Receiver&& rcvr, std::shared_ptr<_SharedState<Sender>> shared_state)
      : _OperationBase{_Notify}, rcvr_(std::move(rcvr)), shared_state_(std::move(shared_state)) {}

  static void _Notify(_OperationBase* self) noexcept {
    auto op_state = static_cast<_Operation*>(self);

    std::apply(
        [&](auto&&... vals) -> void {
          SetValue((Receiver &&) op_state->rcvr_, (decltype(vals)&&)vals...);
        },
        *op_state->shared_state_->data_);
  }

  friend void Start(_Operation& self) {
    auto shared_state = self.shared_state_.get();
    std::atomic<void*>& awaiting = shared_state->awaiting_;
    void* const completion_state = static_cast<void*>(shared_state);
    void* old = awaiting.load(std::memory_order_acquire);

    do {
      if (old == completion_state) {
        _Notify(&self);
        return;
      }
    } while (awaiting.compare_exchange_weak(old, static_cast<void*>(&self),
                                            std::memory_order_release, std::memory_order_acquire));
  }
};

template <class Sender>
struct _Sender {
  using SharedState = _SharedState<Sender>;
  using value_type = completion_signature_for_t<Sender>;

  std::shared_ptr<SharedState> shared_state_;

  explicit _Sender(Sender sndr) : shared_state_(std::make_shared<SharedState>()) {
    Start(shared_state_->op_state2_.emplace(
        __conv{[&] { return Connect((Sender &&) sndr, _Receiver<SharedState>{shared_state_}); }}));
  }

  template <class Self, class Receiver, _decays_to<Self, _Sender, bool> = true>
  friend auto Connect(Self&& self, Receiver&& rcvr) -> _Operation<Sender, std::decay_t<Receiver>> {
    return {(Receiver &&) rcvr, std::move(self.shared_state_)};
  }
};

struct ensure_started_t {
  template <class Sender, class = std::enable_if_t<_is_sender<std::decay_t<Sender>>>>
  _Sender<std::decay_t<Sender>> operator()(Sender&& sender) const {
    return _Sender{(Sender &&) sender};
  }
};

}  // namespace __ensure_started

using __ensure_started::ensure_started_t;
inline constexpr ensure_started_t EnsureStarted{};

namespace __submit {

namespace __impl {

template <class Sender, class Receiver>
struct _Operation {
  struct _Receiver {
    _Operation* op_state_;
    template <class... As>
    friend void SetValue(_Receiver&& self, As&&... as) noexcept {
      std::unique_ptr<_Operation> _g{self.op_state_};
      return SetValue((Receiver &&) self.op_state_->rcvr_, (As &&) as...);
    }
  };
  Receiver rcvr_;
  connect_result_t<Sender, _Receiver> op_state_;
  template <class R, _decays_to<R, Receiver, bool> = true>
  _Operation(Sender&& sndr, R&& rcvr)
      : rcvr_((R &&) rcvr), op_state_(Connect((Sender &&) sndr, _Receiver{this})) {}
};

}  // namespace __impl

}  // namespace __submit

template <class Sender, class Receiver>
void __Submit(Sender&& sndr, Receiver&& rcvr) noexcept(false) {
  using _Operation = __submit::__impl::_Operation<Sender, std::decay_t<Receiver>>;
  Start((new _Operation((Sender &&) sndr, (Receiver &&) rcvr))->op_state_);
}

namespace __start_detached {

struct _Receiver {
  template <class... As>
  friend void SetValue(_Receiver&&, As&&...) noexcept {}
};

struct start_detached_t {
  template <class Sender>
  void operator()(Sender&& sender) const {
    __Submit((Sender &&) sender, __start_detached::_Receiver{});
  }
};

}  // namespace __start_detached

using __start_detached::start_detached_t;
inline constexpr start_detached_t StartDetached{};

namespace __loop {
class RunLoop;

namespace __impl {

struct _Task {
  virtual void _Execute() noexcept = 0;
  _Task* next_ = nullptr;
};

template <class Receiver>
class _Operation final : _Task {
  friend void Start(_Operation& op_state) noexcept { op_state._Start(); }

  void _Execute() noexcept override { SetValue((Receiver &&) rcvr_); }
  void _Start() noexcept;

  Receiver rcvr_;
  RunLoop* const loop_;

 public:
  template <class _Receiver2>
  explicit _Operation(_Receiver2&& rcvr, RunLoop* loop)
      : rcvr_((_Receiver2 &&) rcvr), loop_(loop) {}
};

}  // namespace __impl

class RunLoop {
  template <class>
  friend class __impl::_Operation;

 public:
  class _Scheduler {
    class _ScheduleTask {
      friend _Scheduler;
      template <class _Receiver>
      friend auto Connect(const _ScheduleTask& self, _Receiver&& rcvr)
          -> __impl::_Operation<std::decay_t<_Receiver>> {
        return {(_Receiver &&) rcvr, self.loop_};
      }
      explicit _ScheduleTask(RunLoop* loop) noexcept : loop_(loop) {}
      RunLoop* const loop_;
    };
    friend RunLoop;
    explicit _Scheduler(RunLoop* loop) noexcept : loop_(loop) {}

   public:
    friend _ScheduleTask mmdeploySchedule(const _Scheduler& self) noexcept { return self._Schedule(); }
    bool operator==(const _Scheduler& other) const noexcept { return loop_ == other.loop_; }

   private:
    _ScheduleTask _Schedule() const noexcept { return _ScheduleTask{loop_}; }
    RunLoop* loop_;
  };
  _Scheduler GetScheduler() { return _Scheduler{this}; }
  void _Run();
  void _Finish();

 private:
  void _push_back(__impl::_Task* task);
  __impl::_Task* _pop_front();

  std::mutex mutex_;
  std::condition_variable cv_;
  __impl::_Task* head_ = nullptr;
  __impl::_Task* tail_ = nullptr;
  bool stop_ = false;
};

namespace __impl {

template <class Receiver>
inline void _Operation<Receiver>::_Start() noexcept {
  loop_->_push_back(this);
}

}  // namespace __impl

inline void RunLoop::_Run() {
  while (auto* task = _pop_front()) {
    task->_Execute();
  }
}

inline void RunLoop::_Finish() {
  std::lock_guard lock{mutex_};
  stop_ = true;
  cv_.notify_all();
}

inline void RunLoop::_push_back(__impl::_Task* task) {
  std::lock_guard lock{mutex_};
  if (head_ == nullptr) {
    head_ = task;
  } else {
    tail_->next_ = task;
  }
  tail_ = task;
  task->next_ = nullptr;
  cv_.notify_one();
}

inline __impl::_Task* RunLoop::_pop_front() {
  std::unique_lock lock{mutex_};
  while (head_ == nullptr) {
    if (stop_) {
      return nullptr;
    }
    cv_.wait(lock);
  }
  auto* task = head_;
  head_ = task->next_;
  if (head_ == nullptr) {
    tail_ = nullptr;
  }
  return task;
}

}  // namespace __loop

using RunLoop = __loop::RunLoop;

namespace __sync_wait {

template <class Sender>
struct _State {
  std::optional<completion_signature_for_t<Sender>> data_;
};

template <class Sender>
struct _Receiver {
  _State<Sender>* state_;
  RunLoop* loop_;

  template <class... As>
  inline friend void SetValue(_Receiver&& rcvr, As&&... as) noexcept {
    rcvr.state_->data_.emplace(((As &&) as)...);
    rcvr.loop_->_Finish();
  }
};

struct sync_wait_t {
  template <class Sender>
  completion_signature_for_t<std::decay_t<Sender>> operator()(Sender&& sender) const {
    _State<std::decay_t<Sender>> state;
    RunLoop loop;
    // connect to internal receiver
    auto op_state = Connect((Sender &&) sender, _Receiver<std::decay_t<Sender>>{&state, &loop});
    Start(op_state);

    loop._Run();
    // extract the returned values
    return std::move(*state.data_);
  }
};

}  // namespace __sync_wait

using __sync_wait::sync_wait_t;
inline constexpr sync_wait_t SyncWait{};

class SingleThreadContext {
  RunLoop loop_;
  std::thread thread_;

 public:
  SingleThreadContext() : loop_(), thread_([this] { loop_._Run(); }) {}

  ~SingleThreadContext() {
    loop_._Finish();
    thread_.join();
  }

  auto GetScheduler() noexcept { return loop_.GetScheduler(); }

  std::thread::id GetThreadId() const noexcept { return thread_.get_id(); }
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_CSRC_EXPERIMENTAL_EXECUTION_H_
