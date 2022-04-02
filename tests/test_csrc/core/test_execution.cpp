#include <chrono>
#include <numeric>

#include "catch.hpp"
#include "core/utils/formatter.h"
#include "experimental/execution/static_thread_pool.h"
#include "experimental/execution/timed_single_thread_context.h"
#include "experimental/execution/type_erased.h"

using namespace mmdeploy;

TEST_CASE("test basic execution", "[execution]") {
  InlineScheduler sch;
  auto a = Just(Value{{"a", 100}, {"b", 200}});
  auto b = ScheduleFrom(sch, a);
  //  auto begin = Schedule(sch);
  //  auto b = Then(begin, [] { return Value{{"a", 100}, {"b", 200}}; });
  auto c = Then(b, [](Value v) -> Value { return {{"c", v["a"].get<int>() + v["b"].get<int>()}}; });
  auto d = SyncWait(c);
  MMDEPLOY_ERROR("{}", d);
}

template <class Sender>
auto GetKey(Sender&& sndr, const std::string& key) {
  return Then((Sender &&) sndr, [key](const Value& v) { return v[key]; });
}

TEST_CASE("test split", "[execution]") {
  auto a = Just(Value{{"x", 100}, {"y", 1000}});
  auto s = Split(a);
  auto x = GetKey(s, "x");
  auto y = GetKey(s, "y");
  auto x_v = SyncWait(x);
  auto y_v = SyncWait(y);
  MMDEPLOY_ERROR("x = {}, y = {}", x_v, y_v);
}

TEST_CASE("test when_all", "[execution]") {
  auto a = Just(100);
  auto b = Just(200);
  auto c = Just(300);
  auto d = Just(400);
  auto e = Just(500);
  auto t = WhenAll(a, b, c, d, e);
  auto v = SyncWait(t);
  MMDEPLOY_ERROR("v = {}", v);
}

void Func() {
  auto a = Just(100, 200);
  auto b =
      LetValue(a, [](int& x, int& y) { return Then(Just(x + y), [](int v) { return v * v; }); });
  auto v = SyncWait(b);
  static_assert(std::is_same_v<decltype(v), std::tuple<int>>);
  MMDEPLOY_ERROR("v = {}", v);
}

TEST_CASE("test let_value", "[execution]") { Func(); }

TEST_CASE("test fork-join", "[execution]") {
  auto a = Just(Value{{"x", 100}, {"y", 1000}});
  auto s = Split(a);
  auto x = GetKey(s, "x");
  auto y = GetKey(s, "y");
  auto xy = WhenAll(x, y);
  auto v = SyncWait(xy);
  static_assert(std::is_same_v<decltype(v), std::tuple<Value, Value>>);
  MMDEPLOY_ERROR("v = {}", v);
}

TEST_CASE("test ensure_started", "[execution]") {
  //  auto s = Schedule(gThreadPool().GetScheduler());
  auto pool = __static_thread_pool::StaticThreadPool{};
  auto s = Schedule(pool.GetScheduler());
  auto a = Then(s, []() -> Value {
    MMDEPLOY_INFO("ensure_started sleep start...");
    std::this_thread::sleep_for(std::chrono::seconds(1));
    MMDEPLOY_INFO("ensure_started sleep end");
    return 23333;
  });
  MMDEPLOY_INFO("ensure_started call");
  auto c = EnsureStarted(a);
  MMDEPLOY_INFO("ensure_started ret");
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  MMDEPLOY_INFO("ensure_started sync_wait");
  auto v = SyncWait(c);
  MMDEPLOY_ERROR("ensure_started: {}", v);
}

TEST_CASE("test start_detached", "[execution]") {
  auto pool = __static_thread_pool::StaticThreadPool{4};
  auto s = Schedule(pool.GetScheduler());
  auto a = Then(s, [] {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    return Value(100);
  });
  auto b = Then(a, [](...) {
    MMDEPLOY_INFO("OK");
    return Value(200);
  });
  StartDetached(b);
  MMDEPLOY_INFO("StartDetached ret");
}

TEST_CASE("test on", "[execution]") {
  auto pool = __static_thread_pool::StaticThreadPool{4};
  auto a = Just(100, 200);
  auto b = On(pool.GetScheduler(), a);
  auto c = SyncWait(b);
  static_assert(std::is_same_v<decltype(c), std::tuple<int, int>>);
  MMDEPLOY_ERROR("c = {}", c);
}

mmdeploy_value_t f(mmdeploy_value_t v, void*) {
  auto& arr = ((Value*)v)->array();
  return (mmdeploy_value_t)(new Value{arr[0].get<int>() + arr[1].get<int>()});
}

TEST_CASE("test executor C API", "[execution]") {
  auto sched = mmdeploy_inline_scheduler();
  REQUIRE(sched);
  Value ctx{100, 200};
  auto ctx_sndr = mmdeploy_executor_just(reinterpret_cast<mmdeploy_value_t>(&ctx));
  REQUIRE(ctx_sndr);
  auto a = mmdeploy_executor_transfer(ctx_sndr, sched);
  REQUIRE(a);
  auto b = mmdeploy_executor_then(a, f, nullptr);
  REQUIRE(b);
  auto c = mmdeploy_executor_sync_wait(b);
  REQUIRE(c);
  MMDEPLOY_CRITICAL("{}", *(Value*)c);
}

auto Gen(int k) {
  return [k](...) -> Value {
    MMDEPLOY_INFO("{}: start sleeping", k);
    std::this_thread::sleep_for(std::chrono::seconds(1));
    MMDEPLOY_INFO("{}: done sleeping", k);
    return k;
  };
}

void Fn() {
  auto pool = __static_thread_pool::StaticThreadPool{4};
  auto sched = pool.GetScheduler();
  //  auto sched = InlineScheduler{};
  auto begin = Schedule(sched);
  auto a = Then(begin, []() -> Value { return 100; });
  auto b = LetValue(a, [&](Value& v) {
    auto b1 = Then(Schedule(sched), Gen(1));
    auto b2 = Then(Schedule(sched), Gen(2));
    auto b3 = Then(Schedule(sched), Gen(3));
    auto b4 = Then(Schedule(sched), Gen(4));
    auto b = WhenAll(b1, b2, b3, b4);
    return LetValue(b, [&](auto&... vals) {
      MMDEPLOY_INFO("vals = {}", std::tuple{vals.template get<int>()...});
      return Just(Value((vals.template get<int>() + ...)));
    });
  });
  auto v = SyncWait(b);
  MMDEPLOY_INFO("threaded split: {}", v);
}

void Gn() {
  auto v = SyncWait(LetValue(Just(Value(100)), [&](Value& v) {
    return LetValue(Just(Value(200)), [&](Value& u) {
      return LetValue(Just(Value(300)), [&](Value& w) {
        return LetValue(Just(Value(400)), [&](Value& x) {
          return Just(Value(u.get<int>() + v.get<int>() + w.get<int>() + x.get<int>()));
        });
      });
    });
  }));
  MMDEPLOY_INFO("Gn: {}", v);
}

TEST_CASE("test threaded split", "[execution]") { Fn(); }

TEST_CASE("test inference pipeline", "[execution][pipeline]") { Gn(); }

TEST_CASE("test generic just", "[execution]") {
  auto j = Just(1, 2, 3, 4.0);
  auto s = LetValue(j, [](const auto&... vs) { return Just((vs + ...)); });
  auto v = SyncWait(s);
  MMDEPLOY_INFO("generic: {}", v);
}

TEST_CASE("test generic split", "[execution]") {
  auto j = Just(1, 2, 3);
  auto s = Split(j);
  auto a1 = Then(s, [](int x, auto...) { return x; });
  auto a2 = Then(s, [](int, int y, auto...) { return y; });
  auto a3 = Then(s, [](int, int, int z) { return z; });
  auto a = WhenAll(a3, a2, a1);
  auto [z, y, x] = SyncWait(a);
  MMDEPLOY_INFO("generic split: {} {} {}", z, y, x);
}

TEST_CASE("test bulk", "[execution]") {
  constexpr int N = 1024;
  std::vector<float> a(N), b(N), c(N);
  std::iota(begin(a), end(a), 0);
  std::iota(rbegin(b), rend(b), 0);
  auto init = Just(std::move(a), std::move(b), std::move(c));
  auto fma = Bulk(init, N, [](int index, const auto& a, const auto& b, auto& c) {
    c[index] += a[index] * b[index];
  });
  auto [x, y, z] = SyncWait(fma);
  MMDEPLOY_INFO("{}", z);
}

TEST_CASE("test schedule_after", "[execution]") {
  TimedSingleThreadContext context;
  auto sched = context.GetScheduler();

  auto s = ScheduleAfter(sched, std::chrono::seconds(1));
  std::chrono::steady_clock::time_point start;
  auto t = Then(s, [&start] {
    auto end = std::chrono::steady_clock::now();
    auto dt = std::chrono::duration<double>(end - start).count();
    MMDEPLOY_INFO("{} seconds passed", dt);
    return 0;
  });
  start = std::chrono::steady_clock::now();
  SyncWait(t);
}
