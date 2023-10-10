#include <algorithm>
#include <functional>
#include <vector>

#include <benchmark/benchmark.h>

#include <p2rng/bind.hpp>
#include <p2rng/pcg/pcg_random.hpp>
#include <p2rng/trng/uniform_dist.hpp>
#include <p2rng/algorithm/generate.hpp>

const unsigned long seed_pi{3141592654};

//----------------------------------------------------------------------------//
// generate() algorithm

template <class T>
void stl_generate(benchmark::State& st)
{   size_t n = size_t(st.range());
    std::vector<T> v(n);

    for (auto _ : st)
        std::generate_n
        (   std::begin(v)
        ,   n
        ,   std::bind(trng::uniform_dist<T>(10, 100), pcg32(seed_pi))
        );

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(T)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(stl_generate, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(stl_generate, double)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  Unit(benchmark::kMillisecond);

template <class T>
void p2rng_generate_openmp(benchmark::State& st)
{   size_t n = size_t(st.range());
    std::vector<T> v(n);

    for (auto _ : st)
        p2rng::generate_n
        (   std::begin(v)
        ,   n
        ,   p2rng::bind(trng::uniform_dist<T>(10, 100), pcg32(seed_pi))
        );

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(T)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(p2rng_generate_openmp, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(p2rng_generate_openmp, double)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseRealTime()
->  Unit(benchmark::kMillisecond);

//----------------------------------------------------------------------------//
// main()

BENCHMARK_MAIN();