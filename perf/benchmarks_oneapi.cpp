#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include <benchmark/benchmark.h>

#include <p2rng/bind.hpp>
#include <p2rng/pcg/pcg_random.hpp>
#include <p2rng/trng/uniform_dist.hpp>
#include <p2rng/algorithm/generate.hpp>

const unsigned long seed_pi{3141592654};

//----------------------------------------------------------------------------//
// generate() algorithm

template <class T>
void p2rng_generate_oneapi(benchmark::State& st)
{   size_t n = size_t(st.range());
    // enabling SYCL queue profiling
    auto pl = sycl::property_list{sycl::property::queue::enable_profiling()};
    sycl::queue q = sycl::queue(pl);
    sycl::buffer<T> v(n);

    for (auto _ : st)
    {   auto event = p2rng::generate_n
        (   dpl::begin(v)
        ,   n
        ,   p2rng::bind(trng::uniform_dist<T>(10, 100), pcg32(seed_pi))
        ,   q
        );
        event.wait();
        auto start_time = event.template
            get_profiling_info<sycl::info::event_profiling::command_start>();
        auto end_time = event.template
            get_profiling_info<sycl::info::event_profiling::command_end>();
        st.SetIterationTime((end_time - start_time) * 1e-9f);
    }

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(T)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(p2rng_generate_oneapi, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(p2rng_generate_oneapi, double)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

//----------------------------------------------------------------------------//
// main()

int main(int argc, char** argv)
{   benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;

    // adding oneAPI context
    sycl::queue q;
    std::stringstream os;
    os << "\n  Running on: "
       << q.get_device().get_info<sycl::info::device::name>()
       << "\n  Clock frequency: "
       << q.get_device().get_info<sycl::info::device::max_clock_frequency>()
       << "\n  Compute units: "
       << q.get_device().get_info<sycl::info::device::max_compute_units>();
    benchmark::AddCustomContext("oneAPI", os.str());

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}