#include <benchmark/benchmark.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <p2rng/bind.hpp>
#include <p2rng/pcg/pcg_random.hpp>
#include <p2rng/trng/uniform_dist.hpp>
#include <p2rng/algorithm/generate.hpp>

const unsigned long seed_pi{3141592654};

//----------------------------------------------------------------------------//
// generate() algortithm

template <class T>
void p2rng_generate_cuda(benchmark::State& st)
{   size_t n = size_t(st.range());
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    thrust::device_vector<T> v(n);

    for (auto _ : st)
    {   cudaEventRecord(start);
        p2rng::generate
        (   v.begin()
        ,   v.end()
        ,   p2rng::bind(trng::uniform_dist<T>(10, 100), pcg32(seed_pi))
        );
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        st.SetIterationTime(milliseconds * 0.001f);
    }
    cudaEventDestroy(start); cudaEventDestroy(stop);

    st.counters["BW (GB/s)"] = benchmark::Counter
    (   (n * sizeof(T)) / 1e9
    ,   benchmark::Counter::kIsIterationInvariantRate
    );
}

BENCHMARK_TEMPLATE(p2rng_generate_cuda, float)
->  RangeMultiplier(2)
->  Range(1<<20, 1<<24)
->  UseManualTime()
->  Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(p2rng_generate_cuda, double)
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

    // adding GPU context
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::stringstream os;
    os << "\n  " << prop.name
       << "\n  L2 Cache: " << prop.l2CacheSize / 1024 << " KiB"
       << "\n  Number of SMs: x" << prop.multiProcessorCount
       << "\n  Peak Memory Bandwidth: "
       << std::fixed << std::setprecision(0)
       // based on https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-c
       << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
       << " (GB/s)";
    benchmark::AddCustomContext("GPU", os.str());

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
    return 0;
}