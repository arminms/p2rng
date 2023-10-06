#include <catch2/catch_all.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>

#include <p2rng/bind.hpp>
#include <p2rng/pcg/pcg_random.hpp>
#include <p2rng/trng/uniform_dist.hpp>
#include <p2rng/algorithm/generate.hpp>

const unsigned long seed_pi{3141592654};

struct equal
{   template <typename Tuple>
    __host__ __device__
    bool operator()(Tuple t)
    {   return (std::abs(thrust::get<0>(t) - thrust::get<1>(t)) < 0.00001);   }
};

TEST_CASE( "Device Info - ROCm")
{   hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    CHECK('A' == prop.name[0]);
    std::cout << "Running on: " << prop.name << std::endl;
}

TEMPLATE_TEST_CASE("generate() - ROCm", "[10K][pcg32]", float, double)
{   typedef TestType T;
    const auto n{10'007};
    std::vector<T> vr(n);
    trng::uniform_dist<T> u(10, 100);

    std::generate
    (   std::begin(vr)
    ,   std::end(vr)
    ,   std::bind(u, pcg32(seed_pi))
    );

    SECTION("std::generate()")
    {   CHECK( std::all_of
        (   std::begin(vr)
        ,   std::end(vr)
        ,   [] (T v)
            { return ( v >= 10 && v < 100 ); }
        ) );
    }

    SECTION("p2rng::rocm::generate()")
    {   thrust::device_vector<T> dvt(n);
        p2rng::rocm::generate
        (   std::begin(dvt)
        ,   std::end(dvt)
        ,   p2rng::bind(u, pcg32(seed_pi))
        );

        thrust::device_vector<T> dvr(n);
        thrust::copy(vr.begin(), vr.end(), dvr.begin());

        CHECK( thrust::all_of
        (   thrust::make_zip_iterator(thrust::make_tuple(dvr.begin(), dvt.begin()))
        ,   thrust::make_zip_iterator(thrust::make_tuple(dvr.end(), dvt.end()))
        ,   equal()
        ) );
    }
}
