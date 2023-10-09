#include <oneapi/dpl/execution>
#include <sycl/sycl.hpp>

#include <catch2/catch_all.hpp>

#include <p2rng/bind.hpp>
#include <p2rng/pcg/pcg_random.hpp>
#include <p2rng/trng/uniform_dist.hpp>
#include <p2rng/algorithm/generate.hpp>

const unsigned long seed_pi{3141592654};

TEST_CASE( "Device Info - oneAPI")
{   try
    {   sycl::queue q;
        WARN("Running on: " << q.get_device().get_info<sycl::info::device::name>());
    }
    catch(const sycl::exception& e)
    {   std::cerr << "Error: " << e.what() << std::endl;
    }
    REQUIRE(true);
}

TEMPLATE_TEST_CASE( "generate() - oneAPI", "[10K][pcg32]", float, double )
{   typedef TestType T;
    const auto n{10'007};
    sycl::queue q;
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

    SECTION("p2rng::generate()")
    {   sycl::buffer<T> dvt{sycl::range(n)};
        p2rng::generate
        (   dpl::begin(dvt)
        ,   dpl::end(dvt)
        ,   p2rng::bind(u, pcg32(seed_pi))
        ,   q
        ).wait();

        sycl::host_accessor vt{dvt, sycl::read_only};

        CHECK( std::all_of(
            dpl::counting_iterator<size_t>(0)
        ,   dpl::counting_iterator<size_t>(n)
        ,   [&] (size_t i)
            { return ( std::abs(vr[i] - vt[i]) < 0.00001 ); }
        ) );
    }
}
