#include <algorithm>
#include <numeric>
#include <vector>

#include <catch2/catch_all.hpp>

#include <p2rng/bind.hpp>
#include <p2rng/pcg/pcg_random.hpp>
#include <p2rng/trng/uniform_dist.hpp>
#include <p2rng/algorithm/generate.hpp>

const unsigned long seed_pi{3141592654};

TEMPLATE_TEST_CASE( "generate() - OpenMP", "[10K][pcg32]", float, double)

{   typedef TestType T;
    const auto n{10'007};

    trng::uniform_dist<T> u(10, 100);
    std::vector<T> vr(n), vt(n);
    std::vector<size_t> idx(n);

    std::iota(std::begin(idx), std::end(idx), 0);

    std::generate
    (   std::begin(vr)
    ,   std::end(vr)
    ,   std::bind(u, pcg32(seed_pi))
    // ,   pcg32(seed_pi)
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
    {   p2rng::generate
        (   std::begin(vt)
        ,   std::end(vt)
        ,   p2rng::bind(u, pcg32(seed_pi))
        );

        CHECK( std::all_of
        (   std::begin(idx)
        ,   std::end(idx)
        ,   [&] (size_t i)
            { return ( std::abs(vr[i] - vt[i]) < 0.00001 ); }
        ) );
    }
}
