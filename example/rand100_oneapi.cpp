#include <iostream>
#include <iomanip>

#include <oneapi/dpl/iterator>
#include <sycl/sycl.hpp>
#include <p2rng/p2rng.hpp>

int main(int argc, char* argv[])
{   const unsigned long seed{2718281828};
    const auto n{100};
    sycl::buffer<int> v{sycl::range(n)};
    sycl::queue q;

    p2rng::oneapi::generate_n
    (   dpl::begin(v)
    ,   n
    ,   p2rng::bind(trng::uniform_int_dist(10, 100), pcg32(seed))
    ,   q   // this is optional and can be omitted
    ).wait();

    sycl::host_accessor va{v, sycl::read_only};
    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
            std::cout << '\n';
        std::cout << std::setw(3) << va[i];
    }
    std::cout << '\n' << std::endl;
}


