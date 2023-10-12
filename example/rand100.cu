#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <p2rng/p2rng.hpp>

int main(int argc, char* argv[])
{   const unsigned long seed{2718281828};
    const auto n{100};
    thrust::device_vector<int> v(n);

    p2rng::cuda::generate_n
    (   std::begin(v)
    ,   n
    ,   p2rng::bind(trng::uniform_int_dist(10, 100), pcg32(seed)) 
    );

    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
            std::cout << '\n';
        std::cout << std::setw(3) << v[i];
    }
    std::cout << '\n' << std::endl;
}
