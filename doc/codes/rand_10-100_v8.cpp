#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>
#include <functional>
#include <execution>
#include <p2rng/pcg/pcg_random.hpp>
#include <thread>

int main(int argc, char* argv[])
{   const unsigned long seed{2718281828};
    const auto n{100};
    std::vector<int> v(n);
    std::uniform_int_distribution<int> u(10, 99);
    std::hash<std::thread::id> hasher;

    std::generate_n
    (   std::execution::par
    ,   std::begin(v)
    ,   n
    ,   [&]()
        {   thread_local pcg32 r(seed, hasher(std::this_thread::get_id()));
            return u(r);
        }
    );

    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
            std::cout << '\n';
        std::cout << std::setw(3) << v[i];
    }
    std::cout << '\n' << std::endl;
}
