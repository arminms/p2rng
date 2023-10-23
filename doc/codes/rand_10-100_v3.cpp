#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <algorithm>



int main(int argc, char* argv[])
{   const unsigned long seed{2718281828};
    const auto n{100};
    std::vector<int> v(n);
    std::mt19937 r(seed);
    std::uniform_int_distribution<int> u(10, 100);

    std::generate_n
    (   std::begin(v)
    ,   n
    ,   [&]() { return u(r); }
    );

    for (size_t i = 0; i < n; ++i)
    {   if (0 == i % 10)
            std::cout << '\n';
        std::cout << std::setw(3) << v[i];
    }
    std::cout << '\n' << std::endl;
}
