#ifndef _P2RNG_BIND_HPP_
#define _P2RNG_BIND_HPP_

#include <p2rng/device.hpp>

namespace p2rng {

template<typename Distribution, typename Engine>
struct bind_struct
{   bind_struct(
        Distribution d
    ,   Engine e
    )
    :   _d(d)
    ,   _e(e)
    {}

    P2RNG_DEVICE_CODE
    auto operator() () -> typename Distribution::result_type
    {   return _d(_e);   }

    P2RNG_DEVICE_CODE
    void discard(typename Engine::state_type n)
    {   _e.discard(n);  }

private:
    Distribution _d;
    Engine       _e;
};

template<typename Distribution, typename Engine>
bind_struct<Distribution, Engine> bind(Distribution d, Engine e)
{   return bind_struct<Distribution, Engine>(d, e);   }

} // end p2rng namespace

#endif  //_P2RNG_BIND_HPP_