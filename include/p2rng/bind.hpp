//
// Copyright (c) 2023 Armin Sobhani (https://arminsobhani.ca)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
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