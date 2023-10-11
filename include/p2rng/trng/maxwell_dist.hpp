// Copyright (c) 2000-2022, Heiko Bauke
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
//   * Redistributions of source code must retain the above copyright
//     notice, this list of conditions and the following disclaimer.
//
//   * Redistributions in binary form must reproduce the above
//     copyright notice, this list of conditions and the following
//     disclaimer in the documentation and/or other materials provided
//     with the distribution.
//
//   * Neither the name of the copyright holder nor the names of its
//     contributors may be used to endorse or promote products derived
//     from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
// OF THE POSSIBILITY OF SUCH DAMAGE.

#if !(defined TRNG_MAXWELL_DIST_HPP)

#define TRNG_MAXWELL_DIST_HPP

#include <p2rng/device.hpp>
#include <p2rng/trng/constants.hpp>
#include <p2rng/trng/limits.hpp>
#include <p2rng/trng/utility.hpp>
#include <p2rng/trng/math.hpp>
#include <p2rng/trng/special_functions.hpp>
#include <ostream>
#include <istream>
#include <iomanip>
#include <ciso646>

namespace trng {

  // uniform random number generator class
  template<typename float_t = double>
  class maxwell_dist {
  public:
    using result_type = float_t;

    class param_type {
    private:
      result_type theta_{1};

    public:
      P2RNG_DEVICE_CODE
      result_type theta() const { return theta_; }
      P2RNG_DEVICE_CODE
      void theta(result_type theta_new) { theta_ = theta_new; }
      param_type() = default;
      P2RNG_DEVICE_CODE
      explicit param_type(result_type theta) : theta_{theta} {}

      friend class maxwell_dist;

      // EqualityComparable concept
      friend P2RNG_DEVICE_CODE inline bool operator==(const param_type &P1,
                                                     const param_type &P2) {
        return P1.theta_ == P2.theta_;
      }

      friend P2RNG_DEVICE_CODE inline bool operator!=(const param_type &P1,
                                                     const param_type &P2) {
        return not(P1 == P2);
      }

      // Streamable concept
      template<typename char_t, typename traits_t>
      friend std::basic_ostream<char_t, traits_t> &operator<<(
          std::basic_ostream<char_t, traits_t> &out, const param_type &P) {
        std::ios_base::fmtflags flags(out.flags());
        out.flags(std::ios_base::dec | std::ios_base::fixed | std::ios_base::left);
        out << '(' << std::setprecision(math::numeric_limits<float_t>::digits10 + 1)
            << P.theta() << ')';
        out.flags(flags);
        return out;
      }

      template<typename char_t, typename traits_t>
      friend std::basic_istream<char_t, traits_t> &operator>>(
          std::basic_istream<char_t, traits_t> &in, param_type &P) {
        float_t theta;
        std::ios_base::fmtflags flags(in.flags());
        in.flags(std::ios_base::dec | std::ios_base::fixed | std::ios_base::left);
        in >> utility::delim('(') >> theta >> utility::delim(')');
        if (in)
          P = param_type(theta);
        in.flags(flags);
        return in;
      }
    };

  private:
    param_type P;

  public:
    // constructor
    P2RNG_DEVICE_CODE
    explicit maxwell_dist(result_type theta) : P{theta} {}
    P2RNG_DEVICE_CODE
    explicit maxwell_dist(const param_type &P) : P{P} {}
    // reset internal state
    P2RNG_DEVICE_CODE
    void reset() {}
    // random numbers
    template<typename R>
    P2RNG_DEVICE_CODE result_type operator()(R &r) {
      return icdf(utility::uniformoo<result_type>(r));
    }
    template<typename R>
    P2RNG_DEVICE_CODE result_type operator()(R &r, const param_type &P) {
      maxwell_dist g(P);
      return g(r);
    }
    // property methods
    P2RNG_DEVICE_CODE
    result_type min() const { return result_type(0); }
    P2RNG_DEVICE_CODE
    result_type max() const { return math::numeric_limits<result_type>::infinity(); }
    P2RNG_DEVICE_CODE
    const param_type &param() const { return P; }
    P2RNG_DEVICE_CODE
    void param(const param_type &P_new) { P = P_new; }
    P2RNG_DEVICE_CODE
    result_type theta() const { return P.theta(); }
    // probability density function
    P2RNG_DEVICE_CODE
    result_type pdf(result_type x) const {
      const result_type x2{x * x};
      const result_type t2{P.theta() * P.theta()};
      return math::constants<result_type>::sqrt_2_over_pi * x2 * math::exp(-x2 / (2 * t2)) /
             (t2 * P.theta());
    }
    // cumulative density function
    P2RNG_DEVICE_CODE
    result_type cdf(result_type x) const {
      return math::erf(x * math::constants<result_type>::one_over_sqrt_2 / P.theta()) -
             math::constants<result_type>::sqrt_2_over_pi * x *
                 math::exp(-x * x / (2 * P.theta() * P.theta())) / (P.theta());
    }
    // inverse cumulative density function
    P2RNG_DEVICE_CODE
    result_type icdf(result_type x) const {
      if (x < 0 or x > 1) {
#if !(defined __CUDA_ARCH__)
        errno = EDOM;
#endif
        return math::numeric_limits<result_type>::quiet_NaN();
      }
      if (x == 1)
        return math::numeric_limits<result_type>::infinity();
      if (x == 0)
        return 0;
      result_type y(2 * P.theta() * math::constants<result_type>::sqrt_2_over_pi);
      for (int i{0}; i < math::numeric_limits<result_type>::digits + 2; ++i) {
        result_type y_old = y;
        y -= (cdf(y) - x) / pdf(y);
        if (math::abs(y / y_old - 1) < 4 * math::numeric_limits<result_type>::epsilon())
          break;
      }
      return y;
    }
  };

  // -------------------------------------------------------------------

  // EqualityComparable concept
  template<typename float_t>
  P2RNG_DEVICE_CODE inline bool operator==(const maxwell_dist<float_t> &g1,
                                          const maxwell_dist<float_t> &g2) {
    return g1.param() == g2.param();
  }

  template<typename float_t>
  P2RNG_DEVICE_CODE inline bool operator!=(const maxwell_dist<float_t> &g1,
                                          const maxwell_dist<float_t> &g2) {
    return g1.param() != g2.param();
  }

  // Streamable concept
  template<typename char_t, typename traits_t, typename float_t>
  std::basic_ostream<char_t, traits_t> &operator<<(std::basic_ostream<char_t, traits_t> &out,
                                                   const maxwell_dist<float_t> &g) {
    std::ios_base::fmtflags flags(out.flags());
    out.flags(std::ios_base::dec | std::ios_base::fixed | std::ios_base::left);
    out << "[maxwell " << g.param() << ']';
    out.flags(flags);
    return out;
  }

  template<typename char_t, typename traits_t, typename float_t>
  std::basic_istream<char_t, traits_t> &operator>>(std::basic_istream<char_t, traits_t> &in,
                                                   maxwell_dist<float_t> &g) {
    typename maxwell_dist<float_t>::param_type P;
    std::ios_base::fmtflags flags(in.flags());
    in.flags(std::ios_base::dec | std::ios_base::fixed | std::ios_base::left);
    in >> utility::ignore_spaces() >> utility::delim("[maxwell ") >> P >> utility::delim(']');
    if (in)
      g.param(P);
    in.flags(flags);
    return in;
  }

}  // namespace trng

#endif
