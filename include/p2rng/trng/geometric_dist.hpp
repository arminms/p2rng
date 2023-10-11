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

#if !(defined TRNG_GEOMETRIC_DIST_HPP)

#define TRNG_GEOMETRIC_DIST_HPP

#include <p2rng/device.hpp>
#include <p2rng/trng/utility.hpp>
#include <p2rng/trng/math.hpp>
#include <climits>
#include <ostream>
#include <istream>
#include <iomanip>
#include <vector>
#include <ciso646>

namespace trng {

  // non-uniform random number generator class
  class geometric_dist {
  public:
    using result_type = int;

    class param_type {
    private:
      double p_, q_, one_over_ln_q_;

      P2RNG_DEVICE_CODE
      double q() const { return q_; }
      P2RNG_DEVICE_CODE
      double one_over_ln_q() const { return one_over_ln_q_; }

    public:
      P2RNG_DEVICE_CODE
      double p() const { return p_; }
      P2RNG_DEVICE_CODE
      void p(double p_new) {
        p_ = p_new;
        q_ = 1.0 - p_;
        one_over_ln_q_ = 1.0 / math::ln(q_);
      }
      P2RNG_DEVICE_CODE
      explicit param_type(double p = 0.5)
          : p_{p}, q_{1.0 - p_}, one_over_ln_q_{(1.0 / math::ln(q_))} {}
      friend class geometric_dist;
    };

  private:
    param_type P;

  public:
    // constructor
    P2RNG_DEVICE_CODE
    explicit geometric_dist(double p) : P{p} {}
    P2RNG_DEVICE_CODE
    explicit geometric_dist(const param_type &P) : P{P} {}
    // reset internal state
    P2RNG_DEVICE_CODE
    void reset() {}
    // random numbers
    template<typename R>
    P2RNG_DEVICE_CODE int operator()(R &r) {
      return static_cast<int>(math::ln(utility::uniformoo<double>(r)) * P.one_over_ln_q());
    }
    template<typename R>
    P2RNG_DEVICE_CODE int operator()(R &r, const param_type &p) {
      geometric_dist g(p);
      return g(r);
    }
    // property methods
    P2RNG_DEVICE_CODE
    int min() const { return 0; }
    P2RNG_DEVICE_CODE
    int max() const { return INT_MAX; }
    P2RNG_DEVICE_CODE
    const param_type &param() const { return P; }
    P2RNG_DEVICE_CODE
    void param(const param_type &P_new) { P = P_new; }
    P2RNG_DEVICE_CODE
    double p() const { return P.p(); }
    P2RNG_DEVICE_CODE
    void p(double p_new) { P.p(p_new); }
    // probability density function
    P2RNG_DEVICE_CODE
    double pdf(int x) const {
      return x < 0 ? 0.0 : P.p() * math::pow(P.q(), static_cast<double>(x));
    }
    // cumulative density function
    P2RNG_DEVICE_CODE
    double cdf(int x) const {
      return x < 0 ? 0.0 : 1.0 - math::pow(P.q(), static_cast<double>(x + 1));
    }
  };

  // -------------------------------------------------------------------

  // EqualityComparable concept
  P2RNG_DEVICE_CODE
  inline bool operator==(const geometric_dist::param_type &P1,
                         const geometric_dist::param_type &P2) {
    return P1.p() == P2.p();
  }

  P2RNG_DEVICE_CODE
  inline bool operator!=(const geometric_dist::param_type &P1,
                         const geometric_dist::param_type &P2) {
    return not(P1 == P2);
  }

  // Streamable concept
  template<typename char_t, typename traits_t>
  std::basic_ostream<char_t, traits_t> &operator<<(std::basic_ostream<char_t, traits_t> &out,
                                                   const geometric_dist::param_type &P) {
    std::ios_base::fmtflags flags(out.flags());
    out.flags(std::ios_base::dec | std::ios_base::fixed | std::ios_base::left);
    out << '(' << std::setprecision(math::numeric_limits<double>::digits10 + 1) << P.p() << ')';
    out.flags(flags);
    return out;
  }

  template<typename char_t, typename traits_t>
  std::basic_istream<char_t, traits_t> &operator>>(std::basic_istream<char_t, traits_t> &in,
                                                   geometric_dist::param_type &P) {
    double p;
    std::ios_base::fmtflags flags(in.flags());
    in.flags(std::ios_base::dec | std::ios_base::fixed | std::ios_base::left);
    in >> utility::delim('(') >> p >> utility::delim(')');
    if (in)
      P = geometric_dist::param_type(p);
    in.flags(flags);
    return in;
  }

  // -------------------------------------------------------------------

  // EqualityComparable concept
  P2RNG_DEVICE_CODE
  inline bool operator==(const geometric_dist &g1, const geometric_dist &g2) {
    return g1.param() == g2.param();
  }

  P2RNG_DEVICE_CODE
  inline bool operator!=(const geometric_dist &g1, const geometric_dist &g2) {
    return g1.param() != g2.param();
  }

  // Streamable concept
  template<typename char_t, typename traits_t>
  std::basic_ostream<char_t, traits_t> &operator<<(std::basic_ostream<char_t, traits_t> &out,
                                                   const geometric_dist &g) {
    std::ios_base::fmtflags flags(out.flags());
    out.flags(std::ios_base::dec | std::ios_base::fixed | std::ios_base::left);
    out << "[geometric " << g.param() << ']';
    out.flags(flags);
    return out;
  }

  template<typename char_t, typename traits_t>
  std::basic_istream<char_t, traits_t> &operator>>(std::basic_istream<char_t, traits_t> &in,
                                                   geometric_dist &g) {
    geometric_dist::param_type P;
    std::ios_base::fmtflags flags(in.flags());
    in.flags(std::ios_base::dec | std::ios_base::fixed | std::ios_base::left);
    in >> utility::ignore_spaces() >> utility::delim("[geometric ") >> P >> utility::delim(']');
    if (in)
      g.param(P);
    in.flags(flags);
    return in;
  }

}  // namespace trng

#endif
