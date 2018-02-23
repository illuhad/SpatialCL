/*
 * This file is part of SpatialCL, a library for the spatial processing of
 * particles.
 *
 * Copyright (c) 2018 Aksel Alpay
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstddef>

namespace spatialcl{
namespace utils{
namespace binary{

template <std::size_t N>
struct small_binary_logarithm
{
  static constexpr bool is_input_valid = false;
};

#define SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(N, val) \
  template <>                                           \
  struct small_binary_logarithm<N>                      \
  {                                                     \
    static constexpr std::size_t value = val;           \
    static constexpr bool is_input_valid = true;        \
};

SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(1, 0)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(2, 1)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(4, 2)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(8, 3)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(16, 4)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(32, 5)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(64, 6)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(128, 7)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(256, 8)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(512, 9)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(1024, 10)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(2048, 11)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(4096, 12)
SPATIALCL_IMPL_DECLARE_BINARY_LOGARITHM(8192, 13)

template<std::size_t N>
struct is_small_power2
{
  static constexpr bool value = small_binary_logarithm<N>::is_input_valid;
};

}
}
}
