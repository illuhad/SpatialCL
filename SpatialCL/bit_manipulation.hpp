/*
 * This file is part of SpatialCL, a library for the spatial processing of
 * particles.
 *
 * Copyright (c) 2017 Aksel Alpay
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

#ifndef BIT_MANIPULATION_HPP
#define BIT_MANIPULATION_HPP

#include "configuration.hpp"

namespace spatialcl {

QCL_STANDALONE_MODULE(bit_manipulation)
QCL_STANDALONE_SOURCE
(
  QCL_RAW(
  ulong bit_split3(uint input)
  {
    ulong x = input;
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;

    return x;
  }

  ulong bit_split2(uint input)
  {
    ulong x = input;

    //x &= 0xffffffff // not necessary because we are using all 32 bits
    x = (x | x << 16) & 0xffff0000ffff;
    x = (x | x << 8) & 0xff00ff00ff00ff;
    x = (x | x << 4) & 0xf0f0f0f0f0f0f0f;
    x = (x | x << 2) & 0x3333333333333333;
    x = (x | x << 1) & 0x5555555555555555;

    return x;
  }

  ulong n_bits_set(uint n)
  {
    // shifting a 64 bit integer by 64 bits is undefined on most architectures,
    // hence we need special treatment for ths case.
    if(n == 64)
      return ~0ul;

    ulong result = 1ul << n;
    return result - 1;
  }

  ulong interleave_bits2(uint a, uint b)
  {
    return bit_split2(a) | (bit_split2(b) << 1);
  }

  ulong interleave_bits3(uint a, uint b, uint c)
  {
    return bit_split3(a) | (bit_split3(b) << 1) | (bit_split3(c) << 2);
  }

  uint find_most_significant_bit(ulong x)
  {
    x |= (x >> 1);
    x |= (x >> 2);
    x |= (x >> 4);
    x |= (x >> 8);
    x |= (x >> 16);
    x |= (x >> 32);
    return(x & ~(x >> 1));
  }

  ulong get_next_power_of_two(ulong x)
  {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x++;

    return x;
  }
  )
)

}

#endif
