
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

#ifndef RANDOM_VECTORS_HPP
#define RANDOM_VECTORS_HPP

#include <random>
#include <SpatialCL/types.hpp>

namespace common {

template<class Scalar_type,
         std::size_t Num_dimensions>
class random_vectors
{
public:
  using vector_type = typename spatialcl::cl_vector_type
                      <
                         Scalar_type,
                         Num_dimensions
                      >::value;
  random_vectors(std::size_t seed = 1245)
    : _generator{seed}
  {}


  void operator()(std::size_t num_particles,
                  std::vector<vector_type>& out,
                  Scalar_type min_coordinates = 0.0f,
                  Scalar_type max_coordinates = 1.0f)
  {
    out.clear();

    std::uniform_real_distribution<Scalar_type> distribution{
      min_coordinates, max_coordinates
    };

    for(std::size_t i = 0; i < num_particles; ++i)
    {
      vector_type v = {};

      for(std::size_t j = 0; j < Num_dimensions; ++j)
        v.s[j] = distribution(_generator);
      out.push_back(v);
    }
  }
private:
  std::mt19937 _generator;
};

}

#endif
