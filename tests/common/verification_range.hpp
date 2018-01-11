/*
 * This file is part of SpatialCL, a library for the spatial processing of
 * particles.
 *
 * Copyright (c) 2017, 2018 Aksel Alpay
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

#ifndef VERIFICATION_RANGE_HPP
#define VERIFICATION_RANGE_HPP

#include <SpatialCL/configuration.hpp>

#include <cassert>
#include <vector>

namespace common {
namespace verification {

template<class Type_descriptor>
class naive_cpu_range_verifier
{
public:
  using particle_type =
    typename spatialcl::configuration<Type_descriptor>::particle_type;
  using vector_type =
    typename spatialcl::configuration<Type_descriptor>::vector_type;
  using scalar =
    typename spatialcl::configuration<Type_descriptor>::scalar;

  static constexpr std::size_t dimension = Type_descriptor::dimension;

  naive_cpu_range_verifier(const std::vector<vector_type>& queries_min,
                           const std::vector<vector_type>& queries_max,
                           const std::size_t max_num_retrieved_particles)
    : _queries_min{queries_min},
      _queries_max{queries_max},
      _max_retrieved_particles{max_num_retrieved_particles}
  {
    assert(_queries_min.size() == _queries_max.size());
  }

  /// Verifies the result of a range query
  /// \return The number of detected wrong results
  /// (both incomplete and incorrect results are counted)
  /// \param particles The particle set
  /// \param results The results of range queries described by the set of
  /// ranges supplied in the constructor. Assumes that the memory layout
  /// is such that the i-th result of query j is located at
  /// \c result[j*K + i].
  /// \param A vector in which the i-th entry is the number of found
  /// particles in the given range.
  std::size_t operator()(const std::vector<particle_type>& particles,
                         const std::vector<particle_type>& results,
                         const std::vector<cl_uint>& num_results) const
  {
    std::size_t num_errors = 0;
    assert(_queries_min.size() == num_results.size());
    assert(results.size() == _queries_min.size()*_max_retrieved_particles);

    for(std::size_t i = 0; i < _queries_min.size(); ++i)
    {
      std::size_t num_particles = num_results[i];
      for(std::size_t j = 0; j < num_particles; ++j)
      {
        particle_type particle = results[i*_max_retrieved_particles + j];
        if(!is_particle_within_box(particle, _queries_min[i], _queries_max[i]))
          ++num_errors;
      }
      std::size_t correct_num_particles =
          get_num_particles_in_range(particles,
                                     _queries_min[i],
                                     _queries_max[i]);
      correct_num_particles = std::min(correct_num_particles,
                                       _max_retrieved_particles);
      if(num_particles != correct_num_particles)
      {
        ++num_errors;
      }
    }
    return num_errors;
  }
private:
  bool is_particle_within_box(particle_type particle,
                              vector_type box_min,
                              vector_type box_max) const
  {
    for(std::size_t k = 0; k < dimension; ++k)
      if( particle.s[k] < box_min.s[k]
       || particle.s[k] > box_max.s[k])
        return false;
    return true;
  }

  std::size_t get_num_particles_in_range(const std::vector<particle_type>& particles,
                                         vector_type query_range_min,
                                         vector_type query_range_max) const
  {
    std::size_t counter = 0;

    for(std::size_t i = 0; i < particles.size(); ++i)
    {
      if(is_particle_within_box(particles[i],
                                query_range_min,
                                query_range_max))
        ++counter;
    }

    return counter;
  }

  std::vector<vector_type> _queries_min;
  std::vector<vector_type> _queries_max;

  const std::size_t _max_retrieved_particles;
};

}
}

#endif
