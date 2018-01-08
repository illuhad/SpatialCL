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

#ifndef VERIFICATION_KNN_HPP
#define VERIFICATION_KNN_HPP

#include <SpatialCL/configuration.hpp>

#include <vector>
#include <array>
#include <limits>

// This is necessary to use std::find with cl_float4's
// which is required in the verification process
bool operator==(const cl_float4& a,
                const cl_float4& b)
{
  return a.s[0] == b.s[0] &&
         a.s[1] == b.s[1] &&
         a.s[2] == b.s[2];
}

// This is necessary to use std::find with cl_float4's
// which is required in the verification process
bool operator==(const cl_float2& a,
                const cl_float2& b)
{
  return a.s[0] == b.s[0] &&
         a.s[1] == b.s[1];
}

namespace common {
namespace verification {

/// Verifies KNN queries by comparing results to the query results
/// obtained from a naive (and inefficient) algorithm run on the CPU.
template<class Type_descriptor, std::size_t K>
class naive_cpu_knn_verifier
{
public:
  using particle_type =
    typename spatialcl::configuration<Type_descriptor>::particle_type;
  using vector_type =
    typename spatialcl::configuration<Type_descriptor>::vector_type;
  using scalar =
    typename spatialcl::configuration<Type_descriptor>::scalar;

  static constexpr std::size_t dimension = Type_descriptor::dimension;

  naive_cpu_knn_verifier(const std::vector<vector_type>& query_points)
    : _query_points{query_points}
  {
  }

  /// Verifies the result of a KNN query
  /// \return The number of detected wrong results
  /// \param particles The particle set of which the KNN should be found
  /// \param results The results of a KNN query described by the set of
  /// query points supplied in the constructor. Assumes that the memory layout
  /// is such that the i-th result of query j is located at
  /// \c result[j*K + i].
  std::size_t operator()(const std::vector<particle_type>& particles,
                         const std::vector<particle_type>& results) const
  {
    assert(results.size() == this->_query_points.size() * K);

    std::size_t num_errors = 0;
    for(std::size_t i = 0; i < _query_points.size(); ++i)
    {
      vector_type query = _query_points[i];

      std::array<particle_type, K> knn =
          this->naive_knn_query(particles, query);

      auto results_begin = results.begin() + i * K;
      auto results_end = results_begin + K;
      for(particle_type p : knn)
        if(std::find(results_begin, results_end, p) == results_end)
          ++num_errors;

    }
    return num_errors;
  }

private:

  std::array<particle_type, K>
  naive_knn_query(const std::vector<particle_type>& particles,
                  vector_type query_point) const
  {

    std::array<particle_type, K> result;
    if(particles.size() < K)
    {
      std::copy(particles.begin(), particles.end(), result.begin());
      return result;
    }

    std::array<scalar, K> distances2;
    unsigned max_distance_idx = 0;

    for(std::size_t i = 0; i < K; ++i)
      distances2[i] = std::numeric_limits<scalar>::max();

    for(const auto p: particles)
    {
      particle_type delta = query_point;
      for(std::size_t i = 0; i < dimension; ++i)
        delta.s[i] -= p.s[i];

      scalar dist2 = 0.0f;
      for(std::size_t i = 0; i < dimension; ++i)
        dist2 += delta.s[i] * delta.s[i];

      if(dist2 < distances2[max_distance_idx])
      {
        result[max_distance_idx] = p;
        distances2[max_distance_idx] =  dist2;

        max_distance_idx = std::distance(distances2.begin(),
                               std::max_element(distances2.begin(),
                                                distances2.end()));
      }
    }

    return result;
  }

  std::vector<vector_type> _query_points;
};

}
}

#endif
