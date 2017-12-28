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

#include <iostream>
#include <vector>

#include <SpatialCL/tree.hpp>
#include <SpatialCL/query.hpp>

#include <QCL/qcl.hpp>
#include <QCL/qcl_array.hpp>

#include "../common/environment.hpp"
#include "../common/random_particles.hpp"

using tree_type = spatialcl::hilbert_bvh_sp3d_tree<3>;
using type_system = tree_type::type_system;
using scalar = type_system::scalar;

const std::size_t num_particles = 128;
const std::size_t num_queries = 128;
constexpr std::size_t K = 8;

// This is necessary to use std::find with cl_float4's
// which is required in the verification process
bool operator==(const cl_float4& a,
                const cl_float4& b)
{
  return a.s[0] == b.s[0] &&
         a.s[1] == b.s[1] &&
         a.s[2] == b.s[2];
}

std::array<cl_float4, K>
naive_knn_query(const std::vector<cl_float4>& particles,
                cl_float4 query_point)
{
  std::array<cl_float4, K> result;
  std::array<scalar, K> distances2;
  unsigned max_distance_idx = 0;

  for(std::size_t i = 0; i < K; ++i)
    distances2[i] = std::numeric_limits<scalar>::max();

  for(const auto p: particles)
  {
    cl_float4 delta = query_point;
    for(std::size_t i = 0; i < 3; ++i)
      delta.s[i] -= p.s[i];

    scalar dist2 = 0.0f;
    for(std::size_t i = 0; i < 3; ++i)
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

std::size_t verify_results(const std::vector<cl_float4>& particles,
                           const std::vector<cl_float4>& query_points,
                           const std::vector<cl_float4>& results)
{
  std::size_t num_errors = 0;
  for(std::size_t i = 0; i < query_points.size(); ++i)
  {
    cl_float4 query = query_points[i];

    std::array<cl_float4, K> knn =
        naive_knn_query(particles, query);

    auto results_begin = results.begin() + i * K;
    auto results_end = results_begin + K;
    for(cl_float4 p : knn)
      if(std::find(results_begin, results_end, p) == results_end)
        ++num_errors;

  }
  return num_errors;
}

int main(int argc, char* argv[])
{
  common::environment env;
  qcl::device_context_ptr ctx = env.get_device_context();

  std::vector<cl_float4> particles;
  // Create random 3D positions
  common::random_particles<scalar, 3> rnd;
  rnd(num_particles, particles);

  tree_type gpu_tree{ctx, particles};

  std::vector<cl_float4> query_points;
  rnd(num_queries, query_points);

  qcl::device_array<cl_float4> queries{ctx, query_points};
  qcl::device_array<cl_float4> result{ctx, K * queries.size()};

  // Define Query
  using knn_query_engine =
    spatialcl::query::default_knn_query_engine
    <
      type_system, K
    >;

  knn_query_engine query_engine;
  knn_query_engine::handler_type query_handler {
    queries.get_buffer(),
    result.get_buffer(),
    queries.size()
  };

  // Executy query
  gpu_tree.execute_query(query_engine, query_handler);

  cl_int err = ctx->get_command_queue().finish();
  qcl::check_cl_error(err, "Error while executing KNN query");

  // Retrieve results
  std::vector<cl_float4> host_results;
  result.read(host_results);

  // Verify results
  std::cout << "Verifying results..." << std::endl;
  std::size_t num_errors = verify_results(particles,
                                          query_points,
                                          host_results);
  std::cout << num_queries << " queries were executed with "
            << num_errors << " errors" << std::endl;

}
