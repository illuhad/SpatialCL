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

#include <SpatialCL/tree.hpp>
#include <SpatialCL/query.hpp>

#include <QCL/qcl.hpp>
#include <QCL/qcl_array.hpp>

#include "../common/environment.hpp"
#include "../common/random_vectors.hpp"

const std::size_t num_particles = 128;
const std::size_t num_queries = 128;
constexpr std::size_t max_retrieved_particles = 8;

using tree_type = spatialcl::hilbert_bvh_sp3d_tree<3>;
using type_system = tree_type::type_system;
using scalar = type_system::scalar;

constexpr scalar query_diameter = 0.2f;

bool is_particle_within_box(cl_float4 particle,
                            cl_float4 box_min,
                            cl_float4 box_max)
{
  for(std::size_t k = 0; k < 3; ++k)
    if( particle.s[k] < box_min.s[k]
     || particle.s[k] > box_max.s[k])
      return false;
  return true;
}

std::size_t get_num_particles_in_range(const std::vector<cl_float4>& particles,
                                       cl_float4 query_range_min,
                                       cl_float4 query_range_max)
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

std::size_t verify_results(const std::vector<cl_float4>& particles,
                           const std::vector<cl_float4>& query_ranges_min,
                           const std::vector<cl_float4>& query_ranges_max,
                           const std::vector<cl_float4>& retrieved_particles,
                           const std::vector<cl_uint>& num_retrieved_particles)
{
  std::size_t num_errors = 0;
  assert(query_ranges_min.size() == query_ranges_max.size()
         && query_ranges_min.size() == num_queries);
  for(std::size_t i = 0; i < query_ranges_min.size(); ++i)
  {
    std::size_t num_particles = num_retrieved_particles[i];
    for(std::size_t j = 0; j < num_particles; ++j)
    {
      cl_float4 particle = retrieved_particles[i*max_retrieved_particles + j];
      if(!is_particle_within_box(particle, query_ranges_min[i], query_ranges_max[i]))
      {
        std::cout << "Query " << i << " returned incorrect results." << std::endl;
        ++num_errors;
      }
    }
    if(num_particles < max_retrieved_particles)
    {
      // Make sure we have not missed any particles
      if(get_num_particles_in_range(particles,
                                    query_ranges_min[i],
                                    query_ranges_max[i]) != num_particles)
      {
        std::cout << "Query " << i << " returned incomplete results." << std::endl;
        ++num_errors;
      }
    }
  }
  return num_errors;
}

int main()
{
  // Setup particle tree
  common::environment env;
  qcl::device_context_ptr ctx = env.get_device_context();

  std::vector<cl_float4> particles;
  common::random_vectors<scalar, 3> rnd;
  rnd(num_particles, particles);

  tree_type gpu_tree{ctx, particles};

  // Create random ranges for the queries
  std::vector<cl_float4> query_points;
  rnd(num_queries, query_points);

  std::vector<cl_float4> host_ranges_min = query_points;
  std::vector<cl_float4> host_ranges_max = query_points;

  std::transform(host_ranges_min.begin(), host_ranges_min.end(),
                 host_ranges_min.begin(),
                 [&](cl_float4 current){
    for(int i = 0; i < 4; ++i)
      current.s[i] -= query_diameter / 2;
    return current;
  });

  std::transform(host_ranges_max.begin(), host_ranges_max.end(),
                 host_ranges_max.begin(),
                 [&](cl_float4 current){
    for(int i = 0; i < 4; ++i)
      current.s[i] += query_diameter / 2;
    return current;
  });

  // Create buffers on the compute device
  qcl::device_array<cl_float4> ranges_min{ctx, host_ranges_min};
  qcl::device_array<cl_float4> ranges_max{ctx, host_ranges_max};
  qcl::device_array<cl_float4> result_particles{ctx,
        num_queries*max_retrieved_particles};
  qcl::device_array<cl_uint> result_num_retrieved_particles{ctx,
                                                            num_queries};

  // Define query
  using range_query_engine =
    spatialcl::query::default_range_query_engine
    <
      type_system, max_retrieved_particles
    >;

  range_query_engine query_engine;
  range_query_engine::handler_type query_handler{
    ranges_min.get_buffer(),
    ranges_max.get_buffer(),
    result_particles.get_buffer(),
    result_num_retrieved_particles.get_buffer(),
    num_queries
  };

  // Run query
  gpu_tree.execute_query(query_engine, query_handler);

  cl_int err = ctx->get_command_queue().finish();

  qcl::check_cl_error(err, "Error while executing range query");

  // Retrieve results
  std::vector<cl_float4> host_result_particles;
  std::vector<cl_uint> host_num_results;
  result_particles.read(host_result_particles);
  result_num_retrieved_particles.read(host_num_results);

  for(std::size_t i = 0; i < num_queries; ++i)
  {
    std::cout << "Query " << i << " retrieved "
              << host_num_results[i] << " particles.\n";

  }

  // Verify results
  std::cout << "Verifying results..." << std::endl;
  std::size_t num_errors = verify_results(particles,
                                          host_ranges_min, host_ranges_max,
                                          host_result_particles, host_num_results);
  std::cout << num_queries << " queries were executed with "
            << num_errors << " errors" << std::endl;
}
