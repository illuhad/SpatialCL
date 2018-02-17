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

#include <iostream>

#include <boost/preprocessor/stringize.hpp>

#include <SpatialCL/tree.hpp>
#include <SpatialCL/query.hpp>

#include <QCL/qcl.hpp>
#include <QCL/qcl_array.hpp>

#include "../common/environment.hpp"
#include "../common/random_vectors.hpp"
#include "../common/verification_range.hpp"

constexpr std::size_t particle_dimension = 3;

const std::size_t num_particles = 32000;
const std::size_t num_queries = 2000;

constexpr std::size_t max_retrieved_particles = 8;

using tree_type = spatialcl::hilbert_bvh_sp3d_tree<particle_dimension>;
using type_system = tree_type::type_system;
using scalar = type_system::scalar;

constexpr scalar query_diameter = 0.05f;

// Define queries
using strict_dfs_range_engine =
  spatialcl::query::strict_dfs_range_query_engine<type_system,
                                                  max_retrieved_particles>;

using relaxed_dfs_range_engine =
  spatialcl::query::relaxed_dfs_range_query_engine<type_system,
                                                  max_retrieved_particles>;

template<std::size_t Group_size>
using grouped_dfs_range_engine =
  spatialcl::query::grouped_dfs_range_query_engine<type_system,
                                                    max_retrieved_particles,
                                                    Group_size>;

using register_bfs_range_engine =
  spatialcl::query::register_bfs_range_query_engine<type_system,
                                                    max_retrieved_particles>;


using particle_type = spatialcl::configuration<type_system>::particle_type;
using vector_type = spatialcl::configuration<type_system>::vector_type;
constexpr std::size_t dimension = type_system::dimension;

template<class Query_engine>
std::size_t execute_range_query_test(const qcl::device_context_ptr& ctx,
                                     const tree_type& tree,
                                     const std::vector<vector_type>& host_queries_min,
                                     const std::vector<vector_type>& host_queries_max,
                                     const qcl::device_array<vector_type>& queries_min,
                                     const qcl::device_array<vector_type>& queries_max,
                                     const std::vector<particle_type>& particles,
                                     qcl::device_array<particle_type>& result,
                                     qcl::device_array<cl_uint>& num_results)
{
  Query_engine query_engine;

  typename Query_engine::handler_type query_handler {
    queries_min.get_buffer(),
    queries_max.get_buffer(),
    result.get_buffer(),
    num_results.get_buffer(),
    queries_min.size()
  };

  std::cout << "Executing query..." << std::endl;

  tree.execute_query(query_engine, query_handler);

  cl_int err = ctx->get_command_queue().finish();
  qcl::check_cl_error(err, "Error while executing range query");

  // Retrieve results
  std::vector<particle_type> host_results;
  std::vector<cl_uint> host_num_results;
  result.read(host_results);
  num_results.read(host_num_results);

  std::cout << "Verifying results, please wait..." << std::endl;
  // Verify results
  common::verification::naive_cpu_range_verifier<type_system> verifier{
    host_queries_min,
    host_queries_max,
    max_retrieved_particles
  };

  return verifier(particles, host_results, host_num_results);
}

int main()
{
  // Setup particle tree
  common::environment env;
  qcl::device_context_ptr ctx = env.get_device_context();

  std::vector<particle_type> particles;
  common::random_vectors<scalar, particle_dimension> rnd;
  rnd(num_particles, particles);

  tree_type gpu_tree{ctx, particles};

  // Create random ranges for the queries
  std::vector<vector_type> query_points;
  rnd(num_queries, query_points);

  std::vector<vector_type> host_ranges_min = query_points;
  std::vector<vector_type> host_ranges_max = query_points;

  std::transform(host_ranges_min.begin(), host_ranges_min.end(),
                 host_ranges_min.begin(),
                 [&](vector_type current){
    for(std::size_t i = 0; i < dimension; ++i)
      current.s[i] -= query_diameter / 2;
    return current;
  });

  std::transform(host_ranges_max.begin(), host_ranges_max.end(),
                 host_ranges_max.begin(),
                 [&](vector_type current){
    for(std::size_t i = 0; i < dimension; ++i)
      current.s[i] += query_diameter / 2;
    return current;
  });

  // Create buffers on the compute device
  qcl::device_array<vector_type> ranges_min{ctx, host_ranges_min};
  qcl::device_array<vector_type> ranges_max{ctx, host_ranges_max};
  qcl::device_array<particle_type> result_particles{ctx,
                                    num_queries*max_retrieved_particles};
  qcl::device_array<cl_uint> result_num_retrieved_particles{ctx,
                                                            num_queries};


  std::size_t num_errors = 0;

#define RUN_TEST(test_name) \
  num_errors = \
      execute_range_query_test<test_name>(ctx,              \
                                          gpu_tree,         \
                                          host_ranges_min,  \
                                          host_ranges_max,  \
                                          ranges_min,       \
                                          ranges_max,       \
                                          particles,        \
                                          result_particles, \
                                          result_num_retrieved_particles); \
  std::cout << BOOST_PP_STRINGIZE(test_name) <<" completed queries with " \
            << num_errors << " errors." << std::endl

  RUN_TEST(strict_dfs_range_engine);
  RUN_TEST(relaxed_dfs_range_engine);
  RUN_TEST(grouped_dfs_range_engine<16>);
  RUN_TEST(grouped_dfs_range_engine<32>);
  RUN_TEST(grouped_dfs_range_engine<64>);
  RUN_TEST(register_bfs_range_engine);
 
  return 0;
}
