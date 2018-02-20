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

#include <vector>
#include <iostream>
#include <string>

#include <boost/preprocessor/stringize.hpp>

#include <SpatialCL/tree.hpp>
#include <SpatialCL/query.hpp>

#include <QCL/qcl.hpp>
#include <QCL/qcl_array.hpp>

#include "../../common/environment.hpp"
#include "../../common/random_vectors.hpp"
#include "../../common/timer.hpp"


constexpr std::size_t particle_dimension = 3;
constexpr std::size_t num_runs = 10;

const std::size_t num_particles = 700000;
const std::size_t num_query_groups_xy = 128;
const std::size_t query_group_size_xy = 8;

constexpr std::size_t max_retrieved_particles = 8;

using tree_type = spatialcl::hilbert_bvh_sp3d_tree<particle_dimension>;
using type_system = tree_type::type_system;
using scalar = type_system::scalar;

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
void run_benchmark(const std::string& name,
                   const qcl::device_context_ptr& ctx,
                   const tree_type& tree,
                   const qcl::device_array<vector_type>& queries_min,
                   const qcl::device_array<vector_type>& queries_max,
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

  common::timer t;
  // Execute the query once before measuring to make sure
  // we do not take into account kernel compilation times.
  tree.execute_query(query_engine, query_handler);

  cl_int err = ctx->get_command_queue().finish();
  qcl::check_cl_error(err, "Error while executing range query");

  t.start();
  for (std::size_t run = 0; run < num_runs; ++run)
  {
    tree.execute_query(query_engine, query_handler);

    err = ctx->get_command_queue().finish();
    qcl::check_cl_error(err, "Error while executing range query");
  }
  double time = t.stop();

  std::size_t total_num_retrieved_particles = 0;
  std::vector<cl_uint> host_num_particles;
  num_results.read(host_num_particles);

  for(const cl_uint n: host_num_particles) 
    total_num_retrieved_particles += n;
  

  std::cout << "Benchmark " << name << " completed in "
            << time << "s => " << num_runs * queries_min.size() / time << " Queries/s"
            << " (retrieved: " << total_num_retrieved_particles << " particles)" << std::endl;
}

int main()
{
  common::environment env;
  qcl::device_context_ptr ctx = env.get_device_context();

  // Setup tree

  std::vector<particle_type> particles;
  common::random_vectors<scalar, particle_dimension> rnd;
  rnd(num_particles, particles);
  
  tree_type gpu_tree{ctx, particles};

  // Create queries
  std::size_t total_num_queries = 
    num_query_groups_xy * num_query_groups_xy * query_group_size_xy * query_group_size_xy;

  std::vector<vector_type> query_ranges_min;
  std::vector<vector_type> query_ranges_max;

  double query_stepsize = 1.0 / static_cast<double>(num_query_groups_xy * query_group_size_xy);
  scalar query_diameter =  static_cast<scalar>(3.0 * query_stepsize);

  for (std::size_t x = 0; x < num_query_groups_xy; ++x)
  {
    for (std::size_t y = 0; y < num_query_groups_xy; ++y)
    {
      for (std::size_t tile_x = 0; tile_x < query_group_size_xy; ++tile_x)
      {
        for (std::size_t tile_y = 0; tile_y < query_group_size_xy; ++tile_y)
        {
          std::size_t query_id_x = x * query_group_size_xy + tile_x;
          std::size_t query_id_y = y * query_group_size_xy + tile_y;

          scalar x_position = static_cast<scalar>(query_id_x * query_stepsize);
          scalar y_position = static_cast<scalar>(query_id_y * query_stepsize);

          vector_type query_min, query_max;

          query_min.s[0] = x_position;
          query_min.s[1] = y_position;
          query_min.s[2] = 0.0f;

          query_max.s[0] = x_position + query_diameter;
          query_max.s[1] = y_position + query_diameter;
          query_max.s[2] = query_diameter;

          query_ranges_min.push_back(query_min);
          query_ranges_max.push_back(query_max);
          
        }
      }
    }
  }

  qcl::device_array<vector_type> device_ranges_min{ctx, query_ranges_min};
  qcl::device_array<vector_type> device_ranges_max{ctx, query_ranges_max};
  qcl::device_array<particle_type> result_particles{ctx,
                                                  total_num_queries*max_retrieved_particles};
  qcl::device_array<cl_uint> result_num_retrieved_particles{ctx,
                                                            total_num_queries};

#define RUN_BENCHMARK(engine) \
  run_benchmark<engine>(BOOST_PP_STRINGIZE(engine), \
                        ctx, \
                        gpu_tree, \
                        device_ranges_min, \
                        device_ranges_max, \
                        result_particles, \
                        result_num_retrieved_particles)

  RUN_BENCHMARK(strict_dfs_range_engine);
  RUN_BENCHMARK(relaxed_dfs_range_engine);
  RUN_BENCHMARK(grouped_dfs_range_engine<16>);
  RUN_BENCHMARK(grouped_dfs_range_engine<32>);
  RUN_BENCHMARK(grouped_dfs_range_engine<64>);
  RUN_BENCHMARK(grouped_dfs_range_engine<128>);
  RUN_BENCHMARK(grouped_dfs_range_engine<256>);
  RUN_BENCHMARK(register_bfs_range_engine);

  return 0;
}
