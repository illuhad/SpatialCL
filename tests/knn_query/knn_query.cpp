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

#include <boost/preprocessor/stringize.hpp>

#include <SpatialCL/tree.hpp>
#include <SpatialCL/query.hpp>

#include <QCL/qcl.hpp>
#include <QCL/qcl_array.hpp>

#include "../common/environment.hpp"
#include "../common/random_vectors.hpp"
#include "../common/verification_knn.hpp"

constexpr std::size_t particle_dimension = 3;

using tree_type = spatialcl::hilbert_bvh_sp3d_tree<particle_dimension>;
using type_system = tree_type::type_system;
using scalar = type_system::scalar;

const std::size_t num_particles = 32000;
const std::size_t num_queries = 2000;
constexpr std::size_t K = 8;

using particle_type = spatialcl::configuration<type_system>::particle_type;
using vector_type = spatialcl::configuration<type_system>::vector_type;
constexpr std::size_t dimension = type_system::dimension;

using strict_dfs_knn_engine =
  spatialcl::query::strict_dfs_knn_query_engine<type_system, K>;

using relaxed_dfs_knn_engine =
  spatialcl::query::relaxed_dfs_knn_query_engine<type_system, K>;

using register_bfs_knn_engine =
  spatialcl::query::register_bfs_knn_query_engine<type_system, K>;

template <std::size_t Group_size>
using grouped_dfs_knn_engine =
  spatialcl::query::grouped_dfs_knn_query_engine<type_system, K, Group_size>;

template<class Query_engine>
std::size_t execute_knn_query_test(const qcl::device_context_ptr& ctx,
                                   const tree_type& tree,
                                   const std::vector<vector_type>& host_queries,
                                   const qcl::device_array<vector_type>& queries,
                                   const std::vector<particle_type>& particles,
                                   qcl::device_array<particle_type>& result)
{
  // ToDo: Reset results

  Query_engine query_engine;
  typename Query_engine::handler_type query_handler {
    queries.get_buffer(),
    result.get_buffer(),
    queries.size()
  };

  std::cout << "Executing query..." << std::endl;
  // Executy query
  tree.execute_query(query_engine, query_handler);

  cl_int err = ctx->get_command_queue().finish();
  qcl::check_cl_error(err, "Error while executing KNN query");

  // Retrieve results
  std::vector<particle_type> host_results;
  result.read(host_results);

  std::cout << "Verifying results, please wait..." << std::endl;
  // Verify results
  common::verification::naive_cpu_knn_verifier<type_system, K> verifier{
    host_queries
  };

  return verifier(particles, host_results);
}

int main(int argc, char* argv[])
{
  common::environment env;
  qcl::device_context_ptr ctx = env.get_device_context();

  std::vector<particle_type> particles;
  // Create random 3D positions
  common::random_vectors<scalar, particle_dimension> rnd;
  rnd(num_particles, particles);

  tree_type gpu_tree{ctx, particles};

  std::vector<vector_type> query_points;
  rnd(num_queries, query_points);

  qcl::device_array<vector_type> queries{ctx, query_points};
  qcl::device_array<particle_type> result{ctx, K * queries.size()};

  std::size_t num_errors = 0;

  #define RUN_TEST(test_name) \
  num_errors = \
      execute_knn_query_test<test_name>(ctx,          \
                                        gpu_tree,     \
                                        query_points, \
                                        queries,      \
                                        particles,    \
                                        result);      \
  std::cout << BOOST_PP_STRINGIZE(test_name) <<" completed queries with " \
            << num_errors << " errors." << std::endl

  RUN_TEST(strict_dfs_knn_engine);
  RUN_TEST(relaxed_dfs_knn_engine);
  RUN_TEST(grouped_dfs_knn_engine<16>);
  RUN_TEST(grouped_dfs_knn_engine<32>);
  RUN_TEST(grouped_dfs_knn_engine<64>);
  RUN_TEST(register_bfs_knn_engine);

  return 0;
}
