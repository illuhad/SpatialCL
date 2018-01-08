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

#ifndef QUERY_HPP
#define QUERY_HPP

#include "query/query_engine_dfs.hpp"
#include "query/query_engine_bfs.hpp"

#include "query/query_knn.hpp"
#include "query/query_range.hpp"

namespace spatialcl {
namespace query {

template<class Type_descriptor, class Handler>
using strict_dfs_query_engine = query::engine::depth_first
  <
    Type_descriptor,
    Handler,
    engine::HIERARCHICAL_ITERATION_STRICT
  >;

template<class Type_descriptor, class Handler>
using relaxed_dfs_query_engine = query::engine::depth_first
  <
    Type_descriptor,
    Handler,
    engine::HIERARCHICAL_ITERATION_RELAXED
  >;

/************** Range Queries ***************************/

template<class Type_descriptor, std::size_t Max_retrieved_particles>
using strict_dfs_range_query_engine = strict_dfs_query_engine
  <
    Type_descriptor,
    box_range_query<Type_descriptor, Max_retrieved_particles>
  >;

template<class Type_descriptor, std::size_t Max_retrieved_particles>
using relaxed_dfs_range_query_engine = relaxed_dfs_query_engine
  <
    Type_descriptor,
    box_range_query<Type_descriptor, Max_retrieved_particles>
  >;


template<class Type_descriptor, std::size_t Max_retrieved_particles>
using register_bfs_range_query_engine =
  query::engine::register_breadth_first
  <
    Type_descriptor,
    box_range_query<Type_descriptor, Max_retrieved_particles>,
    Max_retrieved_particles
  >;

template<class Type_descriptor, std::size_t Max_retrieved_particles>
using default_range_query_engine = relaxed_dfs_range_query_engine
  <
    Type_descriptor,
    Max_retrieved_particles
  >;


/********** KNN Queries ********************************/

template<class Type_descriptor, std::size_t K>
using strict_dfs_knn_query_engine = strict_dfs_query_engine
  <
    Type_descriptor,
    knn_query<Type_descriptor, K>
  >;

template<class Type_descriptor, std::size_t K>
using relaxed_dfs_knn_query_engine = relaxed_dfs_query_engine
  <
    Type_descriptor,
    knn_query<Type_descriptor, K>
  >;

template<class Type_descriptor, std::size_t K>
using register_bfs_knn_query_engine =
  query::engine::register_breadth_first
  <
    Type_descriptor,
    knn_query<Type_descriptor, K>,
    K
  >;

template<class Type_descriptor, std::size_t K>
using default_knn_query_engine = register_bfs_knn_query_engine
  <
    Type_descriptor,
    K
  >;

}
}

#endif
