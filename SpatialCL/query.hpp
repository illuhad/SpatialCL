#ifndef QUERY_HPP
#define QUERY_HPP

#include "query/query_engine_dfs.hpp"
#include "query/query_knn.hpp"
#include "query/query_range.hpp"

namespace spatialcl {
namespace query {

template<class Type_descriptor, class Handler>
using strict_dfs_query_engine = query::engine::depth_first_query
  <
    Type_descriptor,
    Handler,
    engine::HIERARCHICAL_ITERATION_STRICT
  >;

template<class Type_descriptor, class Handler>
using relaxed_dfs_query_engine = query::engine::depth_first_query
  <
    Type_descriptor,
    Handler,
    engine::HIERARCHICAL_ITERATION_RELAXED
  >;

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
using default_range_query_engine = relaxed_dfs_range_query_engine
  <
    Type_descriptor,
    Max_retrieved_particles
  >;

}
}

#endif
