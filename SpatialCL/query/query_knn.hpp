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

#ifndef QUERY_KNN_HPP
#define QUERY_KNN_HPP


#include "../configuration.hpp"
#include "../math/geometry.hpp"

namespace spatialcl {
namespace query {


template<class Type_descriptor, std::size_t K>
QCL_STANDALONE_MODULE(knn_query)
QCL_STANDALONE_SOURCE
(
  QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
  QCL_INCLUDE_MODULE(math::geometry<Type_descriptor>)
  QCL_IMPORT_CONSTANT(K)
  QCL_PREPROCESSOR(define,
    dfs_node_selection_criterion(selection_result_ptr,
                                 current_node_key_ptr,
                                 node_index,
                                 bbox_min_corner,
                                 bbox_max_corner)

      scalar dist2 = box_distance2(query_position,
                                   bbox_min_corner,
                                   bbox_max_corner);
      *selection_result_ptr = dist2 < *current_max_distance;

  )
  QCL_PREPROCESSOR(define,
    particle_selection_criterion(selection_result_ptr,
                                 particle_idx,
                                 current_particle)
  )
  QCL_PREPROCESSOR(define,
    node_select_handler(node_index,
                        bbox_min_corner,
                        bbox_max_corner)
  )
  QCL_PREPROCESSOR(define,
    particle_select_handler(particle_index,
                            particle)
  )
  QCL_PREPROCESSOR(define,
    node_discard_handler(current_node_key_ptr,
                         node_idx,
                         bbox_min_corner,
                         bbox_max_corner)
  )
  QCL_PREPROCESSOR(define,
    particle_discard_handler(particle_index,
                             particle)
  )
  R"(
    #define declare_full_query_parameter_set() \
       __global vector_type* query_positions, \
       __global particle_type* query_result, \
                     ulong num_queries
  )"
  QCL_PREPROCESSOR(define,
    declare_resumable_query_parameter_set()
  )
  QCL_PREPROCESSOR(define,
    at_query_init()

      scalar candidate_distances2 [K];

      for(int i = 0; i < K; ++i)
        candidate_distance2[i] = MAX_FLOAT;

      uint max_distance_idx = 0;
      scalar

      vector_type query_position = query_positions[tid];
  )
  QCL_PREPROCESSOR(define,
    at_query_resume()
  )
  QCL_PREPROCESSOR(define,
    at_query_pause()
  )
  QCL_PREPROCESSOR(define,
    at_query_exit()
  )
  QCL_PREPROCESSOR(define,
    get_num_parallel_queries()
      num_queries
  )

)



}
}


#endif
