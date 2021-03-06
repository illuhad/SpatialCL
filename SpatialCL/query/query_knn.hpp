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

#ifndef QUERY_KNN_HPP
#define QUERY_KNN_HPP


#include "../configuration.hpp"
#include "../math/geometry.hpp"

#include "query_base.hpp"

namespace spatialcl {
namespace query {


template<class Type_descriptor, std::size_t K>
class knn_query : public basic_query
{
public:
  QCL_MAKE_MODULE(knn_query)

  static_assert(K > 0, "K must be non-zero, or do you really want"
                       "to find the zero nearest neighbors?");

  knn_query(const cl::Buffer& query_points,
            const cl::Buffer& results,
            std::size_t num_queries)
    : _query_points{query_points},
      _results{results},
      _num_queries{num_queries}
  {
  }

  virtual void push_full_arguments(qcl::kernel_call& call) override
  {
    call.partial_argument_list(_query_points,
                               _results,
                               static_cast<cl_ulong>(_num_queries));
  }

  virtual std::size_t get_num_independent_queries() const override
  {
    return _num_queries;
  }

  virtual ~knn_query(){}

  static_assert(K > 0, "K must be non-zero");

private:
  cl::Buffer _query_points;
  cl::Buffer _results;
  std::size_t _num_queries;

  QCL_MAKE_SOURCE
  (
    QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
    QCL_INCLUDE_MODULE(math::geometry<Type_descriptor>)
    QCL_IMPORT_CONSTANT(K)

    QCL_RAW(
    void knn_update_max_distance(uint* max_distance_idx,
                                 scalar* candidate_distances2)
    {
      *max_distance_idx = 0;
      // Find new max distance index
      for(int i = 0; i < K; ++i)
        if(candidate_distances2[i] >=
           candidate_distances2[*max_distance_idx])
          *max_distance_idx = i;
    }

    void knn_add_candidate_particle(particle_type particle,
                                    scalar particle_dist2,
                                    uint* max_distance_idx,
                                    scalar* candidate_distances2,
                                    particle_type* candidates)
    {
      candidate_distances2[*max_distance_idx] = particle_dist2;
      candidates[*max_distance_idx] = particle;
      knn_update_max_distance(max_distance_idx, candidate_distances2);
    })
    // depth-first-serach for knn queries is currently
    // implemented rather inefficiently, because the DFS
    // query interface currently does not provide a means
    // to compare sibling nodes. This is however required
    // for efficient KNN queries to decide which path down
    // the tree is the better one.
    QCL_PREPROCESSOR(define,
      dfs_node_selector(selection_result_ptr,
                        current_node_key_ptr,
                        node_index,
                        bbox_min_corner,
                        bbox_max_corner)
      {
        scalar dist2 = 0.0f;
        if(!box_contains_particle(CLIP_TO_VECTOR(bbox_min_corner),
                                  CLIP_TO_VECTOR(bbox_max_corner),
                                  query_position))
        {
          dist2 = box_distance2(query_position,
                                CLIP_TO_VECTOR(bbox_min_corner),
                                CLIP_TO_VECTOR(bbox_max_corner));
        }
        
        *selection_result_ptr =
            dist2 < candidate_distances2[max_distance_idx];
      }

    )
    QCL_PREPROCESSOR(define,
      dfs_particle_processor(selection_result_ptr,
                             particle_idx,
                             current_particle)
      {
        vector_type delta = PARTICLE_POSITION(current_particle)
                          - query_position;
        scalar dist2 = VECTOR_NORM2(delta);

        *selection_result_ptr =
               dist2 < candidate_distances2[max_distance_idx];

        if(*selection_result_ptr)
        {
          knn_add_candidate_particle(current_particle,
                                     dist2,
                                     &max_distance_idx,
                                     candidate_distances2,
                                     candidates);
        }
      }
    )
    QCL_PREPROCESSOR(define,
      dfs_unique_node_discard_event(node_idx,
                                    current_bbox_min_corner,
                                    current_bbox_max_corner)
    )
    R"(
      #define declare_full_query_parameter_set() \
         __global vector_type* query_positions, \
         __global particle_type* query_result, \
         ulong num_queries
    )"
    QCL_PREPROCESSOR(define,
      at_query_init()

        scalar candidate_distances2 [K];
        particle_type candidates    [K];

        for(int i = 0; i < K; ++i)
          candidate_distances2[i] = FLT_MAX;

        uint max_distance_idx = 0;

        vector_type query_position;
        if(get_query_id() < num_queries)
          query_position = query_positions[get_query_id()];
    )
    QCL_PREPROCESSOR(define,
      at_query_exit()
        if(get_query_id() < num_queries)
        {
          for(int i = 0; i < K; ++i)
            query_result[get_query_id()*K + i] = candidates[i];
        }
    )
    QCL_PREPROCESSOR(define,
      get_num_queries()
        num_queries
  )

)
};


}
}


#endif
