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

#ifndef QUERY_RANGE_HPP
#define QUERY_RANGE_HPP

#include "../configuration.hpp"
#include "../math/geometry.hpp"

#include "query_base.hpp"

namespace spatialcl {
namespace query {

template<class Type_descriptor,
         std::size_t Max_retrieved_particles>
class box_range_query : public basic_query
{
public:
  box_range_query(const cl::Buffer& query_ranges_min,
                  const cl::Buffer& query_ranges_max,
                  const cl::Buffer& result_retrieved_particles,
                  const cl::Buffer& result_num_retrieved_particles,
                  std::size_t num_queries)
    : _query_ranges_min{query_ranges_min},
      _query_ranges_max{query_ranges_max},
      _result{result_retrieved_particles},
      _num_selected_particles{result_num_retrieved_particles},
      _num_queries{num_queries}
  {
    assert(num_queries > 0);
  }

  virtual void push_full_arguments(qcl::kernel_call& call) override
  {
    call.partial_argument_list(_query_ranges_min,
                               _query_ranges_max,
                               _result,
                               _num_selected_particles,
                               static_cast<cl_ulong>(_num_queries));
  }

  virtual std::size_t get_num_independent_queries() const override
  {
    return _num_queries;
  }

  virtual ~box_range_query(){}

private:
  cl::Buffer _query_ranges_min;
  cl::Buffer _query_ranges_max;
  cl::Buffer _result;
  cl::Buffer _num_selected_particles;
  std::size_t _num_queries;
public:
  QCL_MAKE_MODULE(box_range_query)
  QCL_MAKE_SOURCE
  (
    QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
    QCL_INCLUDE_MODULE(math::geometry<Type_descriptor>)
    QCL_IMPORT_CONSTANT(Max_retrieved_particles)
    QCL_PREPROCESSOR(define,
      dfs_node_selector(selection_result_ptr,
                        current_node_key_ptr,
                        node_index,
                        bbox_min_corner,
                        bbox_max_corner)
        *selection_result_ptr = box_box_intersection(
                                      bbox_min_corner,
                                      bbox_max_corner,
                                      query_range_min,
                                      query_range_max);
    )
    QCL_PREPROCESSOR(define,
      dfs_particle_processor(selection_result_ptr,
                             particle_idx,
                             current_particle)
      {
        *selection_result_ptr = box_contains_particle(query_range_min,
                                                      query_range_max,
                                                      current_particle);
        if(*selection_result_ptr)
        {
          if(num_selected_particles < Max_retrieved_particles)
          {
            ulong result_pos = get_query_id()*Max_retrieved_particles
                             + num_selected_particles;
            query_result[result_pos] = current_particle;
            ++num_selected_particles;
          }
        }
      }
    )
    QCL_PREPROCESSOR(define,
      bfs_node_selector(max_selectable_nodes,
                        num_available_nodes)
      {
        for(uint k = 0; k < num_available_nodes; ++k)
        {
          bfs_load_node(k);

          vector_type bbox_min = bfs_get_node_min_corner();
          vector_type bbox_max = bfs_get_node_max_corner();

          if(box_box_intersection(bbox_min, bbox_max,
                                  query_range_min, query_range_max))
            bfs_select(k);
        }
      }
    )
    QCL_PREPROCESSOR(define,
      bfs_particle_processor(num_available_particles)
      {
        for(uint k = 0; k < num_available_particles; ++k)
        {
          particle_type p = bfs_load_particle(k);

          if(box_contains_particle(query_range_min,
                                   query_range_max,
                                   p)
             && (num_selected_particles < Max_retrieved_particles))
          {
            ulong result_pos = get_query_id()*Max_retrieved_particles
                             + num_selected_particles;
            query_result[result_pos] = p;
            ++num_selected_particles;
          }
        }
      }
    )
    R"(
      #define declare_full_query_parameter_set() \
        __global vector_type* query_ranges_min, \
        __global vector_type* query_ranges_max, \
        __global particle_type* query_result, \
        __global uint* num_retrieved_particles, \
        ulong num_queries
    )"
    QCL_PREPROCESSOR(define,
      declare_resumable_query_parameter_set()
    )
    QCL_PREPROCESSOR(define,
      at_query_init()
        vector_type query_range_min = query_ranges_min[get_query_id()];
        vector_type query_range_max = query_ranges_max[get_query_id()];
        uint num_selected_particles = 0;
        num_retrieved_particles[get_query_id()] = 0;
    )
    QCL_PREPROCESSOR(define,
      at_query_resume()
    )
    QCL_PREPROCESSOR(define,
      at_query_pause()
    )
    QCL_PREPROCESSOR(define,
      at_query_exit()
        num_retrieved_particles[get_query_id()] = num_selected_particles;
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
