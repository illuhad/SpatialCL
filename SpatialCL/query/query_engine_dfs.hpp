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

#ifndef QUERY_ENGINE_DFS_HPP
#define QUERY_ENGINE_DFS_HPP

#include "../configuration.hpp"
#include "../tree/binary_tree.hpp"

namespace spatialcl {
namespace query {
namespace engine {

enum depth_first_iteration_strategy
{
  HIERARCHICAL_ITERATION_STRICT = 0,
  HIERARCHICAL_ITERATION_RELAXED = 1
};

template<class Type_descriptor,
         class Handler_module,
         depth_first_iteration_strategy Iteration_strategy>
class depth_first_query
{
public:
  static constexpr std::size_t group_size = 256;

  using handler_type = Handler_module;

  cl_int operator()(const qcl::device_context_ptr& ctx,
                    const cl::Buffer& particles,
                    const cl::Buffer& bbox_min_corner,
                    const cl::Buffer& bbox_max_corner,
                    std::size_t num_particles,
                    std::size_t effective_num_particles,
                    std::size_t effective_num_levels,
                    Handler_module& handler,
                    cl::Event* evt = nullptr)
  {
    qcl::kernel_call call = query(ctx,
                                  cl::NDRange{handler.get_num_independent_queries()},
                                  cl::NDRange{group_size},
                                  evt);

    call.partial_argument_list(particles,
                               bbox_min_corner,
                               bbox_max_corner,
                               static_cast<cl_ulong>(num_particles),
                               static_cast<cl_ulong>(effective_num_particles),
                               static_cast<cl_ulong>(effective_num_levels));

    handler.push_full_arguments(call);
    return call.enqueue_kernel();
  }

  QCL_MAKE_MODULE(depth_first_query)
  QCL_ENTRYPOINT(query)
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
    QCL_INCLUDE_MODULE(Handler_module)
    QCL_INCLUDE_MODULE(binary_tree)
    QCL_IMPORT_CONSTANT(Iteration_strategy)
    QCL_RAW(
        ulong load_node(binary_tree_key_t* node,
                       __global vector_type* bbox_min_corner,
                       __global vector_type* bbox_max_corner,
                       ulong effective_num_levels,
                       ulong effective_num_particles,
                       vector_type* min_corner_out,
                       vector_type* max_corner_out)
        {
          ulong idx = binary_tree_key_encode_global_id(node,effective_num_levels);
          idx -= effective_num_particles;

          *min_corner_out = bbox_min_corner[idx];
          *max_corner_out = bbox_max_corner[idx];

          return idx;
        }

        ulong load_particle(binary_tree_key_t* node,
                       __global particle_type* particles,
                       ulong effective_num_levels,
                       ulong effective_num_particles,
                       particle_type* particle_out)
        {

          // Since particles are at the lowest level, we know that for them
          // the index equals the local node id
          ulong idx = node->local_node_id;
          *particle_out = particles[idx];
          return idx;
        }

        binary_tree_key_t find_first_left_parent(binary_tree_key_t* node)
        {
          binary_tree_key_t result = *node;
          while(binary_tree_is_right_child(&result))
            result = binary_tree_get_parent(&result);
          return result;
        }
      )
      R"(
      #if Iteration_strategy == 0
        // Strict iteration
        #define NEXT_PARENT(node) find_first_left_parent(&node)
      #elif Iteration_strategy == 1
        // Relaxed iteration
        #define NEXT_PARENT(node) binary_tree_get_parent(&node)
      #else
        #error Invalid iteration strategy
      #endif
      )"
      QCL_PREPROCESSOR(define, get_query_id() tid)
      QCL_PREPROCESSOR(define,
        QUERY_NODE_LEVEL(bbox_min_corner,
                         bbox_max_corner,
                         effective_num_particles,
                         effective_num_levels,
                         current_node,
                         num_covered_particles)
        {
          vector_type current_bbox_min_corner;
          vector_type current_bbox_max_corner;

          ulong node_idx = load_node(&current_node,
                                     bbox_min_corner,
                                     bbox_max_corner,
                                     effective_num_levels,
                                     effective_num_particles,
                                     &current_bbox_min_corner,
                                     &current_bbox_max_corner);

          int node_selected = 0;
          dfs_node_selection_criterion(&node_selected,
                                       &current_node,
                                       node_idx,
                                       current_bbox_min_corner,
                                       current_bbox_max_corner);
          if(node_selected)
          {
            node_select_handler(node_idx,
                                current_bbox_min_corner,
                                current_bbox_max_corner);

            current_node = binary_tree_get_children_begin(&current_node);
          }
          else
          {
            node_discard_handler(&current_node,
                                 node_idx,
                                 current_bbox_min_corner,
                                 current_bbox_max_corner);
            num_covered_particles += BT_LEAVES_PER_NODE(current_node.level,
                                                        effective_num_levels);

            if(binary_tree_is_right_child(&current_node))
            {
              // if we are at a right child node, go up to the parent's
              // sibling...
              current_node = NEXT_PARENT(current_node);
              current_node.local_node_id++;
            }
            else
              // otherwise, first investigate the sibling
              current_node.local_node_id++;
          }
        }
      )
      QCL_PREPROCESSOR(define,
        QUERY_PARTICLE_LEVEL(particles,
                             effective_num_particles,
                             effective_num_levels,
                             current_node,
                             num_covered_particles)
        {
          particle_type current_particle;

          ulong particle_idx = load_particle(&current_node,
                                             particles,
                                             effective_num_levels,
                                             effective_num_particles,
                                             &current_particle);

          int particle_selected = 0;
          particle_selection_criterion(&particle_selected,
                                       particle_idx,
                                       current_particle);
          if(particle_selected)
          {
            particle_select_handler(particle_idx,
                                    current_particle);
            current_node.local_node_id++;
          }
          else
          {
            particle_discard_handler(particle_idx,
                                     current_particle);

            if(binary_tree_is_right_child(&current_node))
            {
              // if we are at a right child node, go up to the parent's
              // sibling...
              current_node = NEXT_PARENT(current_node);
              current_node.local_node_id++;
            }
            else
              // otherwise, first investigate the sibling
              current_node.local_node_id++;
          }
          num_covered_particles++;
        }
      )
      QCL_RAW(
        __kernel void query(__global particle_type* particles,
                            __global vector_type* bbox_min_corner,
                            __global vector_type* bbox_max_corner,
                            ulong num_particles,
                            ulong effective_num_particles,
                            ulong effective_num_levels,
                            declare_full_query_parameter_set())
        {

          for(size_t tid = get_global_id(0);
              tid < get_num_queries();
              tid += get_global_size(0))
          {
            at_query_init();

            binary_tree_key_t current_node;
            current_node.level = 0;
            current_node.local_node_id = 0;

            for(ulong num_covered_particles = 0;
                num_covered_particles < num_particles;)
            {
              int particle_level_reached = (current_node.level == effective_num_levels-1);

              if(particle_level_reached)
              {
                QUERY_PARTICLE_LEVEL(particles,
                                     effective_num_particles,
                                     effective_num_levels,
                                     current_node,
                                     num_covered_particles);
              }
              else
              {
                QUERY_NODE_LEVEL(bbox_min_corner,
                                 bbox_max_corner,
                                 effective_num_particles,
                                 effective_num_levels,
                                 current_node,
                                 num_covered_particles);
              }
            }

            at_query_exit();
          }
        }
      )
  )
};

}
}
}

#endif
