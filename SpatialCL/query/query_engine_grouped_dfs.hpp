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

#ifndef QUERY_ENGINE_GROUPED_DFS_HPP
#define QUERY_ENGINE_GROUPED_DFS_HPP

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>

#include "../configuration.hpp"
#include "../tree/binary_tree.hpp"
#include "../cl_utils.hpp"


namespace spatialcl {
namespace query {
namespace engine {

template<class Type_descriptor,
         class Handler_module,
         std::size_t group_size = 64,
         std::size_t node_batch_load_size = 2,
         std::size_t particle_batch_load_size = 4,
         std::size_t group_coherence_size = 32>
class grouped_depth_first
{
public:
  QCL_MAKE_MODULE(grouped_depth_first)

  static_assert(group_size % 2 == 0, "The group size must be even.");
  static_assert(node_batch_load_size <= group_size, "The number of batch loaded nodes must be "
                                                    "<= the work group size");
  static_assert(node_batch_load_size % 2 == 0, "The number of batch loaded nodes must be even");
  static_assert(particle_batch_load_size % 2 == 0, "The number of batch loaded particles must be even");
  static_assert(particle_batch_load_size % node_batch_load_size == 0,
                "The number of batch loaded particles must be a multiple of the number of "
                "batch loaded nodes");

  static_assert(particle_batch_load_size <= group_size && 
                node_batch_load_size <= group_size,
                "The number of batch loaded particles and nodes cannot be larger than"
                " the work group size");
  using handler_type = Handler_module;
  using particle_type = 
      typename configuration<Type_descriptor>::particle_type;
  using vector_type =
      typename configuration<Type_descriptor>::vector_type;

  static_assert(sizeof(particle_type) >= sizeof(vector_type),
                "particle_type should always be larger/equal than vector_type");

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

    std::size_t local_mem_size = 
          std::max(particle_batch_load_size * sizeof(particle_type), 
                   node_batch_load_size * 2 * sizeof(vector_type)) + sizeof(cl_int);

    call.partial_argument_list(particles,
                               bbox_min_corner,
                               bbox_max_corner,
                               static_cast<cl_ulong>(num_particles),
                               static_cast<cl_ulong>(effective_num_particles),
                               static_cast<cl_ulong>(effective_num_levels),
                               qcl::local_memory<cl_uchar>{local_mem_size});

    handler.push_full_arguments(call);
    return call.enqueue_kernel();
  }

private:
  QCL_ENTRYPOINT(query)
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
    QCL_INCLUDE_MODULE(Handler_module)
    QCL_INCLUDE_MODULE(binary_tree)
    QCL_IMPORT_CONSTANT(group_size)
    QCL_IMPORT_CONSTANT(group_coherence_size)
    QCL_IMPORT_CONSTANT(node_batch_load_size)
    QCL_IMPORT_CONSTANT(particle_batch_load_size)
    R"(
      #if group_size <= barrier_coherence_size
       #define fast_barrier(flags)
      #else
       #define fast_barrier(flags) barrier(flags)
      #endif
    )"
    QCL_RAW(
      ulong get_node_index(binary_tree_key_t* node,
                           ulong effective_num_levels,
                           ulong effective_num_particles)
      {
        return binary_tree_key_encode_global_id(node,effective_num_levels)
               - effective_num_particles;
      }
    )
    QCL_PREPROCESSOR(define, get_query_id() tid)
    QCL_PREPROCESSOR(define,
      QUERY_PARTICLE_LEVEL(particles,
                           effective_num_particles,
                           num_particles,
                           effective_num_levels,
                           group_start_node,
                           num_covered_particles,
                           cache)
      {
        size_t lid = get_local_id(0);

        __local particle_type* particle_cache = cache;
        __local int* some_particles_selected = 
                   (__local int *)(particle_cache + particle_batch_load_size);

        ulong particle_idx_begin = group_start_node.local_node_id;
        const int num_available_particles = min((int)particle_batch_load_size, 
                                                (int)(num_particles-num_covered_particles));
        
        // Load particles collectively into the cache
        if(lid < num_available_particles)
          particle_cache[lid] = particles[particle_idx_begin + lid];
        *some_particles_selected = 0;

        fast_barrier(CLK_LOCAL_MEM_FENCE);

        // For each query, iterate over the particles in the
        // cache and check if particles should be selected.
        if(tid < get_num_queries())
        {
          int any_particle_selected = 0;
          for(int i = 0; i < num_available_particles; ++i)
          {
            int particle_selected = 0;
            dfs_particle_processor(&particle_selected,
                                   (particle_idx_begin + i),
                                   particle_cache[i]);

            any_particle_selected |= particle_selected;
          }

          // If the query has decided to select some particle,
          // mark this in the level state
          if(any_particle_selected)
            *some_particles_selected = 1;
        }
        fast_barrier(CLK_LOCAL_MEM_FENCE);

        // Update the local_node_id - from now on, it contains
        // the position where the next processed block would start
        // in case we remain at the particle level.
        group_start_node.local_node_id += num_available_particles;

        // We can only go up from the particle level, if we have processed 2*node_batch_load_size
        // particles. This guarantees that, when going up, we do not arrive
        // at the same parent nodes that we have already processed previously.
        // This alignment with the parent node happens if the start position of
        // the next particle group is a multiple of the node_batch_load_size.
        int aligned_with_parent = (group_start_node.local_node_id % node_batch_load_size == 0);
        int go_up = !(*some_particles_selected) && aligned_with_parent;
        fast_barrier(CLK_LOCAL_MEM_FENCE);

        if(go_up)
        {
          // No query/work item was interested in any particles from
          // this region, so the entire work group should move up in the tree
          group_start_node.level--;
          group_start_node.local_node_id >>= 1;
        }

        num_covered_particles += num_available_particles;
      }
    )
    QCL_PREPROCESSOR(define,
      QUERY_NODE_LEVEL(bbox_min_corner,
                       bbox_max_corner,
                       effective_num_particles,
                       num_particles,
                       effective_num_levels,
                       group_start_node,
                       num_covered_particles,
                       cache)
      {
         const size_t lid = get_local_id(0);

        __local vector_type* bbox_min_corner_cache =
                       (__local vector_type*)cache;
        __local vector_type* bbox_max_corner_cache =
                       bbox_min_corner_cache + node_batch_load_size;
        __local int* some_nodes_selected =
                       (__local int*)(bbox_max_corner_cache + node_batch_load_size);

        ulong node_idx_begin = get_node_index(&group_start_node,
                                              effective_num_levels,
                                              effective_num_particles);
        int num_available_nodes = 
          min((int)(node_batch_load_size),
              (int)(binary_tree_get_num_populated_nodes(group_start_node.level, 
                                                        effective_num_levels, 
                                                        num_particles)
                                                    - group_start_node.local_node_id));

        // Load nodes collectively into the cache
        if(lid < num_available_nodes)
        {
          bbox_min_corner_cache[lid] = bbox_min_corner[node_idx_begin + lid];
          bbox_max_corner_cache[lid] = bbox_max_corner[node_idx_begin + lid];
        }
        *some_nodes_selected = 0;
        fast_barrier(CLK_LOCAL_MEM_FENCE);

        // For each query, iterate over the nodes in the
        // cache and check if nodes should be selected.
        if(tid < get_num_queries())
        {
          
          for(int i = 0; i < num_available_nodes; ++i)
          {
            int node_selected = 0;
            binary_tree_key_t current_node = group_start_node;
            current_node.local_node_id += i;

            dfs_node_selector(&node_selected,
                              &current_node,
                              (node_idx_begin + i),
                              bbox_min_corner_cache[i],
                              bbox_max_corner_cache[i]);


            // If the query has decided to select some nodes,
            // mark this work item as having selected nodes
            // in the some_nodes_selected local memory area
            if(node_selected)
              *some_nodes_selected = 1;
          }  
        }
        fast_barrier(CLK_LOCAL_MEM_FENCE);

        // We must investigate deeper levels, if at least
        // one query wants to investigate deeper levels.
        // For this, we need a reduction
        int go_deeper = *some_nodes_selected;
        fast_barrier(CLK_LOCAL_MEM_FENCE);

        if(go_deeper)
        {
          // Some queries/work items want to go to the next deeper level
          group_start_node = binary_tree_get_children_begin(&group_start_node);
        }
        else
        {
          // No query/work item wants to deeper, so the entire work group
          // should move up in the tree

          // Trigger the node discard event for all nodes
          if(tid < get_num_queries())
          {
            for(int i = 0; i < num_available_nodes; ++i)
              dfs_unique_node_discard_event(node_idx_begin + i,
                                            bbox_min_corner_cache[i],
                                            bbox_max_corner_cache[i]);
          }

          // Advance the number of covered particles
          num_covered_particles +=
                         num_available_nodes * BT_LEAVES_PER_NODE(group_start_node.level,
                                                                  effective_num_levels);

          // If we already have completed two segments on this level,
          // go up, else remain on this level. This ensures that no
          // redundant work is done with respect to the upper level from
          // where we come.
          // In both cases, we first need to increment the local node id
          int odd_segment = (group_start_node.local_node_id / node_batch_load_size) & 1;

          group_start_node.local_node_id += num_available_nodes;
          if(odd_segment)
          {
            group_start_node.level--;
            group_start_node.local_node_id >>= 1;
          }
        }
      }
    )
    QCL_RAW(

      __kernel void query(__global particle_type* particles,
                          __global vector_type* bbox_min_corner,
                          __global vector_type* bbox_max_corner,
                          ulong num_particles,
                          ulong effective_num_particles,
                          ulong effective_num_levels,
                          // We pass the local memory as particle_type
                          // because particle_type is the largest of the cached
                          // data types (the others being int and vector_type).
                          // This guarantees a correct alignment.
                          __local particle_type* cache,
                          declare_full_query_parameter_set())
      {
        size_t tid = get_global_id(0);

        at_query_init();

        binary_tree_key_t group_start_node;
        group_start_node.level = 0;
        group_start_node.local_node_id = 0;

        for(ulong num_covered_particles = 0;
            num_covered_particles < num_particles;)
        {
          if (group_start_node.level == effective_num_levels - 1)
          {
            QUERY_PARTICLE_LEVEL(particles,
                                 effective_num_particles,
                                 num_particles,
                                 effective_num_levels,
                                 group_start_node,
                                 num_covered_particles,
                                 cache);
          }
          else
          {
            QUERY_NODE_LEVEL(bbox_min_corner,
                             bbox_max_corner,
                             effective_num_particles,
                             num_particles,
                             effective_num_levels,
                             group_start_node,
                             num_covered_particles,
                             cache);
          }
        }
        at_query_exit();
      }

    )
  )
};

}
}
}

#endif
