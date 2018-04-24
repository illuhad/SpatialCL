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
#include "../binary_utils.hpp"

namespace spatialcl {
namespace query {
namespace engine {

/// The grouped depth first query engine is a depth-first query engine
/// that can be faster than the relaxed and strict depth-first engines
/// if 
///    * Adjacent queries in the query array follow a similar path through
///      the tree
///    * A lot of time is spent at the particle level.
/// This is for example the case for ordered range queries over large ranges.
/// Conceptually, this engine works similar to the \c relaxed_dfs_query_engine,
/// with the difference that an entire group of work items moves together to the tree.
/// This group can be an entire work group, or it can be a portion of a work group that
/// we call subgroup.
/// The common movement means that performance will degrade greatly if work items of
/// the same work group want to access very different parts of the tree.
/// At the node levels, a number of nodes will be loaded collectively into local
/// memory where they are processed by the entire subgroup.
/// Similarly, at the particle level particles are loaded collectively into local memory
/// where they are processed by the entire subgroup.
/// This means that the algorithm locally converges to the optimal brute-force
/// local memory based algorithm if enough time is spent at the particle level.
///
/// This query engine satisfies the DFS query engine interface concept.
/// \tparam Tree_type The tree type
/// \tparam Handler_module The query handler. Must satisfy the DFS concept.
/// \tparam group_size The work group size
/// \tparam subgroup_size The number of adjacent work items that share data
/// in the node and particle cache. All work items from one subgroup will exhibit the
/// exact same movement through the tree. The subgroup size must be <= the work group size.
/// If the subgroup size is smaller than the group size, then it must be
/// smaller than the warp/wavefront size (= \c group_coherence_size) due to synchronization
/// details. If it equals the group size, it can be larger than one warp or wavefront.
/// \tparam node_batch_load_size How many consecutive nodes are loaded collectively into
/// local memory by one subgroup.
/// \tparam particle_batch_load_size How many consecutive particles are loaded collectively 
/// into local memory by one subgroup.
/// \tparam group_coherence_size Number of consecutive work items in a work group
/// that are always synchronized with respect to each other. The algorithm will
/// remove explicit synchronization (e.g. barrier()) calls if the synchronization
/// is among that many work items. On nvidia GPUs, this should equal the warp size
/// (typically 32), on AMD GPUs it should equal the wavefront size (typically 64).
template<class Tree_type,
         class Handler_module,
         std::size_t group_size = 64,
         std::size_t subgroup_size = 8,
         std::size_t node_batch_load_size = 8,
         std::size_t particle_batch_load_size = 8,
         std::size_t group_coherence_size = 32,
         std::size_t vertical_level_stride_size =
            spatialcl::utils::binary::small_binary_logarithm<node_batch_load_size>::value
         >
class grouped_depth_first
{
public:
  QCL_MAKE_MODULE(grouped_depth_first)

  using type_system = typename Tree_type::type_system;
  using handler_type = Handler_module;
  using particle_type = 
      typename configuration<type_system>::particle_type;
  using vector_type =
      typename configuration<type_system>::vector_type;

  static_assert(group_size > 0, "group size must be > 0");
  static_assert(node_batch_load_size > 0, "node_batch_load_size must be > 0");
  static_assert(particle_batch_load_size > 0, "particle_batch_load_size must be > 0");
  static_assert(vertical_level_stride_size > 0, "vertical_level_stride_size must be > 0");
  static_assert(group_coherence_size > 0, "group coherence size must be > 0");

  static_assert(spatialcl::utils::binary::is_small_power2<group_size>::value, 
               "The group size must be a power of two.");
  static_assert(spatialcl::utils::binary::is_small_power2<node_batch_load_size>::value, 
               "The number of batch loaded nodes must be a power of two");
  static_assert(spatialcl::utils::binary::is_small_power2<particle_batch_load_size>::value,
               "The number of batch loaded particles must be a power of two");

  static_assert(particle_batch_load_size <= group_size && 
                node_batch_load_size <= group_size,
                "The number of batch loaded particles and nodes cannot be larger than"
                " the work group size");

  static_assert(particle_batch_load_size >= node_batch_load_size,
                "The number of batch loaded particles cannot be smaller than "
                "the number of batch loaded nodes.");
  static_assert(sizeof(particle_type) >= sizeof(vector_type),
                "particle_type cannot be smaller than vector_type, since "
                "this would break alignment on the device");
  static_assert(sizeof(particle_type) % sizeof(vector_type) == 0,
                "The size of particle_type must be a multiple of the size"
                " of vector_type");

  static_assert(vertical_level_stride_size <= 
                spatialcl::utils::binary::small_binary_logarithm<node_batch_load_size>::value,
                "The vertical level stride cannot be larger than log2(node_batch_load_size)");

  static_assert(subgroup_size <= group_size,
                "The subgroup size cannot be larger than the group size");
  static_assert(subgroup_size <= group_coherence_size || subgroup_size == group_size,
                "subgroup size must either be <= the warp/wavefront size, or"
                " the subgroup size must equal the work group size");
  static_assert(group_coherence_size % subgroup_size == 0 || subgroup_size == group_size,
                "warp/wavefront size must be a multiple of the subgroup size, "
                "if the subgroup size does not equal the work group size");
  static_assert(group_size % subgroup_size == 0,
                "group size must be multiple of the subgroup size");

  static_assert(subgroup_size >= node_batch_load_size &&
                subgroup_size >= particle_batch_load_size,
                "Currently, the subgroup size cannot be smaller than the batch load size");

  /// Execute query
  cl_int operator()(const Tree_type& tree,
                    Handler_module& handler,
                    cl::Event* evt = nullptr)
  {
    return this->run(tree.get_device_context(),
                     tree.get_sorted_particles(),
                     tree.get_node_values0(),
                     tree.get_node_values1(),
                     tree.get_num_particles(),
                     tree.get_effective_num_particles(),
                     tree.get_effective_num_levels(),
                     handler,
                     evt);
  }


private:
  cl_int run(const qcl::device_context_ptr& ctx,
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

  // In C++11, std::max is not constexpr (fixed in C++14).
  // We use the following workaround:
  template<class T>
  static constexpr T static_max(T a, T b)
  { return (a > b) ? a : b; }

  static constexpr std::size_t num_subgroups = group_size / subgroup_size;
  // Number of particle_type elements in the cache
  static constexpr std::size_t subgroup_cache_size =
            // We need the maximum of the space needed at particle or node level.
            // The +sizeof(particle_type)-1 rounds up the integer division to ensure
            // that there is always enough space allocated.
            (static_max(particle_batch_load_size * sizeof(particle_type),
                        node_batch_load_size * 2 * sizeof(vector_type))+sizeof(particle_type)-1)
            / sizeof(particle_type);
  // The amount of cache memory in units of particles
  static constexpr std::size_t total_cache_size = subgroup_cache_size * num_subgroups;

  QCL_ENTRYPOINT(query)
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(configuration<type_system>)
    QCL_INCLUDE_MODULE(Handler_module)
    QCL_INCLUDE_MODULE(binary_tree)
    QCL_IMPORT_CONSTANT(group_size)
    QCL_IMPORT_CONSTANT(group_coherence_size)
    QCL_IMPORT_CONSTANT(node_batch_load_size)
    QCL_IMPORT_CONSTANT(particle_batch_load_size)
    QCL_IMPORT_CONSTANT(vertical_level_stride_size)
    QCL_IMPORT_CONSTANT(subgroup_size)
    QCL_IMPORT_CONSTANT(num_subgroups)
    QCL_IMPORT_CONSTANT(total_cache_size)
    QCL_IMPORT_CONSTANT(subgroup_cache_size)
    R"(
      #if subgroup_size < group_size
        // If we use multiple subgroups, we don't need barriers because
        // subgroups are not allowed to span several warps/wavefronts.
        #define fast_barrier(flags)
      #elif group_size <= group_coherence_size
        // Without subgroups, we don't need full synchronization
        // if the group size is smaller than one warp/wavefront
        #define fast_barrier(flags)
      #else
        // Without subgroups and group sizes larger
        // than a warp, we need full-fledged barriers
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

      int subgroup_node_idx_min(volatile __local int* subgroup_mem,
                                const size_t subgroup_lid)
      {
        for(int i = subgroup_size/2; i > 0; i >>= 1)
        {
          if(subgroup_lid < i)
            subgroup_mem[subgroup_lid] = min(subgroup_mem[subgroup_lid  ],
                                             subgroup_mem[subgroup_lid+i]);
          fast_barrier(CLK_LOCAL_MEM_FENCE);
        }
        int result = subgroup_mem[0];
        fast_barrier(CLK_LOCAL_MEM_FENCE);
        return result;
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
                           subgroup_lid,
                           subgroup_selection_map,
                           subgroup_particle_cache)
      {
        ulong particle_idx_begin = group_start_node.local_node_id;
        const int num_available_particles = min((int)particle_batch_load_size, 
                                                (int)(num_particles-num_covered_particles));
        
        // Load particles collectively into the cache
        if(subgroup_lid < num_available_particles)
          subgroup_particle_cache[subgroup_lid] =
                         particles[particle_idx_begin + subgroup_lid];

        //*subgroup_selection_map = 0;
        subgroup_selection_map[subgroup_lid] = num_available_particles;
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
                                   subgroup_particle_cache[i]);

            any_particle_selected |= particle_selected;
          }

          // If the query has decided to select some particle,
          // mark this in the level state
          if(any_particle_selected)
            //*subgroup_selection_map = 1;
            subgroup_selection_map[subgroup_lid] = -1;
        }
        fast_barrier(CLK_LOCAL_MEM_FENCE);

        // Update the local_node_id - from now on, it contains
        // the position where the next processed block would start
        // in case we remain at the particle level.
        group_start_node.local_node_id += num_available_particles;

        //int go_up = !(*subgroup_selection_map);
        int go_up = (subgroup_node_idx_min(subgroup_selection_map, subgroup_lid)!=-1);
        fast_barrier(CLK_LOCAL_MEM_FENCE);

        if(go_up)
        {
          // No query/work item was interested in any particles from
          // this region, so the entire work group should move up in the tree
          //group_start_node.level--;
          //group_start_node.local_node_id >>= 1;
          group_start_node.level -= vertical_level_stride_size;
          group_start_node.local_node_id >>= vertical_level_stride_size;
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
                       subgroup_lid,
                       subgroup_first_selected_nodes,
                       subgroup_cache)
      {

        __local vector_type* const bbox_min_corner_cache =
                       (__local vector_type*)subgroup_cache;
        __local vector_type* const bbox_max_corner_cache =
                       bbox_min_corner_cache + node_batch_load_size;


        const ulong node_idx_begin = get_node_index(&group_start_node,
                                                    effective_num_levels,
                                                    effective_num_particles);
        
        // The number of available nodes is either the distance to the next
        // aligned node or (if we are at the last segment before the data ends)
        // the distance to the end of the data array
        const int num_available_nodes =
            min((int)(node_batch_load_size - node_idx_begin % node_batch_load_size),
                (int)(binary_tree_get_num_populated_nodes(group_start_node.level,
                                                          effective_num_levels,
                                                          num_particles) -
                      group_start_node.local_node_id));


        // Load nodes collectively into the cache
        if(subgroup_lid < num_available_nodes)
        {
          bbox_min_corner_cache[subgroup_lid] =
                             bbox_min_corner[node_idx_begin + subgroup_lid];
          bbox_max_corner_cache[subgroup_lid] =
                             bbox_max_corner[node_idx_begin + subgroup_lid];
        }
        subgroup_first_selected_nodes[subgroup_lid] = num_available_nodes;
        fast_barrier(CLK_LOCAL_MEM_FENCE);

        // For each query, iterate over the nodes in the
        // cache and check if nodes should be selected.
        if(tid < get_num_queries())
        {
//$pragma{ unroll node_batch_load_size}
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
            // in the first_selected_nodes local memory area
            if (node_selected)
            {
              subgroup_first_selected_nodes[subgroup_lid] = i;
              break;
            }
          }
        }
        fast_barrier(CLK_LOCAL_MEM_FENCE);

        // We must investigate deeper levels, if at least
        // one query of the subgroup wants to investigate deeper levels.
        // For this, we need a reduction.
        // We obtain the id of the first node that has been selected
        // by any work item.
        // If no node was selected, the id will still be set to
        // num_available_nodes, which is the value we initialized it with.
        const int first_node = subgroup_node_idx_min(subgroup_first_selected_nodes,
                                                     subgroup_lid);

        // Trigger the discard event for all skipped nodes
        if(tid < get_num_queries())
          for (int i = 0; i < first_node; ++i)
            dfs_unique_node_discard_event(node_idx_begin + i,
                                          bbox_min_corner_cache[i],
                                          bbox_max_corner_cache[i]);

        // Advance the number of covered particles
        num_covered_particles +=
                          first_node * BT_LEAVES_PER_NODE(group_start_node.level,
                                                          effective_num_levels);

        if(first_node < num_available_nodes)
        {
          // Some queries/work items want to go to the next deeper level

          // Calculate the vertical stride, i.e. how many levels we
          // are moving deeper. min() makes sure that we do not go deeper
          // than the lowest level.
          uint vertical_stride = min((uint)vertical_level_stride_size, 
                                     (uint)(effective_num_levels - group_start_node.level - 1));

          group_start_node.level += vertical_stride;
          group_start_node.local_node_id += first_node;
          group_start_node.local_node_id <<= vertical_stride;
        }
        else
        {
          // No query/work item wants to go deeper, so the entire work group
          // should move up in the tree
          group_start_node.level -= vertical_level_stride_size;
          group_start_node.local_node_id += num_available_nodes;
          group_start_node.local_node_id >>= vertical_level_stride_size;
          
        }
      }
    )
    QCL_RAW(
    
      __kernel void query(__global particle_type* particles,
                          __global vector_type* restrict bbox_min_corner,
                          __global vector_type* restrict bbox_max_corner,
                          ulong num_particles,
                          ulong effective_num_particles,
                          ulong effective_num_levels,
                          declare_full_query_parameter_set())
      {
        __local int node_selection_map [group_size];
        // This cache will be used both for particles and nodes.
        // It is important to use particle_type as type, since
        // sizeof(particle_type)>=sizeof(vector_type), hence
        // using particle_type guarantees correct alignment.
        __local particle_type cache [total_cache_size];

        //__local float cache[subgroup_size * 8 * num_subgroups];

        const int subgroup_id = get_local_id(0) / subgroup_size;
        const int subgroup_lid = get_local_id(0) % subgroup_size;

        volatile __local particle_type* const subgroup_cache =
                  cache + subgroup_id * subgroup_cache_size;
        //volatile __local float* const subgroup_cache = cache + subgroup_id*subgroup_size*8;
        volatile __local int* const subgroup_node_selection_map =
                  node_selection_map + subgroup_id * subgroup_size;

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
                                 subgroup_lid,
                                 subgroup_node_selection_map,
                                 subgroup_cache);
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
                             subgroup_lid,
                             subgroup_node_selection_map,
                             subgroup_cache);
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
