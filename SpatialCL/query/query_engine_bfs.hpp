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

#ifndef QUERY_ENGINE_BFS_HPP
#define QUERY_ENGINE_BFS_HPP


#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>

#include "../configuration.hpp"
#include "../tree/binary_tree.hpp"

namespace spatialcl {
namespace query {
namespace engine {

/// Breadth-first query engine that stores the
/// query state in registers, and is hence well suited
/// for queries where the number of investigated nodes
/// per level is known to be small enough to fit in the
/// GPU's registers.
template<class Type_descriptor,
         class Handler_module,
         std::size_t Max_selected_nodes>
class register_breadth_first
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

  QCL_MAKE_MODULE(register_breadth_first)
  QCL_ENTRYPOINT(query)
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
    QCL_INCLUDE_MODULE(Handler_module)
    QCL_INCLUDE_MODULE(binary_tree)
    QCL_IMPORT_CONSTANT(Max_selected_nodes)
    // This macro maps the index (simple enumeration from 0 to Max_children)
    // of available children nodes to their local node ids. This is done
    // by first, calculating the position of the parent in the local node id buffer
    // which is at id/2 == id >> 1. The local node id is then either
    // the local node id of the left or right child.
    // We obtain the local node id of the left child from
    // BT_LOCAL_NODE_ID_OF_LEFT_CHILD(). If its a right child,
    // i.e. if id&1 == 1, we add one since the right child always
    // directly follows the left child.
    QCL_PREPROCESSOR(define,
      MAP_AVAILABLE_CHILDREN_INDEX_TO_LNID(parent_lnid_buffer, id)
        (BT_LOCAL_NODE_ID_OF_LEFT_CHILD(parent_lnid_buffer[(id) >> 1]) + ((id)&1))
    )
    QCL_PREPROCESSOR(define, get_query_id() tid)
    QCL_PREPROCESSOR(define, bfs_load_node(id)
    {
      binary_tree_key_t node_key;
      node_key.level = level;
      node_key.local_node_id =
        MAP_AVAILABLE_CHILDREN_INDEX_TO_LNID(available_nodes_local_id, id);

      global_node_idx = binary_tree_key_encode_global_id(&node_key,
                                                         effective_num_levels);
      global_node_idx -= num_particles;
    })
    // For particles, the local node id always equals the global id
    // because they are at the lowest level. We hence
    // do not need to call binary_tree_key_encode_global_id(),
    // but can directly use the particle's local node id
    // as index to the particle array
    QCL_PREPROCESSOR(define, bfs_load_particle(id)
      particles[MAP_AVAILABLE_CHILDREN_INDEX_TO_LNID(available_nodes_local_id, id)]
    )
    QCL_PREPROCESSOR(define, bfs_get_node_min_corner()
      bbox_min_corner[global_node_idx]
    )
    QCL_PREPROCESSOR(define, bfs_get_node_max_corner()
      bbox_max_corner[global_node_idx]
    )
    QCL_PREPROCESSOR(define, bfs_get_node_global_index()
      global_node_idx
    )
    QCL_PREPROCESSOR(define, bfs_select(id)
    {
      selection_map[id] = 1;
    })
    QCL_PREPROCESSOR(define, bfs_deselect(id)
    {
      selection_map[id] = 0;
    })
    QCL_PREPROCESSOR(define, Max_children
                     (2*Max_selected_nodes)
    )
    QCL_RAW(

      ulong get_root_index(ulong num_levels,
                           ulong num_particles)
      {
        binary_tree_key_t root_node;
        root_node.level = 0;
        root_node.local_node_id = 0;

        return binary_tree_key_encode_global_id(&root_node,
                                                num_levels)
               - num_particles;
      }

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
          // Call init handler
          at_query_init();

          uint num_available_nodes = 1;
          ulong available_nodes_local_id [Max_selected_nodes];
          uchar selection_map            [Max_children];
          available_nodes_local_id[0] = 0;

          for(uint level = 0; level < 64; ++level)
          {
            // Stop if we haven't selected any nodes in
            // the parent level, or if we have reached
            // the lowest level - the lowest level is
            // populated with particles and not nodes
            // and must hence be treated differently.
            if(num_available_nodes == 0
              || level == effective_num_levels-1)
              break;

            // Begin by marking all children as unselected
            for(uint i = 0; i < Max_children; ++i)
              selection_map[i] = 0;

            // Calculate how many children we can investigate.
            // Since we are using a binary tree, this is typically
            // twice the number of nodes that we have available
            // in the current level. However, if a level is underpopulated,
            // the rightmost child in this level may not exist.
            // In this case, subtract one from the available children.
            uint available_children = 2 * num_available_nodes;
            binary_tree_key_t last_child;
            binary_tree_key_init(&last_child, level,
                                 available_nodes_local_id[num_available_nodes-1]);
            last_child = binary_tree_get_children_last(&last_child);
            if(!binary_tree_is_node_used(&last_child,
                                         effective_num_levels,
                                         num_particles))
              --available_children;

            // Will be filled by bfs_load_node calls in
            // the node selector
            ulong global_node_idx = 0;
            // Run node selector to obtain children for investigation
            bfs_node_selector(Max_selected_nodes,
                              available_children);
            // Count number of selected nodes and copy
            // their local node id to the available_nodes_local_id
            // array
            num_available_nodes = 0;
            for(uint i = 0; i < Max_children; ++i)
            {
              if(selection_map[i])
              {
                ulong node_id =
                  MAP_AVAILABLE_CHILDREN_INDEX_TO_LNID(available_nodes_local_id, i);

                available_nodes_local_id[num_available_nodes] = node_id;
                // Increase the number of available nodes, but
                // make sure there are no more available nodes
                // selected than the maximum allowed
                num_available_nodes = min(num_available_nodes+1,
                                          (uint)Max_selected_nodes);
              }
            }
          }
          // Make sure there are parent nodes before
          // trying to investigate particles
          if(num_available_nodes > 0)
          {
            // Process particles at the lowest level. Since we
            // are using a binary tree, we expect twice the number
            // of particles than nodes in the lowest node level.
            uint num_available_particles = 2 * num_available_nodes;
            // The last particle may be nonexistent. We do not need
            // create a full-blown node object to check this,
            // we can exploit that for the particles,
            // the local node id corresponds directly to the
            // index in the particle array
            if(MAP_AVAILABLE_CHILDREN_INDEX_TO_LNID(available_nodes_local_id,
                                                    2*num_available_nodes - 1)
               >= num_particles)
              --num_available_particles;

            bfs_particle_processor(num_available_particles);
          }
          // Call exit handler
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
