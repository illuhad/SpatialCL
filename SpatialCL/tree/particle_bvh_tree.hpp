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

#ifndef PARTICLE_BVH_TREE
#define PARTICLE_BVH_TREE

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>
#include <QCL/qcl_array.hpp>
#include "particle_tree.hpp"

namespace spatialcl {

template<class Particle_sorter,
         class Type_descriptor>
class particle_bvh_tree : public particle_tree<Particle_sorter,
                                               Type_descriptor,
                                               typename configuration<Type_descriptor>::vector_type,
                                               typename configuration<Type_descriptor>::vector_type>
{
public:
  QCL_MAKE_MODULE(particle_bvh_tree)

  using particle_type = typename configuration<Type_descriptor>::particle_type;
  using vector_type = typename configuration<Type_descriptor>::vector_type;

  using base_type = particle_tree<
    Particle_sorter,
    Type_descriptor,
    typename configuration<Type_descriptor>::vector_type,
    typename configuration<Type_descriptor>::vector_type
  >;

  particle_bvh_tree(const qcl::device_context_ptr& ctx,
                    const std::vector<particle_type>& particles,
                    const Particle_sorter& sorter = Particle_sorter{})
    : base_type{ctx, particles, sorter}
  {
    this->rebuild_bounding_boxes();
  }

  particle_bvh_tree(const qcl::device_context_ptr& ctx,
                    const cl::Buffer& particles,
                    std::size_t num_particles,
                    const Particle_sorter& sorter = Particle_sorter{})
    : base_type{ctx, particles, num_particles, sorter}
  {
    this->rebuild_bounding_boxes();
  }

  particle_bvh_tree(const qcl::device_context_ptr& ctx,
                    const qcl::device_array<particle_type>& particles,
                    const Particle_sorter& sorter = Particle_sorter{})
    : base_type{ctx, particles, sorter}
  {
    this->rebuild_bounding_boxes();
  }

  virtual ~particle_bvh_tree(){}

  void rebuild_bounding_boxes()
  {
    // Build the lowest layer of nodes. Because the particles
    // themselves are the lowest level, this is actually the second level.
    this->build_lowest_level_bboxes();

    // Build the next layers. We start from num_levels-3, because num_levels-1 corresponds
    // to the particle layer, and num_levels-2 is the lowest node layer that was already
    // built by build_lowest_level_bboxes().
    for(int i = static_cast<int>(this->get_effective_num_levels())-3; i >= 0; --i)
    {
#ifndef NODEBUG
      std::cout << "Building level " << i << std::endl;
#endif
      build_higher_level_bboxes(static_cast<unsigned>(i));
    }
  }

  const cl::Buffer& get_bbox_min_corners() const
  {
    return this->get_node_values0();
  }

  const cl::Buffer& get_bbox_max_corners() const
  {
    return this->get_node_values1();
  }

private:

  static constexpr std::size_t local_size = 256;

  void build_lowest_level_bboxes() const
  {
    cl::NDRange global_size {this->get_effective_num_particles()/2};
    cl::NDRange local_size {this->local_size};

    cl_int err = bvh_tree_build_ll_bbox(this->get_device_context(), global_size, local_size)(
         this->get_sorted_particles(),
         static_cast<cl_ulong>(this->get_num_particles()),
         this->get_node_values0(),
         this->get_node_values1());

    qcl::check_cl_error(err, "Could not enqueue bvh_tree_build_ll_bbox kernel");
  }

  void build_higher_level_bboxes(unsigned level_id) const
  {
    assert(level_id < this->get_effective_num_levels());


    cl::NDRange global_size {1ull << level_id};
    cl::NDRange local_size {local_size};

    cl_int err = bvh_tree_build_bbox(this->get_device_context(), global_size, local_size)(
        this->get_node_values0(),
        this->get_node_values1(),
        static_cast<cl_uint>(level_id),
        static_cast<cl_uint>(this->get_effective_num_levels()),
        static_cast<cl_ulong>(this->get_num_particles()));

    qcl::check_cl_error(err, "Could not enqueue bvh_tree_build_bbox kernel.");
  }

  QCL_ENTRYPOINT(bvh_tree_build_ll_bbox)
  QCL_ENTRYPOINT(bvh_tree_build_bbox)
  QCL_MAKE_SOURCE
  (
    QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
    QCL_INCLUDE_MODULE(binary_tree)
    QCL_INCLUDE_MODULE(cl_utils::debug)
    QCL_RAW
    (

      __kernel void bvh_tree_build_ll_bbox(__global particle_type* particles,
                                           index_type num_particles,
                                           __global vector_type* nodes_min_corner,
                                           __global vector_type* nodes_max_corner)
      {
        index_type num_threads = get_global_size(0);

        index_type effective_num_particles = BT_EFFECTIVE_NUM_LEAVES(num_particles);
        uint num_levels = find_most_significant_bit(effective_num_particles) + 1;

        for(index_type tid = get_global_id(0);
            tid < effective_num_particles/2;
            tid += num_threads)
        {
          index_type particle0_idx = 2 * tid;
          index_type particle1_idx = 2 * tid + 1;

          particle_type particle0 = (particle_type)(0.0f);

          if(particle0_idx < num_particles)
            particle0 = particles[particle0_idx];

          particle_type particle1 = particle0;

          if(particle1_idx < num_particles)
            particle1 = particles[particle1_idx];

          vector_type bb_min_position = fmin(PARTICLE_POSITION(particle0),
                                             PARTICLE_POSITION(particle1));


          vector_type bb_max_position = fmax(PARTICLE_POSITION(particle0),
                                             PARTICLE_POSITION(particle1));

          nodes_min_corner[tid] = bb_min_position;
          nodes_max_corner[tid] = bb_max_position;
        }
      }

      /// Builds bounding boxed for higher nodes - assumes that the lowest
      /// level of nodes has already been constructed using
      /// bvh_tree_build_ll_bbox()
      __kernel void bvh_tree_build_bbox(__global vector_type* nodes_min_corner,
                                        __global vector_type* nodes_max_corner,
                                        uint level,
                                        uint num_levels,
                                        index_type num_particles)
      {
        index_type num_threads = get_global_size(0);

        index_type num_nodes = BT_NUM_NODES(level);
        //index_type offset_correction = BT_EFFECTIVE_NUM_LEAVES(num_particles) - num_particles;
        index_type offset_correction = BT_EFFECTIVE_NUM_LEAVES(num_particles);

        for(index_type tid = get_global_id(0);
            tid < num_nodes;
            tid += num_threads)
        {
          binary_tree_key_t node_key;
          binary_tree_key_init(&node_key, level, tid);

          int node_exists = binary_tree_is_node_used(&node_key,
                                                     num_levels,
                                                     num_particles);

          vector_type bb_min = (vector_type)(0.0f);
          vector_type bb_max = (vector_type)(0.0f);

          if(node_exists)
          {
            binary_tree_key_t children_begin = binary_tree_get_children_begin(&node_key);
            binary_tree_key_t children_last  = binary_tree_get_children_last (&node_key);

            int right_child_exists = binary_tree_is_node_used(&children_last,
                                                              num_levels,
                                                              num_particles);

            index_type child_idx = binary_tree_key_encode_global_id(&children_begin,
                                                                    num_levels);

            // We already know that the new parent node exists, so at least the left
            // child must exist as well. We therefore do not need to check the existence
            // of the left child, and can directly access its data.
            ASSERT(child_idx >= offset_correction);
            bb_min = nodes_min_corner[child_idx - offset_correction];
            bb_max = nodes_max_corner[child_idx - offset_correction];

            if(right_child_exists)
            {
              child_idx = binary_tree_key_encode_global_id(&children_last, num_levels);
              ASSERT(child_idx >= offset_correction);
              bb_min = fmin(bb_min, nodes_min_corner[child_idx - offset_correction]);
              bb_max = fmax(bb_max, nodes_max_corner[child_idx - offset_correction]);
            }
          }

          index_type node_idx = binary_tree_key_encode_global_id(&node_key, num_levels);
          nodes_min_corner[node_idx - offset_correction] = bb_min;
          nodes_max_corner[node_idx - offset_correction] = bb_max;
        }
      }
    )
  )
};


}
#endif
