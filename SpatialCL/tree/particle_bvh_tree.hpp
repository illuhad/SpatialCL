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

#ifndef PARTICLE_BVH_TREE
#define PARTICLE_BVH_TREE

#include <boost/compute.hpp>
#include <cassert>
#include <functional>
#include "binary_tree.hpp"
#include "../configuration.hpp"
#include "../cl_utils.hpp"


namespace spatialcl {

template<class Particle_sorter, class Type_descriptor>
class particle_bvh_tree
{
public:
  using particle_type = typename configuration<Type_descriptor>::particle_type;
  using vector_type = typename configuration<Type_descriptor>::vector_type;
  using boost_particle = typename qcl::to_boost_vector_type<particle_type>::type;

  using type_system = Type_descriptor;

  particle_bvh_tree(const qcl::device_context_ptr& ctx,
                    const std::vector<particle_type>& particles,
                    const Particle_sorter& sorter = Particle_sorter{})
    : _ctx{ctx},
      _num_particles{particles.size()}
  {
    _ctx->create_buffer<particle_type>(this->_sorted_particles, particles.size());
    _ctx->memcpy_h2d(this->_sorted_particles, particles.data(), particles.size());

    this->init_bvh_tree(sorter);
  }

  particle_bvh_tree(const qcl::device_context_ptr& ctx,
                    const cl::Buffer& particles,
                    std::size_t num_particles,
                    const Particle_sorter& sorter = Particle_sorter{})
    : _ctx{ctx},
      _sorted_particles{particles},
      _num_particles{num_particles}
  {
    this->init_bvh_tree(sorter);
  }

  const cl::Buffer& get_bbox_min_corners() const
  {
    return this->_bb_min_corners;
  }

  const cl::Buffer& get_bbox_max_corners() const
  {
    return this->_bb_max_corners;
  }

  std::size_t get_num_nodes() const
  {
    return this->_effective_num_particles-1;
  }

  std::size_t get_effective_num_levels() const
  {
    return this->_num_levels;
  }

  const cl::Buffer& get_sorted_particles() const
  {
    return this->_sorted_particles;
  }

  template<class Query_engine_type>
  cl_int execute_query(Query_engine_type& engine,
                       typename Query_engine_type::handler_type& handler,
                       cl::Event* evt = nullptr) const
  {
    return engine( this->_ctx,
                   this->_sorted_particles,
                   this->_bb_min_corners,
                   this->_bb_max_corners,
                   this->_num_particles,
                   this->_effective_num_particles,
                   this->_num_levels,
                   handler,
                   evt);
  }

  QCL_MAKE_MODULE(particle_bvh_tree)
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
private:
  void init_bvh_tree(const Particle_sorter& sorter)
  {
    // First sort the particles spatially
    sorter(_ctx, _sorted_particles, _num_particles);


    // Build binary bvh tree on top of the sorted
    // particles
    // First, calculate the required number of levels
    // and the effective number of particles
    _effective_num_particles = get_next_power_of_two(_num_particles);
    _num_levels = get_highest_set_bit(_effective_num_particles)+1;
    std::cout << "Building tree with "
              << _num_levels << " levels over "
              << _effective_num_particles << " effective particles and "
              << _num_particles << " real particles." << std::endl;

    _ctx->create_buffer<vector_type>(this->_bb_min_corners, this->_effective_num_particles);
    _ctx->create_buffer<vector_type>(this->_bb_max_corners, this->_effective_num_particles);
    // Now build the lowest layer of nodes. Because the particles
    // themselves are the lowest level, this is actually the second level.
    this->build_lowest_level_bboxes();

    // Build the next layers. We start from num_levels-3, because num_levels-1 corresponds
    // to the particle layer, and num_levels-2 is the lowest node layer that was already
    // built by build_lowest_level_bboxes().
    for(int i = static_cast<int>(_num_levels)-3; i >= 0; --i)
    {
      std::cout << "Building level " << i << std::endl;
      build_higher_level_bboxes(static_cast<unsigned>(i));
    }
  }

  static unsigned get_highest_set_bit(uint64_t x)
  {
    unsigned result = 0;

    for(int i = 0; i < 64; ++i)
      if(x & (1ull << i))
        result = i;

    return result;
  }

  static uint64_t get_next_power_of_two(uint64_t x)
  {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x++;

    return x;
  }

  void build_lowest_level_bboxes() const
  {
    cl::NDRange global_size {_effective_num_particles/2};
    cl::NDRange local_size {_local_size};

    cl_int err = bvh_tree_build_ll_bbox(_ctx, global_size, local_size)(_sorted_particles,
                                                                       static_cast<cl_ulong>(_num_particles),
                                                                       _bb_min_corners,
                                                                       _bb_max_corners);

    qcl::check_cl_error(err, "Could not enqueue bvh_tree_build_ll_bbox kernel");
  }

  void build_higher_level_bboxes(unsigned level_id) const
  {
    assert(level_id < _num_levels);


    cl::NDRange global_size {1ull << level_id};
    cl::NDRange local_size {_local_size};

    cl_int err = bvh_tree_build_bbox(_ctx, global_size, local_size)(_bb_min_corners,
                                                                    _bb_max_corners,
                                                                    static_cast<cl_uint>(level_id),
                                                                    static_cast<cl_uint>(_num_levels),
                                                                    static_cast<cl_ulong>(_num_particles));

    qcl::check_cl_error(err, "Could not enqueue bvh_tree_build_bbox kernel.");
  }

  qcl::device_context_ptr _ctx;

  cl::Buffer _sorted_particles;

  cl::Buffer _bb_min_corners;
  cl::Buffer _bb_max_corners;

  unsigned _num_levels;

  std::size_t _num_particles;
  std::size_t _effective_num_particles;

  static constexpr std::size_t _local_size = 256;
};

}
#endif
