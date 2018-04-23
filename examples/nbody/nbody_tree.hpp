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

#ifndef NBODY_TREE
#define NBODY_TREE

#include <cassert>

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>
#include <QCL/qcl_array.hpp>

#include <SpatialCL/tree/binary_tree.hpp>
#include <SpatialCL/configuration.hpp>
#include <SpatialCL/tree.hpp>

namespace nbody {

template<class Scalar>
using nbody_type_descriptor = spatialcl::type_descriptor::generic<Scalar,3,4>;

template<class Scalar>
using hilbert_sorter =
  spatialcl::key_based_sorter<
    spatialcl::hilbert_sort_key_generator<
      nbody_type_descriptor<Scalar>
    >
  >;

template<class Scalar>
using nbody_basic_tree = spatialcl::particle_tree
<
  hilbert_sorter<Scalar>,
  nbody_type_descriptor<Scalar>,
  typename spatialcl::configuration<nbody_type_descriptor<Scalar>>::vector_type,
  typename spatialcl::configuration<nbody_type_descriptor<Scalar>>::vector_type
>;

template<class Scalar>
class nbody_tree : public nbody_basic_tree<Scalar>
{
public:
  QCL_MAKE_MODULE(nbody_tree)

  using particle_type =
    typename spatialcl::configuration<nbody_type_descriptor<Scalar>>::particle_type;
  using vector_type =
    typename spatialcl::configuration<nbody_type_descriptor<Scalar>>::vector_type;


  nbody_tree(const qcl::device_context_ptr& ctx,
             const qcl::device_array<particle_type>& particles)
      : nbody_basic_tree<Scalar>{ctx, particles}, _ctx{ctx}
  {
    // Need at least two particles for the tree
    assert(particles.size() > 2);
    this->init_multipoles();
  }

private:
  void init_multipoles()
  {
    // First, calculate the diameter of each node and store in bbox max
    // corner.w
    // TODO Fix this!! This is not correct anymore!
    cl_int err = this->store_bbox_diameter(_ctx,
                                           cl::NDRange{this->get_num_nodes()},
                                           cl::NDRange{256})
        (this->get_node_values0(),
         this->get_node_values1(),
         static_cast<cl_ulong>(this->get_num_nodes()));

    qcl::check_cl_error(err, "Could not enqueue store_bbox_diameter kernel");

    assert(this->get_num_node_levels() > 0);
    // Build monopoles for lowest level.

    // First, build lowest level
    err = this->build_ll_monopoles(_ctx,
                                   cl::NDRange{this->get_num_particles()},
                                   cl::NDRange{256})
        (this->get_node_values0(),
         this->get_sorted_particles(),
         static_cast<cl_ulong>(this->get_num_particles()));

    qcl::check_cl_error(err, "Could not enqueue build_ll_monopoles kernel");

    for(int level = this->get_num_node_levels()-2; level >= 0; --level)
    {
      // Build higher levels
      err = this->build_monopoles(_ctx,
                                  cl::NDRange{1ul << level},
                                  cl::NDRange{256})
          (this->get_node_values0(),
           static_cast<cl_uint>(level),
           static_cast<cl_ulong>(this->get_num_particles()),
           static_cast<cl_ulong>(this->get_effective_num_particles()),
           static_cast<cl_ulong>(this->get_effective_num_levels()));
      qcl::check_cl_error(err, "Could not enqueue build_monopoles kernel");
    }
    err = _ctx->get_command_queue().finish();
    qcl::check_cl_error(err,"Error while waiting for build_monopoles kernel to finish");
  }

  qcl::device_context_ptr _ctx;


  QCL_ENTRYPOINT(build_ll_monopoles)
  QCL_ENTRYPOINT(build_monopoles)
  QCL_ENTRYPOINT(store_bbox_diameter)
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(spatialcl::configuration<nbody_type_descriptor<Scalar>>)
    QCL_INCLUDE_MODULE(spatialcl::binary_tree)
    QCL_RAW (
      // Stores the mean bbox diameter in bbox_max_corners.w
      __kernel void store_bbox_diameter(__global vector_type* bbox_min_corners,
                                        __global vector_type* bbox_max_corners,
                                        ulong num_nodes)
      {
        for(ulong tid = get_global_id(0);
            tid < num_nodes;
            tid += get_global_size(0))
        {
          vector_type bbox_min = bbox_min_corners[tid];
          vector_type bbox_max = bbox_max_corners[tid];

          vector_type delta = bbox_max - bbox_min;
          scalar diameter = (delta.x + delta.y + delta.z) / 3.0f;

          bbox_max_corners[tid].w = diameter;
        }
      }

      // Build monopoles on lowest level
      __kernel void build_ll_monopoles(__global vector_type* monopoles,
                                       __global particle_type* particles,
                                       ulong num_particles)
      {
        ulong num_nodes = num_particles >> 1;
        if(num_particles & 1)
          ++num_nodes;

        for(ulong tid = get_global_id(0);
            tid < num_nodes;
            tid += get_global_size(0))
        {
          ulong left_particle_idx = tid << 1;
          ulong right_particle_idx = left_particle_idx + 1;

          particle_type left_particle = particles[left_particle_idx];
          particle_type right_particle = particles[right_particle_idx];

          vector_type monopole;
          monopole.xyz = left_particle.s3 * left_particle.s012;
          monopole.xyz += right_particle.s3 * right_particle.s012;

          scalar total_mass = left_particle.s3 + right_particle.s3;

          monopole.xyz /= total_mass;
          monopole.w = total_mass;

          monopoles[tid] = monopole;
        }
      }
      // Build monopoles on higher levels
      __kernel void build_monopoles(__global vector_type* monopoles,
                                    uint current_level,
                                    ulong num_particles,
                                    ulong effective_num_particles,
                                    ulong effective_num_levels)
      {
        ulong num_nodes = BT_NUM_NODES(current_level);

        for(ulong tid = get_global_id(0);
            tid < num_nodes;
            tid += get_global_size(0))
        {
          binary_tree_key_t node_key;
          binary_tree_key_init(&node_key, current_level, tid);

          if(binary_tree_is_node_used(&node_key,
                                      effective_num_levels,
                                      num_particles))
          {
            binary_tree_key_t children_begin = binary_tree_get_children_begin(&node_key);
            binary_tree_key_t children_last  = binary_tree_get_children_last (&node_key);

            int right_child_exists = binary_tree_is_node_used(&children_last,
                                                              effective_num_levels,
                                                              num_particles);

            ulong child_idx = binary_tree_key_encode_global_id(&children_begin,
                                                               effective_num_levels);

            vector_type left_child_monopole = monopoles[child_idx - effective_num_particles];
            vector_type right_child_monopole = (vector_type)0.0f;

            if(right_child_exists)
            {
              child_idx = binary_tree_key_encode_global_id(&children_last,
                                                           effective_num_levels);
              right_child_monopole = monopoles[child_idx - effective_num_particles];
            }
            scalar left_mass = left_child_monopole.w;
            scalar right_mass = right_child_monopole.w;
            // Calculate center of mass
            vector_type parent_monopole =
              left_mass * left_child_monopole + right_mass * right_child_monopole;
            scalar total_mass = left_mass + right_mass;
            parent_monopole.xyz /= total_mass;
            // Set total mass
            parent_monopole.w = total_mass;

            ulong node_idx = binary_tree_key_encode_global_id(&node_key,
                                                              effective_num_levels);
            // Set result
            monopoles[node_idx - effective_num_particles] = parent_monopole;
          }
        }
      }
    )
  )
};


}

#endif
