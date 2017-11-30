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

#ifndef ZCURVE_HPP
#define ZCURVE_HPP


#include "bit_manipulation.hpp"
#include "grid.hpp"
#include "configuration.hpp"
#include "sfc_position_generator.hpp"

namespace spatialcl {
namespace space_filling_curve {

template<class Type_descriptor>
class zcurve : public position_generator<Type_descriptor>
{
public:
  using vector_type = typename configuration<Type_descriptor>::vector_type;

  virtual void operator()(const qcl::device_context_ptr& ctx,
                          const vector_type& particles_min,
                          const vector_type& particles_max,
                          const cl::Buffer& particles,
                          cl_ulong num_particles,
                          const cl::Buffer& out) const override
  {
    cl::NDRange global_size{num_particles};
    cl::NDRange local_size{128};

    qcl::kernel_call gen_position = this->generate_zcurve_position(ctx,global_size,local_size);
    cl_int err = gen_position(particles_min, particles_max, particles, num_particles, out);

    qcl::check_cl_error(err, "Could not enqueue generate_zcurve_position kernel");
  }

  QCL_MAKE_MODULE(zcurve)

  // Number of cells resolved in 2d along each axis (32 bits set)
  static constexpr unsigned zcurve_num_cells2d = 0xffffffff;
  static constexpr unsigned zcurve_num_resolved_levels2d = 32;

  static constexpr unsigned zcurve_num_cells3d = 0x1fffff; //21 bits set
  static constexpr unsigned zcurve_num_resolved_levels3d = 21;

  QCL_ENTRYPOINT(generate_zcurve_position)
  QCL_MAKE_SOURCE
  (
    QCL_INCLUDE_MODULE(bit_manipulation)
    QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
    QCL_INCLUDE_MODULE(grid<Type_descriptor>)
    QCL_IMPORT_CONSTANT(zcurve_num_cells2d)
    QCL_IMPORT_CONSTANT(zcurve_num_resolved_levels2d)
    QCL_IMPORT_CONSTANT(zcurve_num_cells3d)
    QCL_IMPORT_CONSTANT(zcurve_num_resolved_levels3d)
    QCL_RAW
    (
      typedef ulong zcurve_key;

      typedef uint2 zcurve_cell_indices2d;
      typedef uint4 zcurve_cell_indices3d;

      // level 0 is the leaf level
      zcurve_key zcurve_min_key(zcurve_key leaf, uint level)
      {
        return leaf & ~n_bits_set(dimension * (level + 1));
      }

      zcurve_key zcurve_max_key(zcurve_key leaf, uint level)
      {
        return leaf | n_bits_set(dimension * (level + 1));
      }

      zcurve_key zcurve_position2d(zcurve_cell_indices2d pos)
      {
        return interleave_bits2(pos.x, pos.y);
      }

      zcurve_key zcurve_position3d(zcurve_cell_indices3d pos)
      {
        return interleave_bits3(pos.x, pos.y, pos.z);
      }

      __kernel void generate_zcurve_position(vector_type particles_min,
                                             vector_type particles_max,
                                             __global particle_type* particles,
                                             ulong num_particles,
                                             __global zcurve_key* out)
      {
        ulong num_cells = 0;

        if(dimension == 2)
          num_cells = zcurve_num_cells2d;
        else if(dimension == 3)
          num_cells = zcurve_num_cells3d;

        grid_t grid;
        grid_init(&grid, particles_min, particles_max, num_cells);

        size_t num_threads = get_global_size(0);

        for(size_t tid = get_global_id(0);
            tid < num_particles;
            tid += num_threads)
        {
          particle_type particle = particles[tid];
          vector_type position = PARTICLE_POSITION(particle);

          cell_index_type idx = grid_get_cell(&grid, position, num_cells);

          out[tid] = DIMENSIONALITY_SWITCH(
                         interleave_bits2(idx.x, idx.y),
                         interleave_bits3(idx.x, idx.y, idx.z));
        }
      }

    )
  )
};

}
}

#endif
