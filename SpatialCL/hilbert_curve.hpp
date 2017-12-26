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

#ifndef HILBERT_CURVE_HPP
#define HILBERT_CURVE_HPP

#include "configuration.hpp"
#include "bit_manipulation.hpp"
#include "sfc_position_generator.hpp"
#include "grid.hpp"


namespace spatialcl {
namespace space_filling_curve {

template<class Type_descriptor>
class hilbert_curve : public position_generator<Type_descriptor>
{
public:
  QCL_MAKE_MODULE(hilbert_curve)

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

    qcl::kernel_call gen_position = this->generate_hilbert_position(ctx,global_size,local_size);
    cl_int err = gen_position(particles_min, particles_max, particles, num_particles, out);

    qcl::check_cl_error(err, "Could not enqueue generate_hilbert_position kernel");
  }

  virtual ~hilbert_curve(){}

  static constexpr unsigned hilbert_num_cells2d = 0xffffffff;
  static constexpr unsigned hilbert_num_resolved_levels2d = 32;

  static constexpr unsigned hilbert_num_cells3d = 0x1fffff; //21 bits set
  static constexpr unsigned hilbert_num_resolved_levels3d = 21;

  QCL_ENTRYPOINT(generate_hilbert_position)
  QCL_MAKE_SOURCE
  (
    QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
    QCL_INCLUDE_MODULE(grid<Type_descriptor>)
    QCL_INCLUDE_MODULE(bit_manipulation)
    QCL_IMPORT_CONSTANT(hilbert_num_cells2d)
    QCL_IMPORT_CONSTANT(hilbert_num_resolved_levels2d)
    QCL_IMPORT_CONSTANT(hilbert_num_cells3d)
    QCL_IMPORT_CONSTANT(hilbert_num_resolved_levels3d)
    QCL_RAW(

      typedef ulong hilbert_key;

      typedef uint2 hilbert_cell_indices2d;
      typedef uint4 hilbert_cell_indices3d;

      // This calculation of the position along the hilbert
      // curve is based on the algorithm presented in:
      // John Skilling: Programming the Hilbert curve
      // AIP Conference Proceedings 707, 381 (2004); doi: 10.1063/1.1751381
      hilbert_key hilbert_position2d(hilbert_cell_indices2d pos)
      {
        uint X [2];
        X[0] = pos.x;
        X[1] = pos.y;

        const int n = 2;
        const int bits = 32;

        uint M = 1U << (bits - 1);
        uint P;
        uint Q;
        uint t;
        int i;
        // Inverse undo
        for (Q = M; Q > 1; Q >>= 1)
        {
          P = Q - 1;
          for (i = 0; i < n; i++)
            if ((X[i] & Q) != 0)
              X[0] ^= P; // invert
            else
            {
              t = (X[0] ^ X[i]) & P;
              X[0] ^= t;
              X[i] ^= t;
            }
        }// exchange
        // Gray encode
        for (i = 1; i < n; i++)
          X[i] ^= X[i - 1];
        t = 0;
        for (Q = M; Q > 1; Q >>= 1)
          if ((X[n - 1] & Q)!=0)
            t ^= Q - 1;
        for (i = 0; i < n; i++)
          X[i] ^= t;

        return interleave_bits2(X[1], X[0]);
      }

      hilbert_key hilbert_position3d(hilbert_cell_indices3d pos)
      {
        uint X [3];
        X[0] = pos.x;
        X[1] = pos.y;
        X[2] = pos.z;

        const int n = 3;
        const int bits = 21;

        uint M = 1U << (bits - 1);
        uint P;
        uint Q;
        uint t;
        int i;
        // Inverse undo
        for (Q = M; Q > 1; Q >>= 1)
        {
          P = Q - 1;
          for (i = 0; i < n; i++)
            if ((X[i] & Q) != 0)
              X[0] ^= P; // invert
            else
            {
              t = (X[0] ^ X[i]) & P;
              X[0] ^= t;
              X[i] ^= t;
            }
        }// exchange
        // Gray encode
        for (i = 1; i < n; i++)
          X[i] ^= X[i - 1];
        t = 0;
        for (Q = M; Q > 1; Q >>= 1)
          if ((X[n - 1] & Q)!=0)
            t ^= Q - 1;
        for (i = 0; i < n; i++)
          X[i] ^= t;

        return interleave_bits3(X[2], X[1], X[0]);
      }

      ulong hilbert_position(cell_index_type pos)
      {
        return DIMENSIONALITY_SWITCH(hilbert_position2d(pos),
                                     hilbert_position3d(pos));
      }

      __kernel void generate_hilbert_position(const vector_type particles_min,
                                              const vector_type particles_max,
                                              __global particle_type* particles,
                                              ulong num_particles,
                                              __global hilbert_key* out)
      {
        ulong num_cells = 0;

        if(dimension == 2)
          num_cells = hilbert_num_cells2d;
        else if(dimension == 3)
          num_cells = hilbert_num_cells3d;

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

          out[tid] = hilbert_position(idx);
        }
      }
    )
  )
};

}
}

#endif
