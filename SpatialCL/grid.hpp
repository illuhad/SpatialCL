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

#ifndef GRID_HPP
#define GRID_HPP


#include "configuration.hpp"

namespace spatialcl {

template<class Type_descriptor>
struct grid
{
  QCL_MAKE_MODULE(grid)
  QCL_MAKE_SOURCE
  (
    QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
    QCL_RAW(

      typedef struct
      {
        vector_type min_corner;
        vector_type cell_sizes;
      } grid_t;

      void grid_init(grid_t* ctx,
                     vector_type grid_min,
                     vector_type grid_max,
                     ulong num_cells_per_dim)
      {
        ctx->min_corner = grid_min;
        ctx->cell_sizes = (grid_max - grid_min) / num_cells_per_dim;
      }

      cell_index_type grid_get_cell(grid_t* ctx,
                                    vector_type point,
                                    ulong num_cells_per_dim)
      {
        vector_type float_cell_index = (point - ctx->min_corner) / ctx->cell_sizes;
        cell_index_type result = CONVERT_VECTOR_TO_CELL_INDEX(float_cell_index);

        return result;
      }
    )
  )
};
}

#endif
