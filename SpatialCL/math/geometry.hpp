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

#ifndef MATH_GEOMETRY_HPP
#define MATH_GEOMETRY_HPP

#include "../configuration.hpp"

namespace spatialcl {
namespace math {

template<class Type_descriptor>
QCL_STANDALONE_MODULE(geometry)
QCL_STANDALONE_SOURCE
(
  QCL_INCLUDE_MODULE(configuration<Type_descriptor>)
  QCL_RAW
  (
    scalar box_distance2(vector_type point,
                         vector_type box_min,
                         vector_type box_max)
    {
      vector_type zero = (vector_type)0.0f;
      vector_type delta = fmax(box_min - point,
                              fmax(zero, point - box_max));
      return VECTOR_NORM2(delta);
    }

    int box_box_intersection(vector_type a_min,
                             vector_type a_max,
                             vector_type b_min,
                             vector_type b_max)
    {
      // if b_min > a_max || a_min > b_max
      //   return false

      int_vector_type intersects = ((a_max >= b_min) && (b_max >= a_min));
      int result = DIMENSIONALITY_SWITCH(intersects.x & intersects.y & 1,
                                         intersects.x & intersects.y & intersects.z & 1);

      return result;
    }


    int box_contains_point(vector_type box_min,
                           vector_type box_max,
                           vector_type point)
    {
      int_vector_type contains = (box_min <= point) &&
                                 (box_max >= point);
      return DIMENSIONALITY_SWITCH(contains.x & contains.y & 1,
                                   contains.x & contains.y & contains.z & 1);
    }

    int box_contains_particle(vector_type box_min,
                              vector_type box_max,
                              particle_type p)
    {
      return box_contains_point(box_min, box_max,
                                PARTICLE_POSITION(p));
    }
  )
)

}
}

#endif
