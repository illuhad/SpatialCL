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

#ifndef CONFIGURATION_HPP
#define CONFIGURATION_HPP

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>
#include <QCL/qcl_boost_compat.hpp>

#include "types.hpp"

namespace spatialcl {
namespace type_descriptor{

template<class Scalar_type,
         std::size_t Dimension,
         std::size_t Num_particle_components>
struct generic
{
  using scalar = Scalar_type;
  static constexpr std::size_t dimension = Dimension;
  static constexpr std::size_t particle_dimension = Num_particle_components;

  static_assert(particle_dimension >= dimension, "Number of components of a particle must"
                                                 " be greater or equal to the dimensionality of"
                                                 " the problem.");
};

template<std::size_t Num_particle_components>
using single_precision2d = generic<float, 2, Num_particle_components>;

template<std::size_t Num_particle_components>
using single_precision3d = generic<float, 3, Num_particle_components>;

template<std::size_t Num_particle_components>
using double_precision2d = generic<double, 2, Num_particle_components>;

template<std::size_t Num_particle_components>
using double_precision3d = generic<double, 3, Num_particle_components>;

}

template<class Type_descriptor>
struct configuration
{
  QCL_MAKE_MODULE(configuration)

  using scalar = typename Type_descriptor::scalar;
  using cell_index_type = typename cl_vector_type<unsigned, Type_descriptor::dimension>::value;
  using int_vector_type = typename cl_vector_type<int, Type_descriptor::dimension>::value;
  using vector_type   = typename cl_vector_type<scalar, Type_descriptor::dimension>::value;
  using particle_type = typename cl_vector_type<scalar, Type_descriptor::particle_dimension>::value;
  static constexpr std::size_t dimension = Type_descriptor::dimension;

  static_assert(dimension == 2 || dimension == 3,
                "Only 2D and 3D is supported");

private:
  QCL_MAKE_SOURCE(
    QCL_IMPORT_TYPE(cell_index_type)
    QCL_IMPORT_TYPE(vector_type)
    QCL_IMPORT_TYPE(int_vector_type)
    QCL_IMPORT_TYPE(particle_type)
    QCL_IMPORT_TYPE(scalar)
    QCL_IMPORT_CONSTANT(dimension)
    R"(
     #if dimension == 2
      #define PARTICLE_POSITION(p) p.s01
      #define CONVERT_VECTOR_TO_CELL_INDEX(v) convert_uint2(v)
      #define VECTOR_NORM2(v) dot(v,v)
      #define CLIP_TO_VECTOR(v) ((v).xy)
     #elif dimension == 3
      #define PARTICLE_POSITION(p) p.s0123
      #define CONVERT_VECTOR_TO_CELL_INDEX(v) convert_uint4(v)
      #define VECTOR_NORM2(v) dot((v).s012, (v).s012)
      #define CLIP_TO_VECTOR(v) ((v).xyzw)
     #else
      #error Invalid dimension, only 2d and 3d is supported.
     #endif

     #if dimension == 2
       #define DIMENSIONALITY_SWITCH(case2d, case3d) case2d
     #elif dimension == 3
       #define DIMENSIONALITY_SWITCH(case2d, case3d) case3d
     #else
       #error Invalid dimension.
     #endif

    )"
  )
};

template<class Tree_type>
class tree_configuration
{
public:
  QCL_MAKE_MODULE(tree_configuration)

  using type_system = typename Tree_type::type_system;
  using node_type0 = typename Tree_type::node_type0;
  using node_type1 = typename Tree_type::node_type1;

private:
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(configuration<type_system>)
    QCL_IMPORT_TYPE(node_type0)
    QCL_IMPORT_TYPE(node_type1)
    QCL_RAW()
  )
};

}

#endif
