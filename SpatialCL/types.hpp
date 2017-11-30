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

#ifndef TYPES_HPP
#define TYPES_HPP

#include <QCL/qcl.hpp>

namespace spatialcl {

template<class T, std::size_t Dimension>
struct cl_vector_type
{
};

#define DECLARE_VECTOR_TYPE(type, dimension, val) \
  template<> struct cl_vector_type<type, dimension>{ using value = val; }

DECLARE_VECTOR_TYPE(float, 2, cl_float2);
DECLARE_VECTOR_TYPE(float, 3, cl_float4);
DECLARE_VECTOR_TYPE(float, 4, cl_float4);
DECLARE_VECTOR_TYPE(float, 8, cl_float8);
DECLARE_VECTOR_TYPE(float, 16, cl_float16);

DECLARE_VECTOR_TYPE(double, 2, cl_double2);
DECLARE_VECTOR_TYPE(double, 3, cl_double4);
DECLARE_VECTOR_TYPE(double, 4, cl_double4);
DECLARE_VECTOR_TYPE(double, 8, cl_double8);
DECLARE_VECTOR_TYPE(double, 16, cl_double16);

DECLARE_VECTOR_TYPE(int, 2, cl_int2);
DECLARE_VECTOR_TYPE(int, 3, cl_int4);
DECLARE_VECTOR_TYPE(int, 4, cl_int4);
DECLARE_VECTOR_TYPE(int, 8, cl_int8);
DECLARE_VECTOR_TYPE(int, 16, cl_int16);

DECLARE_VECTOR_TYPE(unsigned, 2, cl_uint2);
DECLARE_VECTOR_TYPE(unsigned, 3, cl_uint4);
DECLARE_VECTOR_TYPE(unsigned, 4, cl_uint4);
DECLARE_VECTOR_TYPE(unsigned, 8, cl_uint8);
DECLARE_VECTOR_TYPE(unsigned, 16, cl_uint16);

}

#endif
