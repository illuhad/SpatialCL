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


#ifndef CL_UTILS_HPP
#define CL_UTILS_HPP

#define WITH_BOOST_COMPUTE_COMPAT
#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>

namespace cl_utils {

#ifdef NODEBUG
 const int CL_NODEBUG = 1;
#else
 const int CL_NODEBUG = 0;
#endif

QCL_STANDALONE_MODULE(debug)
QCL_STANDALONE_SOURCE
(
  QCL_IMPORT_CONSTANT(CL_NODEBUG)
  R"(
  #if CL_NODEBUG == 0
    #define NAMED_ASSERT(name, cond) \
      if(!(cond)) \
        printf("Assert failed: %s, Line %d", name, __LINE__);
    #define ASSERT(cond) \
      if(!(cond)) \
        printf("Assert failed: %s, Line %d\n", __FILE__, __LINE__);
  #else
    #define ASSERT(cond)
    #define NAMED_ASSERT(name, cond)
  #endif
  )"
)

}

#endif
