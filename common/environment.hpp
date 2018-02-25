
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

#ifndef COMMON_ENVIRONMENT_HPP
#define COMMON_ENVIRONMENT_HPP

#include <QCL/qcl.hpp>
#include <iostream>

namespace common {

class environment
{
public:
  environment()
  {
    const cl::Platform& platform = _env.get_platform_by_preference({"pocl",
                                                                   "AMD",
                                                                   "Intel"});
    qcl::global_context_ptr global_ctx = _env.create_global_context(platform,
                                                                    CL_DEVICE_TYPE_GPU);
    if(global_ctx->get_num_devices() == 0)
      throw std::runtime_error("No available OpenCL devices!");

    _ctx = global_ctx->device();

    std::cout << "Using OpenCL device:\n";
    std::cout << "  Vendor:      " << _ctx->get_device_vendor() << std::endl;
    std::cout << "  Device name: " << _ctx->get_device_name() << std::endl;
    std::cout << "via Platform:\n";
    std::cout << "  Vendor: " << _env.get_platform_vendor(platform) << std::endl;
    std::cout << "  Name:   " << _env.get_platform_name(platform) << std::endl;

  }

  const qcl::device_context_ptr& get_device_context() const
  {
    return _ctx;
  }

private:
  qcl::environment _env;
  qcl::device_context_ptr _ctx;
};

}

#endif
