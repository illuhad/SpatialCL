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

#include <iostream>
#include <random>


#include <SpatialCL/particle_bvh_sfc_tree.hpp>

void print_vector(const std::vector<cl_float4>& data, std::size_t begin, std::size_t size)
{
  std::cout << "[";
  for(std::size_t i = begin; i < data.size() && (i < begin+size); ++i)
  {
    std::cout << "("
              << data[i].s[0] << ", "
              << data[i].s[1] << ", "
              << data[i].s[2] << ")";
    if(i != data.size() - 1 && i != (begin+size-1))
      std::cout << ",";
    std::cout << std::endl;
  }
  std::cout << "]" << std::endl;
}

int main(int argc, char** argv)
{
  qcl::environment env;
  const cl::Platform& platform = env.get_platform_by_preference({"NVIDIA",
                                                                 "AMD",
                                                                 "Intel"});
  qcl::global_context_ptr global_ctx = env.create_global_context(platform, CL_DEVICE_TYPE_GPU);

  qcl::device_context_ptr ctx = global_ctx->device();

  std::cout << "Using OpenCL device:\n";
  std::cout << "  Vendor:      " << ctx->get_device_vendor() << std::endl;
  std::cout << "  Device name: " << ctx->get_device_name() << std::endl;
  std::cout << "via Platform:\n";
  std::cout << "  Vendor: " << env.get_platform_vendor(platform) << std::endl;
  std::cout << "  Name:   " << env.get_platform_name(platform) << std::endl;

  std::vector<cl_float4> particles;

  std::mt19937 generator(1245);
  std::uniform_real_distribution<float> distribution;

  for(std::size_t i = 0; i < 128; ++i)
  {
    cl_float4 p;
    p.s[0] = distribution(generator);
    p.s[1] = distribution(generator);
    p.s[2] = distribution(generator);
    p.s[3] = 1.0f;
    particles.push_back(p);
  }

  spatialcl::hilbert_bvh_sp3d_tree<3> gpu_tree{ctx, particles};

  cl::Buffer bbox_min_corners = gpu_tree.get_bbox_min_corners();
  cl::Buffer bbox_max_corners = gpu_tree.get_bbox_max_corners();

  std::vector<cl_float4> host_min_corners(gpu_tree.get_num_nodes());
  std::vector<cl_float4> host_max_corners(gpu_tree.get_num_nodes());
  std::vector<cl_float4> sorted_particles(particles.size());

  ctx->memcpy_d2h<cl_float4>(host_min_corners.data(), bbox_min_corners, gpu_tree.get_num_nodes());
  ctx->memcpy_d2h<cl_float4>(host_max_corners.data(), bbox_max_corners, gpu_tree.get_num_nodes());
  ctx->memcpy_d2h<cl_float4>(sorted_particles.data(), gpu_tree.get_sorted_particles(), particles.size());


  std::cout << "l6min=";
  print_vector(host_min_corners,0,64);
  std::cout << "l5min=";
  print_vector(host_min_corners,64,32);
  std::cout << "l4min=";
  print_vector(host_min_corners,64+32,16);
  std::cout << "l3min=";
  print_vector(host_min_corners,64+32+16,8);
  std::cout << "l2min=";
  print_vector(host_min_corners,64+32+16+8,4);
  std::cout << "l1min=";
  print_vector(host_min_corners,64+32+16+8+4,2);
  std::cout << "l0min=";
  print_vector(host_min_corners,64+32+16+8+4+2,1);

  std::cout << "l6max=";
  print_vector(host_max_corners,0,64);
  std::cout << "l5max=";
  print_vector(host_max_corners,64,32);
  std::cout << "l4max=";
  print_vector(host_max_corners,64+32,16);
  std::cout << "l3max=";
  print_vector(host_max_corners,64+32+16,8);
  std::cout << "l2max=";
  print_vector(host_max_corners,64+32+16+8,4);
  std::cout << "l1max=";
  print_vector(host_max_corners,64+32+16+8+4,2);
  std::cout << "l0max=";
  print_vector(host_max_corners,64+32+16+8+4+2,1);

  std::cout << "particles=";
  print_vector(sorted_particles, 0, sorted_particles.size());
}
