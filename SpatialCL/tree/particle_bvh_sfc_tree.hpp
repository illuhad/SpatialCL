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

#ifndef PARTICLE_BVH_SFC_TREE
#define PARTICLE_BVH_SFC_TREE

#include <memory>
#include <string>

#include "particle_bvh_tree.hpp"
#include "../bit_manipulation.hpp"
#include "../sfc_position_generator.hpp"
#include "../zcurve.hpp"
#include "../hilbert_curve.hpp"

namespace spatialcl {

template<class Type_descriptor, class Space_filling_curve>
class sfc_sort_key_generator
{
public:
  using key_type = cl_ulong;
  using particle_type = typename configuration<Type_descriptor>::particle_type;
  using vector_type = typename configuration<Type_descriptor>::vector_type;
  static constexpr std::size_t particle_dimension = Type_descriptor::particle_dimension;
  static constexpr std::size_t dimension = Type_descriptor::dimension;



  void operator()(const qcl::device_context_ptr& ctx,
                  const cl::Buffer& particles,
                  std::size_t num_particles,
                  const cl::Buffer& particle_sort_keys_out) const
  {
    vector_type min, max;
    this->get_particle_extent(ctx, particles, num_particles, min, max);

    Space_filling_curve curve;

    curve(ctx,
          min,
          max,
          particles,
          static_cast<cl_ulong>(num_particles),
          particle_sort_keys_out);
  }

private:

  void get_particle_extent(const qcl::device_context_ptr& ctx,
                           const cl::Buffer& particles,
                           std::size_t num_particles,
                           vector_type& min,
                           vector_type& max) const
  {

    using boost_particle_type = typename qcl::to_boost_vector_type<particle_type>::type;


    BOOST_COMPUTE_FUNCTION(int, vector_less_x, (boost_particle_type a, boost_particle_type b),
    { return a.x < b.x; });
    BOOST_COMPUTE_FUNCTION(int, vector_less_y, (boost_particle_type a, boost_particle_type b),
    { return a.y < b.y; });
    BOOST_COMPUTE_FUNCTION(int, vector_less_z, (boost_particle_type a, boost_particle_type b),
    { return a.z < b.z; });

    boost::compute::command_queue boost_queue{ctx->get_command_queue().get()};
    auto begin = qcl::create_buffer_iterator<boost_particle_type>(particles,0);
    auto end = qcl::create_buffer_iterator<boost_particle_type>(particles,num_particles);

    auto result_x = boost::compute::minmax_element(begin, end, vector_less_x,
                                                   boost_queue);
    auto result_y = boost::compute::minmax_element(begin, end, vector_less_y,
                                                   boost_queue);
    auto result_z = result_y;
    if(dimension >= 3)
      result_z = boost::compute::minmax_element(begin, end, vector_less_z,
                                                boost_queue);

    min.s[0] = result_x.first.read(boost_queue)[0];
    max.s[0] = result_x.second.read(boost_queue)[0];

    min.s[1] = result_y.first.read(boost_queue)[1];
    max.s[1] = result_y.second.read(boost_queue)[1];

    if(dimension >= 3)
    {
      min.s[2] = result_z.first.read(boost_queue)[2];
      max.s[2] = result_z.second.read(boost_queue)[2];
    }

  }


  static constexpr std::size_t local_size = 256;
};

template<class Type_descriptor>
using zcurve_sort_key_generator =
  sfc_sort_key_generator<Type_descriptor, space_filling_curve::zcurve<Type_descriptor>>;

template<class Type_descriptor>
using hilbert_sort_key_generator =
  sfc_sort_key_generator<Type_descriptor, space_filling_curve::hilbert_curve<Type_descriptor>>;




template<class Key_generator>
class key_based_sorter
{
public:
  using key_type = typename Key_generator::key_type;
  using particle_type = typename Key_generator::particle_type;
  using boost_particle_type = typename qcl::to_boost_vector_type<particle_type>::type;

  void operator()(const qcl::device_context_ptr& ctx,
                  const cl::Buffer& particles,
                  std::size_t num_particles) const
  {
    cl::Buffer sort_keys;
    ctx->create_buffer<key_type>(sort_keys, num_particles);

    Key_generator sort_key_generator;
    sort_key_generator(ctx, particles, num_particles, sort_keys);

    boost::compute::command_queue boost_queue{ctx->get_command_queue().get()};
    boost::compute::sort_by_key(qcl::create_buffer_iterator<key_type>(sort_keys,0),
                                qcl::create_buffer_iterator<key_type>(sort_keys,num_particles),
                                qcl::create_buffer_iterator<boost_particle_type>(particles, 0),
                                boost_queue);
  }
};

}
/*
class kd_sorter
{
public:
  void operator()(const qcl::device_context_ptr& ctx,
                  const cl::Buffer& particles,
                  std::size_t num_particles) const
  {
    BOOST_COMPUTE_FUNCTION(int, vector4_less_x, (boost_float4 a, boost_float4 b),
    { return a.x < b.x; });
    BOOST_COMPUTE_FUNCTION(int, vector4_less_y, (boost_float4 a, boost_float4 b),
    { return a.y < b.y; });
    BOOST_COMPUTE_FUNCTION(int, vector4_less_z, (boost_float4 a, boost_float4 b),
    { return a.z < b.z; });

    boost::compute::command_queue boost_queue{ctx->get_command_queue().get()};

    std::size_t current_num_particles_per_node = num_particles;

    while(current_num_particles_per_node > _group_size)
    {
      boost::compute::nth_element(qcl::create_buffer_iterator<boost::compute::float4_>(particles, 0),
                                  qcl::create_buffer_iterator<boost::compute::float4_>(particles, num_particles/2),
                                  qcl::create_buffer_iterator<boost::compute::float4_>(particles, num_particles),
                                  vector4_less_x,
                                  boost_queue);
    }
  }

private:
  static constexpr std::size_t _group_size = 512;
};

class kd_sort_key_generator
{
public:
  using key_type = cl_ulong;

  void operator()(const qcl::device_context_ptr& ctx,
                  const cl::Buffer& particles,
                  std::size_t num_particles,
                  cl::Buffer& particle_sort_keys_out) const
  {
    using boost_float4 = boost::compute::float4_;

    cl::Buffer x_sort_indices, y_sort_indices, z_sort_indices;

    BOOST_COMPUTE_FUNCTION(int, vector4_less_x, (boost_float4 a, boost_float4 b),
    { return a.x < b.x; });
    BOOST_COMPUTE_FUNCTION(int, vector4_less_y, (boost_float4 a, boost_float4 b),
    { return a.y < b.y; });
    BOOST_COMPUTE_FUNCTION(int, vector4_less_z, (boost_float4 a, boost_float4 b),
    { return a.z < b.z; });

    // First, obtain the indices where a particle would be moved to in a sort along the
    // x,y and z axes
    this->obtain_sort_indices(ctx, particles, num_particles, vector4_less_x, x_sort_indices);
    this->obtain_sort_indices(ctx, particles, num_particles, vector4_less_y, y_sort_indices);
    this->obtain_sort_indices(ctx, particles, num_particles, vector4_less_z, z_sort_indices);
    // The final kd tree sort key can now be obtained by interleaving the x,y,z sort indices
    // in a bitwise manner.

    ctx->register_source_module<kd_tree_interleave_sort_indices>(
          std::vector<std::string>{"kd_tree_interleave_sort_indices"});
    qcl::kernel_ptr kernel = ctx->get_kernel("kd_tree_interleave_sort_indices");
    qcl::kernel_argument_list args{kernel};

    args.push(x_sort_indices);
    args.push(y_sort_indices);
    args.push(z_sort_indices);
    args.push(static_cast<cl_ulong>(num_particles));
    args.push(particle_sort_keys_out);

    cl_int err = ctx->enqueue_ndrange_kernel(kernel,
                                             cl::NDRange{num_particles},
                                             cl::NDRange{128});
    qcl::check_cl_error(err, "Could not enqueue kd_tree_interleave_sort_indices"
                             " interleave kernel");

    err = ctx->get_command_queue().finish();
    qcl::check_cl_error(err, "Error while waiting for the kd_tree_interleave_sort_indices"
                             " to finish.");

  }

private:
  QCL_MODULE_BEGIN(kd_tree_interleave_sort_indices);
  QCL_INCLUDE_MODULE(bit_manipulation);
  cl_source(
  __kernel void kd_tree_interleave_sort_indices(__global ulong* x_sort_indices,
                                                __global ulong* y_sort_indices,
                                                __global ulong* z_sort_indices,
                                                ulong num_elements,
                                                __global ulong* out)
  {
    size_t gid = get_global_id(0);

    if(gid < num_elements)
    {
      ulong x_idx = x_sort_indices[gid];
      ulong y_idx = y_sort_indices[gid];
      ulong z_idx = z_sort_indices[gid];
      out[gid] = interleave_bits3(x_idx, y_idx, z_idx);

    }
  }
  );
  QCL_MODULE_END();

  QCL_MODULE_BEGIN(kd_tree_generate_sequence);
  cl_source(
  __kernel void kd_tree_generate_sequence(__global ulong* out, ulong num_elements)
  {
    size_t gid = get_global_id(0);
    if(gid < num_elements)
      out[gid] = gid;
  }
  );
  QCL_MODULE_END();

  QCL_MODULE_BEGIN(kd_tree_invert_indices);
  cl_source(
  __kernel void kd_tree_invert_indices(__global ulong* sorted_indices,
                                       ulong num_elements,
                                       __global ulong* out)
  {
    size_t gid = get_global_id(0);
    if(gid < num_elements)
    {
      ulong idx = sorted_indices[gid];
      out[idx] = gid;
    }
  }
  );
  QCL_MODULE_END();

  void run_generate_sequence_kernel(const qcl::device_context_ptr& ctx,
                                    const cl::Buffer& out,
                                    std::size_t num_elements) const
  {
    qcl::kernel_ptr kernel = ctx->get_kernel("kd_tree_generate_sequence");
    qcl::kernel_argument_list args{kernel};
    args.push(out);
    args.push(static_cast<cl_ulong>(num_elements));

    cl_int err = ctx->enqueue_ndrange_kernel(kernel,
                                             cl::NDRange{num_elements},
                                             cl::NDRange{128});
    qcl::check_cl_error(err, "Could not enqueue kd_tree_generate_sequence kernel");
  }

  void run_invert_indices_kernel(const qcl::device_context_ptr& ctx,
                                 const cl::Buffer& sorted_indices,
                                 std::size_t num_elements,
                                 cl::Buffer& out) const
  {
    qcl::kernel_ptr kernel = ctx->get_kernel("kd_tree_invert_indices");
    qcl::kernel_argument_list args{kernel};

    args.push(sorted_indices);
    args.push(static_cast<cl_ulong>(num_elements));
    args.push(out);

    cl_int err = ctx->enqueue_ndrange_kernel(kernel, cl::NDRange{num_elements}, cl::NDRange{128});
    qcl::check_cl_error(err, "Could not enqueue kd_tree_invert_indices kernel");
  }

  template<class Comparator>
  void obtain_sort_indices(const qcl::device_context_ptr& ctx,
                           const cl::Buffer& particles,
                           std::size_t num_particles,
                           Comparator compare,
                           cl::Buffer& out) const
  {
    // Make sure source module is compiled
    ctx->register_source_module<kd_tree_generate_sequence>(
          std::vector<std::string>{"kd_tree_generate_sequence"});

    using boost_float4 = boost::compute::float4_;

    cl::Buffer sorted_indices;
    ctx->create_buffer<cl_ulong>(sorted_indices, num_particles);
    run_generate_sequence_kernel(ctx, sorted_indices, num_particles);

    // Sort indices
    boost::compute::command_queue boost_queue{ctx->get_command_queue().get()};

    cl::Buffer temp_particles;
    ctx->create_buffer<cl_float4>(temp_particles, num_particles);
    ctx->get_command_queue().enqueueCopyBuffer(particles, temp_particles, 0, 0, num_particles*sizeof(cl_float4));

    boost::compute::sort_by_key(qcl::create_buffer_iterator<boost_float4>(temp_particles, 0),
                                qcl::create_buffer_iterator<boost_float4>(temp_particles, num_particles),
                                qcl::create_buffer_iterator<cl_ulong>(sorted_indices, 0),
                                compare,
                                boost_queue);

    // Map indices back to obtain sorted position
    ctx->register_source_module<kd_tree_invert_indices>(std::vector<std::string>{"kd_tree_invert_indices"});

    ctx->create_buffer<cl_ulong>(out, num_particles);
    run_invert_indices_kernel(ctx, sorted_indices, num_particles, out);

  }


};*/


#endif
