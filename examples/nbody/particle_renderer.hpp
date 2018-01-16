
/*
 * This file is part of SpatialCL, a library for the spatial processing of
 * particles.
 *
 * Copyright (c) 2017, 2018 Aksel Alpay
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

#ifndef PARTICLE_RENDERER_HPP
#define PARTICLE_RENDERER_HPP

#include <cassert>
#include <array>

#include <QCL/qcl.hpp>
#include <QCL/qcl_array.hpp>
#include <QCL/qcl_module.hpp>
#include <QCL/qcl_boost_compat.hpp>

#include <SpatialCL/configuration.hpp>

#include <boost/compute/algorithm/max_element.hpp>

#include <png++/png.hpp>

#include "nbody_tree.hpp"

namespace nbody {

template<class Scalar>
class particle_renderer
{
public:
  QCL_MAKE_MODULE(particle_renderer)

  using particle_type =
    typename spatialcl::configuration<nbody_type_descriptor<Scalar>>::particle_type;


  particle_renderer(const qcl::device_context_ptr& ctx,
                    std::size_t bins_x,
                    std::size_t bins_y)
    : _ctx{ctx},
      _bins_x{bins_x},
      _bins_y{bins_y},
      _rgb_histogram{ctx, bins_x * bins_y}
  {
    assert(bins_x > 0);
    assert(bins_y > 0);
  }

  void render(const qcl::device_array<particle_type>& particles,
              const std::array<Scalar,3>& render_plane_center,
              const std::array<Scalar,2>& render_plane_size)
  {
    // We do not use enqueueFillBuffer to reset the buffer to 0
    // because the pocl cuda backend currently does not support it.
    cl_int err = reset_buffer(_ctx,
                              cl::NDRange{_bins_x * _bins_y},
                              cl::NDRange{128})(_rgb_histogram,
                                                static_cast<cl_ulong>(_bins_x*_bins_y));

    qcl::check_cl_error(err, "Could not reset histogram buffer");

    Scalar cell_size_x = render_plane_size[0] / static_cast<Scalar>(_bins_x);
    Scalar cell_size_y = render_plane_size[1] / static_cast<Scalar>(_bins_y);

    Scalar min_x, min_y;
    min_x = render_plane_center[0] - render_plane_size[0] / 2.0f;
    min_y = render_plane_center[1] - render_plane_size[1] / 2.0f;

    err = count_masses_per_bin(_ctx,
                               cl::NDRange{particles.size()},
                               cl::NDRange{128})(
          particles,
          particles.size(),
          min_x, min_y,
          cell_size_x, cell_size_y,
          static_cast<cl_ulong>(_bins_x), static_cast<cl_ulong>(_bins_y),
          _rgb_histogram);

    qcl::check_cl_error(err, "Could not enqueue count_masses_per_bin kernel");

    // Determine maximum bin value
    boost::compute::command_queue boost_queue{_ctx->get_command_queue().get()};
    auto max_iterator = boost::compute::max_element(
          qcl::create_buffer_iterator<cl_uint>(_rgb_histogram.get_buffer(), 0),
          qcl::create_buffer_iterator<cl_uint>(_rgb_histogram.get_buffer(), _bins_x*_bins_y),
          boost_queue);

    cl_uint maximum_mass = max_iterator.read(boost_queue);
    if(maximum_mass == 0)
      maximum_mass = 1;
    // Calculate rgb values
    err = mass_to_rgb(_ctx,
                      cl::NDRange{_bins_x * _bins_y},
                      cl::NDRange{128})(
          _rgb_histogram,
          static_cast<cl_ulong>(_bins_x * _bins_y),
          maximum_mass);

    qcl::check_cl_error(err, "Could not enqueue mass_to_rgb kernel");

    err = _ctx->get_command_queue().finish();

    qcl::check_cl_error(err, "Error while waiting for mass_to_rgb "
                             "kernel to finish");
  }

  void read_rendered_image(std::vector<cl_uint>& result) const
  {
    _rgb_histogram.read(result);
  }

  void save_png(const std::string& filename) const
  {
    png::image<png::rgb_pixel> image{
      static_cast<png::uint_32>(_bins_x),
      static_cast<png::uint_32>(_bins_y)
    };

    std::vector<cl_uint> raw_data;
    read_rendered_image(raw_data);

    for (std::size_t y = 0; y < image.get_height(); ++y)
    {
      for (std::size_t x = 0; x < image.get_width(); ++x)
      {
        cl_uint pixel = raw_data[y * _bins_x + x];
        cl_uchar* pixel_bytes =
            reinterpret_cast<cl_uchar*>(&pixel);


        image[y][x] = png::rgb_pixel(pixel_bytes[0],
                                     pixel_bytes[1],
                                     pixel_bytes[2]);
      }
    }

    image.write(filename.c_str());
  }


  std::size_t get_num_bins_x() const
  {
    return _bins_x;
  }

  std::size_t get_num_bins_y() const
  {
    return _bins_y;
  }

  static constexpr Scalar mass_quantum = 0.01f;

private:
  QCL_ENTRYPOINT(reset_buffer)
  QCL_ENTRYPOINT(count_masses_per_bin)
  QCL_ENTRYPOINT(mass_to_rgb)
  QCL_MAKE_SOURCE(
    QCL_IMPORT_TYPE(particle_type)
    QCL_IMPORT_TYPE(Scalar)
    QCL_IMPORT_CONSTANT(mass_quantum)
    QCL_RAW(
      __kernel void reset_buffer(__global uint* buffer,
                                 ulong num_entries)
      {
        size_t tid = get_global_id(0);

        if(tid < num_entries)
          buffer[tid] = 0;
      }


      __kernel void count_masses_per_bin(__global particle_type* particles,
                                         ulong num_particles,
                                         Scalar grid_min_x,
                                         Scalar grid_min_y,
                                         Scalar cell_size_x,
                                         Scalar cell_size_y,
                                         ulong num_bins_x,
                                         ulong num_bins_y,
                                         __global volatile uint* output)
      {
        size_t tid = get_global_id(0);

        for(; tid < num_particles;
            tid += get_global_size(0))
        {
          particle_type p = particles[tid];

          uint discretized_mass = convert_uint_rte(p.s3 / mass_quantum);

          int bin_x = convert_int_rtn((p.s0 - grid_min_x) / cell_size_x);
          int bin_y = convert_int_rtn((p.s1 - grid_min_y) / cell_size_y);

          if(bin_x >= 0 && bin_x < num_bins_x)
          {
            if(bin_y >= 0 && bin_y < num_bins_y)
            {
              ulong bin_index = bin_y * num_bins_x + bin_x;
              atomic_add(&(output[bin_index]), discretized_mass);
            }
          }
        }
      }

      __kernel void mass_to_rgb(__global uint* discretized_masses,
                                ulong total_num_bins,
                                uint maximum_mass)
      {
        size_t tid = get_global_id(0);

        if(tid < total_num_bins)
        {

          // We adopt a sqrt scale for better visibility of
          // low density structures
          Scalar relative_mass = sqrt((Scalar)(discretized_masses[tid])) /
                                   sqrt((Scalar)maximum_mass);

          uint rgb = 0;

          uchar color_value = (uchar)(255 * clamp(relative_mass, 0.0f, 1.0f));
          for(int i = 0; i < 4; ++i)
          {
            rgb <<= 8;
            rgb |= (uint)color_value;
          }

          discretized_masses[tid] = rgb;
        }
      }

    )
  )

  qcl::device_context_ptr _ctx;

  std::size_t _bins_x;
  std::size_t _bins_y;

  qcl::device_array<cl_uint> _rgb_histogram;
};

}

#endif
