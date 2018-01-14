
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


#include <iostream>
#include <vector>
#include <string>
#include <fstream>

#include <QCL/qcl.hpp>

#include "model.hpp"
#include "nbody.hpp"
#include "particle_renderer.hpp"

using scalar = float;
using particle_type =
  typename nbody::nbody_simulation<scalar>::particle_type;


constexpr scalar final_time = 100.f;
constexpr scalar dt = 0.1f;
constexpr scalar opening_angle = 0.5f;

constexpr std::array<scalar,3> viewport_center{0.0f, 0.0f, 0.0f};
constexpr std::array<scalar,2> viewport_width{400.f, 400.f};

/// Creates the initial model: two random particle clouds
void create_model(std::vector<particle_type>& particles)
{
  std::vector<particle_type> cloud0, cloud1;

  nbody::model::random_particle_cloud<float> cloud0_sampler{
    {0.0f, 100.0f, 0.0f}, // position
    {10.0f, 15.0f, 11.0f}, // stddev position
    1.0f, // particle mass
    0.1f, // stddev particle mass
    {0.0f, -26.0f, 5.0f}, // velocity
    {5.0f, 20.0f, 12.f} // stddev velocity
  };

  nbody::model::random_particle_cloud<float> cloud1_sampler{
    {50.0f, 5.0f, 0.0f}, // position
    {17.0f, 7.0f, 5.0f}, // stddev position
    1.3f, // particle mass
    0.2f, // stddev particle mass
    {-5.f, 20.0f, 1.0f}, // velocity
    {18.0f, 11.f, 8.f} // stddev velocity
  };

  cloud0_sampler.sample(100000, cloud0);
  cloud1_sampler.sample(150000, cloud1);
  // Concatenate data
  particles.clear();
  for(const auto& p : cloud0)
  {
    particles.push_back(p);
  }
  for(const auto& p : cloud1)
  {
    particles.push_back(p);
  }
}

void save_state(const qcl::device_array<particle_type>& particles,
                std::size_t step_id)
{
  std::string filename = "nbody_"+std::to_string(step_id)+".dat";

  std::vector<particle_type> data;
  particles.read(data);

  std::ofstream file{filename.c_str(), std::ios::trunc};

  if(file.is_open())
  {
    file << "# Format: x y z mass\n";
    for(const particle_type& p : data)
    {
      file << p.s[0] << "\t "
           << p.s[1] << "\t "
           << p.s[2] << "\t "
           << p.s[3] << "\n";
    }
  }
  else
    std::cout << "Warning: could not save state" << std::endl;
}

int main()
{
  try
  {
    qcl::environment env;
    const cl::Platform& platform =
        env.get_platform_by_preference({"NVIDIA",
                                        "AMD",
                                        "Intel"});

    qcl::global_context_ptr global_ctx =
        env.create_global_context(platform,
                                  CL_DEVICE_TYPE_GPU);

    if(global_ctx->get_num_devices() == 0)
      throw std::runtime_error("No available OpenCL devices!");

    qcl::device_context_ptr ctx = global_ctx->device();

    std::cout << "Using device: " << ctx->get_device_name()
              << " (Vendor: " << ctx->get_device_vendor()
              << ")" << std::endl;

    // Create model: two random particle clouds
    std::vector<particle_type> particles;
    create_model(particles);

    // Copy particles to the GPU
    qcl::device_array<particle_type> device_particles{ctx, particles};

    nbody::nbody_simulation<scalar> simulation{ctx, device_particles};
    nbody::particle_renderer<scalar> renderer{ctx, 512, 512};

    std::size_t step_id = 0;
    for(;simulation.get_current_time() < final_time;
        ++step_id)
    {
      std::cout << "t = " << simulation.get_current_time() << std::endl;
      std::cout << "  Time step..." << std::endl;
      simulation.time_step(opening_angle, dt);

      std::cout << "  Rendering particles..." << std::endl;
      renderer.render(device_particles,
                      viewport_center, viewport_width);
      std::cout << "  Saving image..." << std::endl;

      renderer.save_png(std::string("nbody_")
                        +std::to_string(step_id)
                        +".png");

      if(step_id % 100 == 0)
      {
        // Save state as text file
        std::cout << "  Saving state..." << std::endl;
        save_state(device_particles, step_id);
      }

      std::cout << "  Step complete." << std::endl;

    }
    std::cout << "Saving final state..." << std::endl;
    save_state(device_particles, step_id);

  }
  catch(std::exception& e)
  {
    std::cout << "Fatal error: " << e.what() << std::endl;
  }
  return 0;
}

