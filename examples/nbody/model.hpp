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

#ifndef NBODY_MODEL_HPP
#define NBODY_MODEL_HPP

#include <array>
#include <random>
#include <cmath>

#include "nbody.hpp"

#include <QCL/qcl_array.hpp>

namespace nbody {
namespace model {

template<class Scalar>
class random_particle_cloud
{
public:
  using host_vector3d = std::array<Scalar,3>;

  random_particle_cloud(const host_vector3d& position,
                        const host_vector3d& width,
                        Scalar mean_mass,
                        Scalar mass_distribution_width,
                        const host_vector3d& mean_velocity,
                        const host_vector3d& velocity_distribution_width)
    : _generator{generate_seed()}
  {
    for(std::size_t i = 0; i < 3; ++i)
    {
      _means[i] = position[i];
      _stddevs[i] = width[i];
    }

    _means[3] = mean_mass;
    _stddevs[3] = mass_distribution_width;

    for(std::size_t i = 0; i < 3; ++i)
    {
      _means[i+4] = mean_velocity[i];
      _stddevs[i+4] = velocity_distribution_width[i];
    }
  }

  using particle_type = typename nbody_simulation<Scalar>::particle_type;

  void sample(std::size_t n,
              std::vector<particle_type>& out)
  {
    out.resize(n);

    std::array<std::normal_distribution<Scalar>, 7> distributions;
    for(std::size_t i = 0; i < distributions.size(); ++i)
    {
      distributions[i] = std::normal_distribution<Scalar>{
          _means[i],
          _stddevs[i]
      };
    }

    for(std::size_t i = 0; i < n; ++i)
    {
      particle_type sampled_particle;
      for(std::size_t j = 0; j < distributions.size(); ++j)
      {
        sampled_particle.s[j] = distributions[j](_generator);
      }
      // Make sure the mass is positive
      sampled_particle.s[3] = std::abs(sampled_particle.s[3]);
      out[i] = sampled_particle;
    }
  }

private:
  static std::size_t generate_seed()
  {
    std::random_device rd;
    return rd();
  }

  std::array<Scalar, 7> _means;
  std::array<Scalar, 7> _stddevs;

  std::mt19937 _generator;
};

}
}

#endif
