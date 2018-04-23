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

#ifndef PARTICLE_TREE
#define PARTICLE_TREE

#include <QCL/qcl.hpp>
#include <QCL/qcl_array.hpp>

#include <boost/compute.hpp>
#include <cassert>
#include <functional>
#include "binary_tree.hpp"
#include "../configuration.hpp"
#include "../cl_utils.hpp"


namespace spatialcl {

/// Base class for particle trees. Does not calculate
/// the content of the tree nodes (this should be done
/// by derived classes)
/// \tparam Particle_sorter A functor to sort the particles
/// spatially
/// \tparam Type_descriptor The type descriptor for the desired
/// type system (dimensionality, single/double precision etc)
/// \tparam Node_data_type0 Data type of the first value of the data
/// per node
/// \tparam Node_data_type1 Data type of the second value of the data
/// per node
template<class Particle_sorter,
         class Type_descriptor,
         class Node_data_type0,
         class Node_data_type1>
class particle_tree
{
public:

  using particle_type = typename configuration<Type_descriptor>::particle_type;
  using vector_type   = typename configuration<Type_descriptor>::vector_type;
  using boost_particle = typename qcl::to_boost_vector_type<particle_type>::type;
  using node_type0 = Node_data_type0;
  using node_type1 = Node_data_type1;

  using type_system = Type_descriptor;

  particle_tree(const qcl::device_context_ptr& ctx,
                const std::vector<particle_type>& particles,
                const Particle_sorter& sorter = Particle_sorter{})
    : _ctx{ctx},
      _num_particles{particles.size()}
  {
    _ctx->create_buffer<particle_type>(this->_sorted_particles, particles.size());
    _ctx->memcpy_h2d(this->_sorted_particles, particles.data(), particles.size());

    this->init_tree(sorter);
  }

  particle_tree(const qcl::device_context_ptr& ctx,
                const cl::Buffer& particles,
                std::size_t num_particles,
                const Particle_sorter& sorter = Particle_sorter{})
    : _ctx{ctx},
      _sorted_particles{particles},
      _num_particles{num_particles}
  {
    this->init_tree(sorter);
  }

  particle_tree(const qcl::device_context_ptr& ctx,
                const qcl::device_array<particle_type>& particles,
                const Particle_sorter& sorter = Particle_sorter{})
    : particle_tree{ctx, particles.get_buffer(), particles.size(), sorter}
  {}

  virtual ~particle_tree(){}

  std::size_t get_num_nodes() const
  {
    return this->_effective_num_particles-1;
  }

  std::size_t get_effective_num_levels() const
  {
    return this->_num_levels;
  }

  std::size_t get_num_node_levels() const
  {
    return get_effective_num_levels() - 1;
  }

  const cl::Buffer& get_sorted_particles() const
  {
    return this->_sorted_particles;
  }

  std::size_t get_num_particles() const
  {
    return this->_num_particles;
  }

  std::size_t get_effective_num_particles() const
  {
    return this->_effective_num_particles;
  }

  const qcl::device_context_ptr& get_device_context() const
  {
    return _ctx;
  }

  const cl::Buffer& get_node_values0() const
  {
    return _nodes0.get_buffer();
  }

  const cl::Buffer& get_node_values1() const
  {
    return _nodes1.get_buffer();
  }


private:
  void init_tree(const Particle_sorter& sorter)
  {
    // First sort the particles spatially
    sorter(_ctx, _sorted_particles, _num_particles);

    // Calculate the required number of levels
    // and the effective number of particles
    _effective_num_particles = get_next_power_of_two(_num_particles);
    _num_levels = get_highest_set_bit(_effective_num_particles)+1;
#ifndef NODEBUG
    std::cout << "Building tree with "
              << _num_levels << " levels over "
              << _effective_num_particles << " effective particles and "
              << _num_particles << " real particles." << std::endl;
#endif

    _nodes0 = qcl::device_array<Node_data_type0>{_ctx, _effective_num_particles};
    _nodes1 = qcl::device_array<Node_data_type1>{_ctx, _effective_num_particles};

  }

  static unsigned get_highest_set_bit(uint64_t x)
  {
    unsigned result = 0;

    for(int i = 0; i < 64; ++i)
      if(x & (1ull << i))
        result = i;

    return result;
  }

  static uint64_t get_next_power_of_two(uint64_t x)
  {
    x--;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    x++;

    return x;
  }

  qcl::device_context_ptr _ctx;

  cl::Buffer _sorted_particles;

  unsigned _num_levels;

  std::size_t _num_particles;
  std::size_t _effective_num_particles;

  qcl::device_array<Node_data_type0> _nodes0;
  qcl::device_array<Node_data_type1> _nodes1;
};




}
#endif
