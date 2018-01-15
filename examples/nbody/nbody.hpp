
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

#ifndef NBODY_HPP
#define NBODY_HPP

#include <cassert>
#include <memory>
#include <array>

#include <QCL/qcl.hpp>
#include <QCL/qcl_module.hpp>
#include <QCL/qcl_array.hpp>

#include <SpatialCL/query/query_base.hpp>
#include <SpatialCL/query.hpp>

#include "nbody_tree.hpp"

namespace nbody {

template<class Scalar>
class nbody_query_handler : public spatialcl::query::basic_query
{
public:
  QCL_MAKE_MODULE(nbody_query_handler)

  /// \param evaluation_particles particles at whose location the
  /// acceleration should be calculated
  /// \param accelerations output buffer for the accelerations,
  /// must have \c num_particles entries of \c vector_type
  /// \param num_particles The number of particles in the \c
  /// evaluation_particles buffer.
  /// \param The opening angle for the tree walk
  nbody_query_handler(const cl::Buffer& evaluation_particles,
                      std::size_t num_particles,
                      Scalar opening_angle,
                      const cl::Buffer& accelerations)
    : _eval_particles{evaluation_particles},
      _accelerations{accelerations},
      _eval_num_particles{num_particles},
      _opening_angle_squared{opening_angle * opening_angle}
  {
    assert(opening_angle > 0.0f);
  }

  virtual void push_full_arguments(qcl::kernel_call& call) override
  {
    call.partial_argument_list(_eval_particles,
                               _accelerations,
                               static_cast<cl_ulong>(_eval_num_particles),
                               _opening_angle_squared);
  }

  virtual std::size_t get_num_independent_queries() const override
  {
    return _eval_num_particles;
  }

  virtual ~nbody_query_handler(){}
private:
  cl::Buffer _eval_particles;
  cl::Buffer _accelerations;

  std::size_t _eval_num_particles;
  Scalar _opening_angle_squared;
public:
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(spatialcl::configuration<nbody_type_descriptor<Scalar>>)
    QCL_PREPROCESSOR(define, gravitational_softening_squared 1.e-4f)
    QCL_PREPROCESSOR(define,
        dfs_node_selector(selection_result_ptr,
                          current_node_key_ptr,
                          node_index,
                          bbox_min_corner,
                          bbox_max_corner)
        {
          scalar node_width = bbox_max_corner.w;
          vector_type delta = bbox_min_corner - evaluation_position;
          scalar r2 = VECTOR_NORM2(delta);
          *selection_result_ptr = (node_width*node_width/r2 > opening_angle_squared);

          if(!(*selection_result_ptr))
          {
            // Evaluate monopole
            acceleration.s012 +=
              bbox_min_corner.w * normalize(delta.s012) / (r2+gravitational_softening_squared);
          }
        }
    )
    QCL_PREPROCESSOR(define,
        dfs_particle_processor(selection_result_ptr,
                               particle_idx,
                               current_particle)
        {
          // Add particle's contribution
          vector_type delta = PARTICLE_POSITION(current_particle)-evaluation_position;
          scalar r2 = VECTOR_NORM2(delta);
          vector_type contribution;
          contribution.s012 =
              current_particle.s3 * delta.s012 * rsqrt(r2) /
                       (r2 + gravitational_softening_squared);
          // This ensures that the force is not calculated from
          // the particle to itself
          if(particle_idx != get_query_id())
            acceleration += contribution;

          // For the relaxed dfs engine, we need to avoid remaining
          // at the lowest node indefinitely. We ensure this by
          // deselecting a particle if it is a right child, which
          // forces the query engine to go one level up and check
          // if the node there should be selected (i.e. if it can be approximated).
          // For the strict query engine, this makes no difference.
          //
          // For a right child, the index & 1 always evaluates to true,
          // because every odd index belongs to a right child.
          *selection_result_ptr = !(particle_idx & 1);
        }
    )
  R"(
    #define declare_full_query_parameter_set() \
      __global particle_type* evaluated_particles, \
      __global vector_type* accelerations, \
      ulong num_evaluated_particles, \
      scalar opening_angle_squared
  )"
  QCL_PREPROCESSOR(define,
    at_query_init()
      vector_type evaluation_position =
          PARTICLE_POSITION(evaluated_particles[get_query_id()]);
      vector_type acceleration = (vector_type)0.0f;
  )
  QCL_PREPROCESSOR(define,
    at_query_exit()
      accelerations[get_query_id()] = acceleration;
  )
  QCL_PREPROCESSOR(define,
    get_num_queries()
      num_evaluated_particles
  )
  )
};

template<class Scalar>
class nbody_integrator
{
public:
  nbody_integrator(const qcl::device_context_ptr& ctx)
    : _ctx{ctx}, _previous_dt{0.0f}, _t{0.0f}
  {}

  void advance(const cl::Buffer& particles,
               const cl::Buffer& accelerations,
               std::size_t num_particles,
               Scalar dt)
  {
    cl_int err =
        leapfrog_advance(_ctx,
                         cl::NDRange{num_particles},
                         cl::NDRange{256})(
            particles,
            accelerations,
            static_cast<cl_ulong>(num_particles),
            _previous_dt,
            dt);
    qcl::check_cl_error(err, "Could not enqueue leapfrog_advance kernel");

    err = _ctx->get_command_queue().finish();
    qcl::check_cl_error(err, "Error while waiting for leapfrog_advance kernel"
                             " to finish.");

    _t += dt;

    _previous_dt = dt;
  }

  Scalar get_current_time() const
  {
    return _t;
  }

private:
  qcl::device_context_ptr _ctx;
  Scalar _previous_dt;
  Scalar _t;

  QCL_MAKE_MODULE(nbody_integrator)
  QCL_ENTRYPOINT(leapfrog_advance)
  QCL_MAKE_SOURCE(
    QCL_INCLUDE_MODULE(spatialcl::configuration<nbody_type_descriptor<Scalar>>)
    QCL_RAW(
      __kernel void leapfrog_advance(__global particle_type* particles,
                                     __global vector_type* accelerations,
                                     ulong num_particles,
                                     scalar previous_dt,
                                     scalar dt)
      {
        size_t tid = get_global_id(0);

        if(tid < num_particles)
        {
          particle_type p = particles[tid];
          vector_type acceleration = accelerations[tid];

          // Bring v to the current state
          p.s456 += acceleration.s012 * previous_dt * 0.5f;

          // Calculate v_i+1/2
          p.s456 += acceleration.s012 * dt * 0.5f;
          // Update position
          p.s012 += p.s456 * dt;
          // We need to update the velocities  to i+1,
          // but for performance reasons we do this always
          // at the beginning of a step

          // Save result
          particles[tid] = p;
        }
      }

    )
  )
};



template<class Scalar>
class nbody_simulation
{
public:
  using nbody_tree_ptr = std::unique_ptr<nbody_tree<Scalar>>;

  using particle_type =
    typename spatialcl::configuration<nbody_type_descriptor<Scalar>>::particle_type;
  using vector_type =
    typename spatialcl::configuration<nbody_type_descriptor<Scalar>>::vector_type;
  using host_vector3d = std::array<Scalar,3>;

  static particle_type encode_particle(vector_type position,
                                       Scalar mass,
                                       vector_type velocity)
  {
    particle_type result;
    for(std::size_t i = 0; i < 3; ++i)
      result.s[i] = position.s[i];

    result.s[3] = mass;

    for(std::size_t i = 0; i < 3; ++i)
      result.s[i+4] = velocity.s[i];

    return result;
  }


  nbody_simulation(const qcl::device_context_ptr& ctx,
                   const qcl::device_array<particle_type>& initial_particles)
    : _ctx{ctx},
      _particles{initial_particles},
      _integrator{ctx},
      _acceleration{ctx, initial_particles.size()}
  {}

  void time_step(Scalar opening_angle,
                 Scalar dt = 0.1f)
  {
    // Calculate acceleration
    // - Build tree
    // -- Discard old tree
    _tree = nullptr;
    // -- Create new one
    _tree = nbody_tree_ptr{new nbody_tree<Scalar>{_ctx, _particles}};

    // - Query tree to obtain accelerations
    // -- Define queries
    using query_handler = nbody_query_handler<Scalar>;
    using query_engine =
      spatialcl::query::relaxed_dfs_query_engine<
        nbody_type_descriptor<Scalar>,
        query_handler
      >;

    query_engine engine;
    query_handler handler{
      _particles.get_buffer(),
      _particles.size(),
      opening_angle,
      _acceleration.get_buffer()
    };
    // -- Execute query to obtain accelerations
    cl_int err =_tree->get_tree_backend().execute_query(engine, handler);
    qcl::check_cl_error(err, "Error during tree query!");

    // Perform time integration
    _integrator.advance(_particles.get_buffer(),
                        _acceleration.get_buffer(),
                        _particles.size(),
                        dt);
  }

  Scalar get_current_time() const
  {
    return _integrator.get_current_time();
  }

  void retrieve_results(std::vector<host_vector3d>& positions,
                        std::vector<host_vector3d>& velocities,
                        std::vector<Scalar>& masses) const
  {
    std::vector<particle_type> raw_data;
    _particles.read(raw_data);

    positions.resize(_particles.size());
    velocities.resize(_particles.size());
    masses.resize(_particles.size());

    assert(raw_data.size() == _particles.size());
    for(std::size_t i = 0; i < raw_data.size(); ++i)
    {
      for(std::size_t j = 0; j < 3; ++j)
        positions[i][j] = raw_data[i].s[j];

      masses[i] = raw_data[i].s[3];

      for(std::size_t j = 0; j < 3; ++j)
        velocities[i][j] = raw_data[i].s[4+j];
    }
  }
private:
  qcl::device_context_ptr _ctx;
  nbody_tree_ptr _tree;
  qcl::device_array<particle_type> _particles;

  nbody_integrator<Scalar> _integrator;

  qcl::device_array<vector_type> _acceleration;
};

}

#endif
