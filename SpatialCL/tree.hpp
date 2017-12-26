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

#include "tree/particle_bvh_sfc_tree.hpp"

namespace spatialcl {


template<class Type_descriptor>
using zcurve_bvh_tree =
  particle_bvh_tree<key_based_sorter<zcurve_sort_key_generator<Type_descriptor>>, Type_descriptor>;

template<std::size_t Num_particle_components>
using zcurve_bvh_sp2d_tree = zcurve_bvh_tree<type_descriptor::single_precision2d<Num_particle_components>>;

template<std::size_t Num_particle_components>
using zcurve_bvh_sp3d_tree = zcurve_bvh_tree<type_descriptor::single_precision3d<Num_particle_components>>;

template<std::size_t Num_particle_components>
using zcurve_bvh_dp2d_tree = zcurve_bvh_tree<type_descriptor::double_precision2d<Num_particle_components>>;

template<std::size_t Num_particle_components>
using zcurve_bvh_dp3d_tree = zcurve_bvh_tree<type_descriptor::double_precision3d<Num_particle_components>>;


template<class Type_descriptor>
using hilbert_bvh_tree =
  particle_bvh_tree<key_based_sorter<hilbert_sort_key_generator<Type_descriptor>>, Type_descriptor>;

template<std::size_t Num_particle_components>
using hilbert_bvh_sp2d_tree = hilbert_bvh_tree<type_descriptor::single_precision2d<Num_particle_components>>;

template<std::size_t Num_particle_components>
using hilbert_bvh_sp3d_tree = hilbert_bvh_tree<type_descriptor::single_precision3d<Num_particle_components>>;

template<std::size_t Num_particle_components>
using hilbert_bvh_dp2d_tree = hilbert_bvh_tree<type_descriptor::double_precision2d<Num_particle_components>>;

template<std::size_t Num_particle_components>
using hilbert_bvh_dp3d_tree = hilbert_bvh_tree<type_descriptor::double_precision3d<Num_particle_components>>;

//using kd_bvh_tree = particle_bvh_tree<key_based_sorter<kd_sort_key_generator>>;


}
