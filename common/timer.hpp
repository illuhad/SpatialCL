/*
 * This file is part of SpatialCL, a library for the spatial processing of
 * particles.
 *
 * Copyright (c) 2018 Aksel Alpay
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

#ifndef TIMER_HPP
#define TIMER_HPP

#include <chrono>
#include <ctime>

namespace common {
  
class timer
{
public:
  timer()
  : _is_running{false}
  {}

  inline
  bool is_running() const 
  {return _is_running;}

  void start()
  {
    _is_running = true;
    _start = std::chrono::high_resolution_clock::now();
  }

  double stop()
  {
    if(!_is_running)
      return 0.0;

    _stop = std::chrono::high_resolution_clock::now();
    _is_running = false;

    auto ticks = std::chrono::duration_cast<std::chrono::nanoseconds>(_stop - _start).count();
    return static_cast<double>(ticks) * 1.e-9;
  }

private:
  using time_point_type = 
    std::chrono::time_point<std::chrono::high_resolution_clock>;
  time_point_type _start;
  time_point_type _stop;

  bool _is_running;
};

class cumulative_timer
{
public:
  cumulative_timer()
  {
    reset();
  }

  void reset()
  {
    _num_runs = 0;
    _total_runtime = 0.0;
  }

  double get_average_runtime() const
  {
    if(_num_runs == 0)
      return 0.0;

    return _total_runtime / static_cast<double>(_num_runs);
  }

  double get_total_runtime() const
  {
    return _total_runtime;
  }

  unsigned get_num_runs() const
  {
    return _num_runs;
  }

  void start()
  {
    _timer.start();
  }

  void stop()
  {
    _total_runtime += _timer.stop();
    _num_runs++;
  }

private:
  double _total_runtime;
  unsigned _num_runs;
  timer _timer;
};

}

#endif
