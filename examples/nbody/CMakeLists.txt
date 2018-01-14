add_executable(nbody nbody.cpp)
target_link_libraries (nbody ${OpenCL_LIBRARIES} -lpng)
