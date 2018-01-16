
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -Wextra -Wno-ignored-attributes -Wno-unused-parameter")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -Wall -Wno-ignored-attributes -Wextra -Wno-unused-parameter")


cmake_minimum_required (VERSION 2.6)
project (SpatialCL)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
       "Choose the type of build, options are: Debug Release"
       FORCE)
endif(NOT CMAKE_BUILD_TYPE)

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

find_package(OpenCL REQUIRED)
find_package(Boost
  1.61      # Minimum version
  REQUIRED)  # Fail with error if Boost is not found


include_directories(${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}  ${OpenCL_INCLUDE_DIRS})
subdirs(tests examples)




