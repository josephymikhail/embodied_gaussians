cmake_minimum_required(VERSION 3.20...3.27)
project(pysegreduce
    LANGUAGES CXX CUDA
)


find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(pybind11 CONFIG REQUIRED)


pybind11_add_module(${PROJECT_NAME}
    segmented_reduce.cu
)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
)

install(TARGETS ${PROJECT_NAME}
    DESTINATION ${Python_SITELIB}
)