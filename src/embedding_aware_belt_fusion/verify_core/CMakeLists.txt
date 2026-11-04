cmake_minimum_required(VERSION 3.18)
project(verify_core CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(verify_core verify_core.cpp)
# Same flags as fusion_core: -O3 without -ffast-math, no FMA contraction, so float32 rounding follows the source.
target_compile_options(verify_core PRIVATE -O3 -march=native -ffp-contract=off)
