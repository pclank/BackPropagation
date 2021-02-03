cmake_minimum_required(VERSION 3.17)
project(BackPropagation C)

set(CMAKE_C_STANDARD 99)

add_executable(BackPropagation ebp.c ebp_omp.c)