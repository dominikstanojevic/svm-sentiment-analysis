cmake_minimum_required(VERSION 3.6)
project(SMO)

set(CMAKE_C_STANDARD 99)

include_directories(C:/ProgramData/Anaconda3/include C:/ProgramData/Anaconda3/Lib/site-packages/numpy/core/include)

set(SOURCE_FILES smo.c smo.h)
add_executable(SMO ${SOURCE_FILES})