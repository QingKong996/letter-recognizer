cmake_minimum_required(VERSION 3.20)

project(letter-recognizer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_SCAN_FOR_MODULES ON)

set(CMAKE_ABSOLUTE_DESTINATION_FILES ON)

add_executable(letter-recognizer main.cpp)
target_compile_options(letter-recognizer PRIVATE
    $<$<CONFIG:RELEASE>:O3 -march=native>)
