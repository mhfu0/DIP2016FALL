cmake_minimum_required(VERSION 2.8.12)

project(dip_hw2)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(hist hist.cpp)
add_executable(unsharp unsharp.cpp)

target_link_libraries(hist ${OpenCV_LIBS})
target_link_libraries(unsharp ${OpenCV_LIBS})
