cmake_minimum_required(VERSION 2.8.12)

project(dip_hw)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include(./dlib-19.2/dlib/cmake)

ADD_EXECUTABLE(face_landmark_detection face_landmark_detection.cpp)
TARGET_LINK_LIBRARIES(face_landmark_detection ${OpenCV_LIBS})
TARGET_LINK_LIBRARIES(face_landmark_detection dlib)

# Since there are a lot of examples I'm going to use a macro to simply this
# CMakeLists.txt file.  However, usually you will create only one executable in
# your cmake projects and use the syntax shown above.
MACRO(add_example name)
   ADD_EXECUTABLE(${name} ${name}.cpp)
   TARGET_LINK_LIBRARIES(${name} dlib )
ENDMACRO()
