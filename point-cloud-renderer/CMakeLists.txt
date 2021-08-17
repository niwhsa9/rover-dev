cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(viewer)

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_PATH})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
add_executable (viewer viewer.cpp) 
target_link_libraries(viewer 
                        ${SPECIAL_OS_LIBS}
                        ${OPENGL_LIBRARIES}
                        ${GLUT_LIBRARY}
                        ${GLEW_LIBRARIES})


