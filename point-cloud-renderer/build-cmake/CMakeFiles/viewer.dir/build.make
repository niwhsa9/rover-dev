# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ashwin/Desktop/rover-dev/point-cloud-renderer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ashwin/Desktop/rover-dev/point-cloud-renderer/build-cmake

# Include any dependencies generated for this target.
include CMakeFiles/viewer.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/viewer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/viewer.dir/flags.make

CMakeFiles/viewer.dir/viewer.cpp.o: CMakeFiles/viewer.dir/flags.make
CMakeFiles/viewer.dir/viewer.cpp.o: ../viewer.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ashwin/Desktop/rover-dev/point-cloud-renderer/build-cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/viewer.dir/viewer.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/viewer.dir/viewer.cpp.o -c /home/ashwin/Desktop/rover-dev/point-cloud-renderer/viewer.cpp

CMakeFiles/viewer.dir/viewer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/viewer.dir/viewer.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ashwin/Desktop/rover-dev/point-cloud-renderer/viewer.cpp > CMakeFiles/viewer.dir/viewer.cpp.i

CMakeFiles/viewer.dir/viewer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/viewer.dir/viewer.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ashwin/Desktop/rover-dev/point-cloud-renderer/viewer.cpp -o CMakeFiles/viewer.dir/viewer.cpp.s

CMakeFiles/viewer.dir/viewer.cpp.o.requires:

.PHONY : CMakeFiles/viewer.dir/viewer.cpp.o.requires

CMakeFiles/viewer.dir/viewer.cpp.o.provides: CMakeFiles/viewer.dir/viewer.cpp.o.requires
	$(MAKE) -f CMakeFiles/viewer.dir/build.make CMakeFiles/viewer.dir/viewer.cpp.o.provides.build
.PHONY : CMakeFiles/viewer.dir/viewer.cpp.o.provides

CMakeFiles/viewer.dir/viewer.cpp.o.provides.build: CMakeFiles/viewer.dir/viewer.cpp.o


# Object files for target viewer
viewer_OBJECTS = \
"CMakeFiles/viewer.dir/viewer.cpp.o"

# External object files for target viewer
viewer_EXTERNAL_OBJECTS =

viewer: CMakeFiles/viewer.dir/viewer.cpp.o
viewer: CMakeFiles/viewer.dir/build.make
viewer: /usr/lib/x86_64-linux-gnu/libGL.so
viewer: /usr/lib/x86_64-linux-gnu/libGLU.so
viewer: /usr/lib/x86_64-linux-gnu/libglut.so
viewer: /usr/lib/x86_64-linux-gnu/libXmu.so
viewer: /usr/lib/x86_64-linux-gnu/libXi.so
viewer: /usr/lib/x86_64-linux-gnu/libGLEW.so
viewer: CMakeFiles/viewer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ashwin/Desktop/rover-dev/point-cloud-renderer/build-cmake/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable viewer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/viewer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/viewer.dir/build: viewer

.PHONY : CMakeFiles/viewer.dir/build

CMakeFiles/viewer.dir/requires: CMakeFiles/viewer.dir/viewer.cpp.o.requires

.PHONY : CMakeFiles/viewer.dir/requires

CMakeFiles/viewer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/viewer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/viewer.dir/clean

CMakeFiles/viewer.dir/depend:
	cd /home/ashwin/Desktop/rover-dev/point-cloud-renderer/build-cmake && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ashwin/Desktop/rover-dev/point-cloud-renderer /home/ashwin/Desktop/rover-dev/point-cloud-renderer /home/ashwin/Desktop/rover-dev/point-cloud-renderer/build-cmake /home/ashwin/Desktop/rover-dev/point-cloud-renderer/build-cmake /home/ashwin/Desktop/rover-dev/point-cloud-renderer/build-cmake/CMakeFiles/viewer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/viewer.dir/depend

