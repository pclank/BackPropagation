# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.17

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

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
CMAKE_COMMAND = "/home/pclank/Dev Tools/clion-2020.2.4/bin/cmake/linux/bin/cmake"

# The command to remove a file.
RM = "/home/pclank/Dev Tools/clion-2020.2.4/bin/cmake/linux/bin/cmake" -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/pclank/CLionProjects/BackPropagation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/pclank/CLionProjects/BackPropagation/cmake-build-debug

# Include any dependencies generated for this target.
include CMakeFiles/BackPropagation.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/BackPropagation.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/BackPropagation.dir/flags.make

CMakeFiles/BackPropagation.dir/ebp.c.o: CMakeFiles/BackPropagation.dir/flags.make
CMakeFiles/BackPropagation.dir/ebp.c.o: ../ebp.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pclank/CLionProjects/BackPropagation/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/BackPropagation.dir/ebp.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/BackPropagation.dir/ebp.c.o   -c /home/pclank/CLionProjects/BackPropagation/ebp.c

CMakeFiles/BackPropagation.dir/ebp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/BackPropagation.dir/ebp.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pclank/CLionProjects/BackPropagation/ebp.c > CMakeFiles/BackPropagation.dir/ebp.c.i

CMakeFiles/BackPropagation.dir/ebp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/BackPropagation.dir/ebp.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pclank/CLionProjects/BackPropagation/ebp.c -o CMakeFiles/BackPropagation.dir/ebp.c.s

CMakeFiles/BackPropagation.dir/ebp_omp.c.o: CMakeFiles/BackPropagation.dir/flags.make
CMakeFiles/BackPropagation.dir/ebp_omp.c.o: ../ebp_omp.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pclank/CLionProjects/BackPropagation/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/BackPropagation.dir/ebp_omp.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/BackPropagation.dir/ebp_omp.c.o   -c /home/pclank/CLionProjects/BackPropagation/ebp_omp.c

CMakeFiles/BackPropagation.dir/ebp_omp.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/BackPropagation.dir/ebp_omp.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pclank/CLionProjects/BackPropagation/ebp_omp.c > CMakeFiles/BackPropagation.dir/ebp_omp.c.i

CMakeFiles/BackPropagation.dir/ebp_omp.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/BackPropagation.dir/ebp_omp.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pclank/CLionProjects/BackPropagation/ebp_omp.c -o CMakeFiles/BackPropagation.dir/ebp_omp.c.s

CMakeFiles/BackPropagation.dir/ebp_p.c.o: CMakeFiles/BackPropagation.dir/flags.make
CMakeFiles/BackPropagation.dir/ebp_p.c.o: ../ebp_p.c
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/pclank/CLionProjects/BackPropagation/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/BackPropagation.dir/ebp_p.c.o"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -o CMakeFiles/BackPropagation.dir/ebp_p.c.o   -c /home/pclank/CLionProjects/BackPropagation/ebp_p.c

CMakeFiles/BackPropagation.dir/ebp_p.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/BackPropagation.dir/ebp_p.c.i"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/pclank/CLionProjects/BackPropagation/ebp_p.c > CMakeFiles/BackPropagation.dir/ebp_p.c.i

CMakeFiles/BackPropagation.dir/ebp_p.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/BackPropagation.dir/ebp_p.c.s"
	/usr/bin/cc $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/pclank/CLionProjects/BackPropagation/ebp_p.c -o CMakeFiles/BackPropagation.dir/ebp_p.c.s

# Object files for target BackPropagation
BackPropagation_OBJECTS = \
"CMakeFiles/BackPropagation.dir/ebp.c.o" \
"CMakeFiles/BackPropagation.dir/ebp_omp.c.o" \
"CMakeFiles/BackPropagation.dir/ebp_p.c.o"

# External object files for target BackPropagation
BackPropagation_EXTERNAL_OBJECTS =

BackPropagation: CMakeFiles/BackPropagation.dir/ebp.c.o
BackPropagation: CMakeFiles/BackPropagation.dir/ebp_omp.c.o
BackPropagation: CMakeFiles/BackPropagation.dir/ebp_p.c.o
BackPropagation: CMakeFiles/BackPropagation.dir/build.make
BackPropagation: CMakeFiles/BackPropagation.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/pclank/CLionProjects/BackPropagation/cmake-build-debug/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable BackPropagation"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/BackPropagation.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/BackPropagation.dir/build: BackPropagation

.PHONY : CMakeFiles/BackPropagation.dir/build

CMakeFiles/BackPropagation.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/BackPropagation.dir/cmake_clean.cmake
.PHONY : CMakeFiles/BackPropagation.dir/clean

CMakeFiles/BackPropagation.dir/depend:
	cd /home/pclank/CLionProjects/BackPropagation/cmake-build-debug && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/pclank/CLionProjects/BackPropagation /home/pclank/CLionProjects/BackPropagation /home/pclank/CLionProjects/BackPropagation/cmake-build-debug /home/pclank/CLionProjects/BackPropagation/cmake-build-debug /home/pclank/CLionProjects/BackPropagation/cmake-build-debug/CMakeFiles/BackPropagation.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/BackPropagation.dir/depend

