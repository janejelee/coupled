# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

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
CMAKE_SOURCE_DIR = /scratch/leej/deal.II/my_examples/coupled/directsolver/tests

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /scratch/leej/deal.II/my_examples/coupled/directsolver/tests

# Include any dependencies generated for this target.
include CMakeFiles/homogeneous_pv.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/homogeneous_pv.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/homogeneous_pv.dir/flags.make

CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o: CMakeFiles/homogeneous_pv.dir/flags.make
CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o: homogeneous_pv.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/scratch/leej/deal.II/my_examples/coupled/directsolver/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o -c /scratch/leej/deal.II/my_examples/coupled/directsolver/tests/homogeneous_pv.cc

CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /scratch/leej/deal.II/my_examples/coupled/directsolver/tests/homogeneous_pv.cc > CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.i

CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /scratch/leej/deal.II/my_examples/coupled/directsolver/tests/homogeneous_pv.cc -o CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.s

CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o.requires:

.PHONY : CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o.requires

CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o.provides: CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o.requires
	$(MAKE) -f CMakeFiles/homogeneous_pv.dir/build.make CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o.provides.build
.PHONY : CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o.provides

CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o.provides.build: CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o


# Object files for target homogeneous_pv
homogeneous_pv_OBJECTS = \
"CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o"

# External object files for target homogeneous_pv
homogeneous_pv_EXTERNAL_OBJECTS =

homogeneous_pv: CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o
homogeneous_pv: CMakeFiles/homogeneous_pv.dir/build.make
homogeneous_pv: /scratch/leej/deal.II/lib/libdeal_II.g.so.8.4.1
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libbz2.so
homogeneous_pv: /usr/lib/openmpi/lib/libmpi_usempif08.so
homogeneous_pv: /usr/lib/openmpi/lib/libmpi_usempi_ignore_tkr.so
homogeneous_pv: /usr/lib/openmpi/lib/libmpi_mpifh.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libz.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libmuelu-adapters.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libmuelu-interface.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libmuelu.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libteko.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libstratimikos.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libstratimikosbelos.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libstratimikosaztecoo.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libstratimikosamesos.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libstratimikosml.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libstratimikosifpack.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libifpack2-adapters.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libifpack2.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libanasazitpetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libModeLaplace.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libanasaziepetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libanasazi.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libamesos2.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libbelostpetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libbelosepetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libbelos.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libml.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libifpack.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libzoltan2.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libpamgen_extras.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libpamgen.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libamesos.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libgaleri-xpetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libgaleri-epetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libaztecoo.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libisorropia.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libxpetra-sup.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libxpetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libthyratpetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libthyraepetraext.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libthyraepetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libthyracore.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libepetraext.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libtpetraext.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libtpetrainout.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libtpetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libkokkostsqr.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libtpetrakernels.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libtpetraclassiclinalg.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libtpetraclassicnodeapi.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libtpetraclassic.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libtriutils.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libzoltan.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libepetra.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libsacado.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/librtop.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libteuchoskokkoscomm.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libteuchoskokkoscompat.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libteuchosremainder.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libteuchosnumerics.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libteuchoscomm.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libteuchosparameterlist.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libteuchoscore.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libkokkosalgorithms.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libkokkoscontainers.so
homogeneous_pv: /scratch/trilinos-12.10.1-Source/lib/libkokkoscore.so
homogeneous_pv: /usr/lib/libblas.so
homogeneous_pv: /usr/lib/openmpi/lib/libmpi_cxx.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libumfpack.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libcholmod.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libccolamd.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libcolamd.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libcamd.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libamd.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libmetis.so
homogeneous_pv: /usr/lib/libparpack.so
homogeneous_pv: /usr/lib/libarpack.so
homogeneous_pv: /usr/lib/liblapack.so
homogeneous_pv: /usr/lib/libf77blas.so
homogeneous_pv: /usr/lib/libatlas.so
homogeneous_pv: /usr/lib/openmpi/lib/libmpi.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libnetcdf_c++.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libnetcdf.so
homogeneous_pv: /usr/lib/x86_64-linux-gnu/libslepc.so
homogeneous_pv: /scratch/leej/petsc-3.6.4/x86_64/lib/libpetsc.so
homogeneous_pv: CMakeFiles/homogeneous_pv.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/scratch/leej/deal.II/my_examples/coupled/directsolver/tests/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable homogeneous_pv"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/homogeneous_pv.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/homogeneous_pv.dir/build: homogeneous_pv

.PHONY : CMakeFiles/homogeneous_pv.dir/build

CMakeFiles/homogeneous_pv.dir/requires: CMakeFiles/homogeneous_pv.dir/homogeneous_pv.cc.o.requires

.PHONY : CMakeFiles/homogeneous_pv.dir/requires

CMakeFiles/homogeneous_pv.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/homogeneous_pv.dir/cmake_clean.cmake
.PHONY : CMakeFiles/homogeneous_pv.dir/clean

CMakeFiles/homogeneous_pv.dir/depend:
	cd /scratch/leej/deal.II/my_examples/coupled/directsolver/tests && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /scratch/leej/deal.II/my_examples/coupled/directsolver/tests /scratch/leej/deal.II/my_examples/coupled/directsolver/tests /scratch/leej/deal.II/my_examples/coupled/directsolver/tests /scratch/leej/deal.II/my_examples/coupled/directsolver/tests /scratch/leej/deal.II/my_examples/coupled/directsolver/tests/CMakeFiles/homogeneous_pv.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/homogeneous_pv.dir/depend

