# defined since 2.8.3
if (CMAKE_VERSION VERSION_LESS 2.8.3)
  get_filename_component (CMAKE_CURRENT_LIST_DIR ${CMAKE_CURRENT_LIST_FILE} PATH)
endif ()

# Allows loading FFTW3 settings from another project
set (FFTW3_CONFIG_FILE "${CMAKE_CURRENT_LIST_FILE}")

set (FFTW3f_LIBRARIES fftw3f)
set (FFTW3f_LIBRARY_DIRS /anvil/projects/x-phy130027/phd2020/GHOST_BT/main_files/GHOST-master/3D/src/fftw-3.3.10/lib)
set (FFTW3f_INCLUDE_DIRS /anvil/projects/x-phy130027/phd2020/GHOST_BT/main_files/GHOST-master/3D/src/fftw-3.3.10/include)

include ("${CMAKE_CURRENT_LIST_DIR}/FFTW3LibraryDepends.cmake")

if (CMAKE_VERSION VERSION_LESS 2.8.3)
  set (CMAKE_CURRENT_LIST_DIR)
endif ()
