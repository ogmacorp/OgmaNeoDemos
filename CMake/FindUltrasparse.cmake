# Locate ultrasparse library
#
# This module defines
# ULTRASPARSE_LIBRARY, the name of the library to link against
# ULTRASPARSE_FOUND, if false, do not try to link to ultrasparse
# ULTRASPARSE_INCLUDE_DIR, where to find ultrasparse headers

if(ULTRASPARSE_INCLUDE_DIR)
    # Already in cache, be silent
    set(ULTRASPARSE_FIND_QUIETLY TRUE)
endif(ULTRASPARSE_INCLUDE_DIR)

find_path(ULTRASPARSE_INCLUDE_DIR ultrasparse/sdr.h)

set(ULTRASPARSE_NAMES ultrasparse Ultrasparse UltraSparse ULTRASPARSE)

find_library(ULTRASPARSE_LIBRARY NAMES ${ULTRASPARSE_NAMES})

# Per-recommendation
set(ULTRASPARSE_INCLUDE_DIRS "${ULTRASPARSE_INCLUDE_DIR}")
set(ULTRASPARSE_LIBRARIES "${ULTRASPARSE_LIBRARY}")

# handle the QUIETLY and REQUIRED arguments and set ULTRASPARSE_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ultrasparse DEFAULT_MSG ULTRASPARSE_LIBRARY ULTRASPARSE_INCLUDE_DIR)
