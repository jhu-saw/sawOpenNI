#
# $Id$
#
# (C) Copyright 2011 Johns Hopkins University (JHU), All Rights
# Reserved.
#
# --- begin cisst license - do not edit ---
#
# This software is provided "as is" under an open source license, with
# no warranty.  The complete license can be found in license.txt and
# http://www.cisst.org/cisst/license.txt.
#
# --- end cisst license ---

set (CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/code")

find_package (OpenNI REQUIRED)

if (OPENNI_FOUND)
  include_directories (${OPENNI_INCLUDE_DIRS})
  add_subdirectory (code)
  add_subdirectory (examples)
else (OPENNI_FOUND)
  message (SEND_ERROR "code in ${CMAKE_CURRENT_SOURCE_DIR} requires OpenNI.")
endif (OPENNI_FOUND)
