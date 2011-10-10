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

set (REQUIRED_CISST_LIBRARIES cisstCommon
                              cisstVector
                              cisstOSAbstraction
                              cisstMultiTask
                              cisstStereoVision)

find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})

if (cisst_FOUND_AS_REQUIRED)

  # load cisst configuration
  include (${CISST_USE_FILE})

  set (HEADER_FILES
       ../sawOpenNIExport.h
       ../osaOpenNI.h
       ../osaOpenNISkeleton.h
       osaOpenNIData.h
       )
  set (SOURCE_FILES
       osaOpenNI.cpp
       osaOpenNISkeleton.cpp
       osaOpenNIData.cpp
       )

  # SVL (cisstStereoVision) headers and sources
  set (HEADER_FILES
       ${HEADER_FILES}
       ../svlFilterSourceKinect.h
       )
  set (SOURCE_FILES
       ${SOURCE_FILES}
       svlFilterSourceKinect.cpp
       )


  add_library (sawOpenNI
               ${HEADER_FILES}
               ${SOURCE_FILES}
               )

  target_link_libraries (sawOpenNI
                         ${OPENNI_LIBRARIES}
                         )

  cisst_target_link_libraries (sawOpenNI
                               cisstCommon
                               cisstVector
                               cisstOSAbstraction
                               cisstMultiTask
                               cisstStereoVision
                               )

  set (sawOpenNI_INCLUDE_DIR "${sawComponents_SOURCE_DIR}")

  set (sawOpenNI_LIBRARY_DIR "${LIBRARY_OUTPUT_PATH}")

  set (sawOpenNI_LIBRARIES sawOpenNI ${OPENNI_LIBRARIES})

  configure_file ("${CMAKE_CURRENT_SOURCE_DIR}/sawOpenNIConfig.cmake.in"
                  "${sawComponents_BINARY_DIR}/sawOpenNIConfig.cmake"
                  @ONLY@
                  )

else (cisst_FOUND_AS_REQUIRED)
  message ("Information: code in ${CMAKE_CURRENT_SOURCE_DIR} will not be compiled, it requires ${REQUIRED_CISST_LIBRARIES}")
endif (cisst_FOUND_AS_REQUIRED)
