#
# $Id$
#
# (C) Copyright 2009-2011 Johns Hopkins University (JHU), All Rights Reserved.
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


# find cisst with all available libraries
find_package (cisst REQUIRED ${REQUIRED_CISST_LIBRARIES})

if (cisst_FOUND)

  # load cisst configuration
  include (${CISST_USE_FILE})

  find_package (sawOpenNI REQUIRED HINTS "${CISST_BINARY_DIR}/saw/components")

  if (sawOpenNI_FOUND)

    include_directories (${sawOpenNI_INCLUDE_DIR})
    link_directories (${sawOpenNI_LIBRARY_DIR})

    # example without cisstStereoVision
    add_executable (oniExKinectTest oniExKinectTest.cpp)
    target_link_libraries (oniExKinectTest sawOpenNI ${OPENNI_LIBRARIES})
    cisst_target_link_libraries (oniExKinectTest cisstCommon cisstVector cisstOSAbstraction)

    # copy the configration file to binary tree
    add_custom_command (TARGET oniExKinectTest
                        POST_BUILD
                        COMMAND ${CMAKE_COMMAND}
                        ARGS -E copy
                             ${CMAKE_CURRENT_SOURCE_DIR}/SamplesConfig.xml
                             ${EXECUTABLE_OUTPUT_PATH}/${CMAKE_CFG_INTDIR}/SamplesConfig.xml
                        COMMENT "Copy Kinect configuration file to binary tree")

    # example with cisstStereoVision
    add_executable (oniExKinectView oniExKinectView.cpp)
    target_link_libraries (oniExKinectView sawOpenNI ${OPENNI_LIBRARIES})
    cisst_target_link_libraries (oniExKinectView ${REQUIRED_CISST_LIBRARIES})

  endif (sawOpenNI_FOUND)

endif (cisst_FOUND)

