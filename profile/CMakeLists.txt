cmake_minimum_required(VERSION 2.8)

if(ANDROID)
  include_directories("include") # This is needed for gtest.
  add_executable(caffe-profile
    src/cmdparser.cpp
    src/cmdoptions.cpp
    src/caffe_mobile.cpp
    src/basic.cpp
    src/main.cpp
  )
  target_link_libraries(caffe-profile ${Caffe_LINK})

  # set back RUNTIME_OUTPUT_DIRECTORY
  set_target_properties(caffe-profile PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/android")

  caffe_set_solution_folder(caffe-profile android)
endif()
