cmake_minimum_required(VERSION 3.0.2)
project(pointcloud_seg)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  sensor_msgs
  tf
  pcl_ros
  message_filters
  image_transport
  cv_bridge
)

# Find system dependencies
find_package(PCL REQUIRED)

# Declare catkin package
catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs tf pcl_ros message_filters image_transport cv_bridge
)

# Include directories
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

# Declare the executable
add_executable(seg_node src/seg_node.cpp)

# Link the libraries
target_link_libraries(seg_node
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
