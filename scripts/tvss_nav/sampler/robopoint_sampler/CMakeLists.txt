cmake_minimum_required(VERSION 3.0.2)
project(robopoint_sampler)

find_package(catkin REQUIRED
  COMPONENTS rospy sensor_msgs std_msgs geometry_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/robopoint_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
