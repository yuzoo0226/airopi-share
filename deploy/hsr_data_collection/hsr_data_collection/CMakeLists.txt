cmake_minimum_required(VERSION 3.0.2)
project(hsr_data_collection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tmc_msgs
  hsr_data_msgs
)

catkin_package(
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)