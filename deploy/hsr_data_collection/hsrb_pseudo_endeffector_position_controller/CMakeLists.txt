cmake_minimum_required(VERSION 3.0.2)
project(hsrb_pseudo_endeffector_position_controller)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  tmc_manipulation_types_bridge
  tmc_robot_kinematics_model
  tmc_control_msgs
  trajectory_msgs
  rospy
)

include_directories(${catkin_INCLUDE_DIRS})
find_package(cmake_modules REQUIRED)
find_package(Eigen3 REQUIRED)
include_directories(${Eigen_INCLUDE_DIRS})
add_definitions(${EIGEN_DEFINITIONS})

catkin_package(
  CATKIN_DEPENDS
  geometry_msgs
  nav_msgs
  roscpp
  sensor_msgs
  tmc_manipulation_types_bridge
  tmc_robot_kinematics_model
  trajectory_msgs
)

add_executable(${PROJECT_NAME}
  src/hsrb_pseudo_endeffector_position_controller.cpp
)
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
)

install(TARGETS ${PROJECT_NAME} ${PROJECT_NAME}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)
