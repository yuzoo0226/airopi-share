cmake_minimum_required(VERSION 3.0.2)
project(hsr_data_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
)

add_service_files(DIRECTORY srv
  FILES
  StringTrigger.srv 
)

generate_messages()

catkin_package(CATKIN_DEPENDS message_runtime)