cmake_minimum_required(VERSION 3.5)
project(contact_plugin)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)

include_directories(
  include
  ${GAZEBO_INCLUDE_DIRS}
)

add_library(contact_plugin SHARED src/contact_plugin.cpp)
ament_target_dependencies(contact_plugin rclcpp geometry_msgs gazebo_ros)

install(TARGETS contact_plugin
  LIBRARY DESTINATION lib
)

ament_package()

