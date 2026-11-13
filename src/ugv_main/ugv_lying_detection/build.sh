#!/bin/bash

rm -rf ~/ugv_ws/build/ugv_lying_detection
rm -rf ~/ugv_ws/install/ugv_lying_detection
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ugv_lying_detection
source install/setup.bash
ros2 pkg executables ugv_lying_detection

echo "If everything succeeded, to run the executable : ros2 run ugv_lying_detection lying_detection"
echo "Do not forgot to run the camera : ros2 run usb_cam usb_cam_node_exe"
echo "And the slam nav : ros2 launch ugv_nav slam_nav.launch.py use_rviz:=true"