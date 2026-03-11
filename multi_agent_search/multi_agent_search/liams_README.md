

rm -rf build install log
colcon build --packages-select stage stage_ros2 floorplan_generator_stage
colcon build --symlink-install --packages-select multi_agent_search_interfaces multi_agent_search
source install/setup.bash

ros2 launch multi_agent_search multi_agent_search.launch.py num_robots:=2 num_targets:=2 agent_executable:=convoy_agent
ros2 launch multi_agent_search multi_agent_search.launch.py num_robots:=2 num_targets:=1 agent_executable:=frontier_meetup_agent
ros2 launch multi_agent_search multi_agent_search.launch.py num_robots:=2 num_targets:=2 agent_executable:=conscientious_reactive_agent


export RMW_IMPLEMENTATION=rmw_zenoh_cpp
ros2 run rmw_zenoh_cpp rmw_zenohd

robot_n-1 Managed nodes are active
    this is the last blue log you want to see

ros2 node list
ros2 daemon stop && fastdds shm clean && ros2 daemon start

ros2 topic info /ground_truth_map --verbose

ros2 daemon stop && fastdds shm clean && ros2 daemon start
pgrep -af 'ros2|python.launch|gz sim|gazebo|stage|slam_toolbox|amcl|nav2|lifecycle_manager|robot_state_publisher|static_transform_publisher|rviz'
pkill -f 'python.launch'
pkill -f 'ros2'
pkill -f 'gz sim|gazebo|stage'
pkill -f 'slam_toolbox|amcl|nav2|lifecycle_manager|robot_state_publisher|static_transform_publisher|rviz'
pkill -9 -f 'ros2|python.*launch|gz sim|gazebo|stage|slam_toolbox|amcl|nav2|lifecycle_manager|robot_state_publisher|static_transform_publisher|rviz'
ros2 launch multi_agent_search multi_agent_search.launch.py num_robots:=5 num_targets:=2 agent_executable:=conscientious_reactive_agent > log.txt