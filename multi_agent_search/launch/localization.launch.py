"""Launch file for localization nodes."""

import ast

from launch.action import Action
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.launch_description import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node, SetParameter
from launch_ros.substitutions import FindPackageShare


def _launch_localization_nodes(context: LaunchContext) -> list[Action]:
    """Generate AMCL or slam_toolbox nodes for each robot based on use_known_map."""
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower() == "true"
    known_initial_poses = context.perform_substitution(LaunchConfiguration("known_initial_poses")).lower() == "true"
    agent_ids = ast.literal_eval(context.perform_substitution(LaunchConfiguration("agent_ids")))

    pkg_share = FindPackageShare("multi_agent_search")
    amcl_params_file = context.perform_substitution(PathJoinSubstitution([pkg_share, "config", "amcl_params.yaml"]))
    slam_params_file = context.perform_substitution(
        PathJoinSubstitution([pkg_share, "config", "slam_toolbox_params.yaml"])
    )

    actions: list[Action] = []

    if use_known_map:
        for agent_id in agent_ids:
            amcl_node = LifecycleNode(
                package="nav2_amcl",
                executable="amcl",
                name="amcl",
                namespace=agent_id,
                output="screen",
                parameters=[
                    amcl_params_file,
                    {
                        "base_frame_id": f"{agent_id}/base_link",
                        "odom_frame_id": f"{agent_id}/odom",
                        "global_frame_id": "map",
                        "set_initial_pose": not known_initial_poses,
                    },
                ],
                remappings=[
                    ("scan", f"/{agent_id}/base_scan"),
                    ("map", "/ground_truth_map"),
                ],
            )
            actions.append(amcl_node)

        lifecycle_manager = Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="localization_lifecycle_manager",
            output="screen",
            parameters=[
                {
                    "node_names": [f"{agent_id}/amcl" for agent_id in agent_ids],
                    "autostart": True,
                    "bond_timeout": 0.0,
                }
            ],
        )
        actions.append(lifecycle_manager)
    else:
        for agent_id in agent_ids:
            slam_node = LifecycleNode(
                package="slam_toolbox",
                executable="async_slam_toolbox_node",
                name="slam_toolbox",
                namespace=agent_id,
                output="screen",
                parameters=[
                    slam_params_file,
                    {
                        "base_frame": f"{agent_id}/base_link",
                        "odom_frame": f"{agent_id}/odom",
                        "map_frame": f"{agent_id}/map",
                    },
                ],
                remappings=[
                    ("/scan", f"/{agent_id}/base_scan"),
                    ("/map", f"/{agent_id}/map"),
                    ("/map_metadata", f"/{agent_id}/map_metadata"),
                ],
            )
            actions.append(slam_node)

        lifecycle_manager = Node(
            package="nav2_lifecycle_manager",
            executable="lifecycle_manager",
            name="localization_lifecycle_manager",
            output="screen",
            parameters=[
                {
                    "node_names": [f"{agent_id}/slam_toolbox" for agent_id in agent_ids],
                    "autostart": True,
                    "bond_timeout": 0.0,
                }
            ],
        )
        actions.append(lifecycle_manager)

    return actions


def generate_launch_description() -> LaunchDescription:
    """Generate the launch description for the localization nodes."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_known_map",
                default_value="true",
                description="If true, use AMCL with known map; if false, use slam_toolbox for SLAM",
            ),
            DeclareLaunchArgument(
                "agent_ids",
                description="Python list of agent IDs, e.g. \"['robot_0', 'robot_1']\"",
            ),
            DeclareLaunchArgument(
                "known_initial_poses",
                default_value="false",
                description="If true, disable AMCL's default initial pose so it waits for /initialpose",
            ),
            SetParameter("use_sim_time", True),
            OpaqueFunction(function=_launch_localization_nodes),
        ]
    )
