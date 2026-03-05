"""Launch file for Nav2 navigation nodes (phased per-robot to avoid DDS overload)."""

import ast

from nav2_common.launch import RewrittenYaml

from launch.action import Action
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_context import LaunchContext
from launch.launch_description import LaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def _create_nav_group(
    agent_id: str,
    use_known_map: bool,
    params_file: str,
) -> GroupAction:
    """Create the Nav2 node group for a single robot."""
    map_topic = "/ground_truth_map" if use_known_map else f"/{agent_id}/map"

    controller_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=agent_id,
            param_rewrites={
                "robot_base_frame": f"{agent_id}/base_link",
                "global_frame": f"{agent_id}/odom",
                "local_frame": f"{agent_id}/odom",
                "odom_topic": f"/{agent_id}/odom",
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    nav_params = ParameterFile(
        RewrittenYaml(
            source_file=params_file,
            root_key=agent_id,
            param_rewrites={
                "robot_base_frame": f"{agent_id}/base_link",
                "global_frame": "map",
                "local_frame": f"{agent_id}/odom",
                "odom_topic": f"/{agent_id}/odom",
                "map_topic": map_topic,
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    lifecycle_nodes = [
        "controller_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
    ]

    return GroupAction(
        actions=[
            SetParameter("use_sim_time", True),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                namespace=agent_id,
                output="screen",
                parameters=[controller_params],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                namespace=agent_id,
                output="screen",
                parameters=[nav_params],
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                namespace=agent_id,
                output="screen",
                parameters=[nav_params],
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                namespace=agent_id,
                output="screen",
                parameters=[nav_params],
            ),
            Node(
                package="nav2_lifecycle_manager",
                executable="lifecycle_manager",
                name="navigation_lifecycle_manager",
                namespace=agent_id,
                output="screen",
                parameters=[
                    {
                        "autostart": True,
                        "node_names": lifecycle_nodes,
                        "bond_timeout": 0.0,
                    }
                ],
            ),
        ],
    )


def _create_nav_monitor(agent_id: str) -> Node:
    """Create a lifecycle monitor that waits for a single robot's Nav2 nodes to become active."""
    nav_nodes = [
        f"{agent_id}/controller_server",
        f"{agent_id}/planner_server",
        f"{agent_id}/behavior_server",
        f"{agent_id}/bt_navigator",
    ]
    return Node(
        package="multi_agent_search",
        executable="lifecycle_monitor",
        name=f"{agent_id}_nav_monitor",
        parameters=[
            {"node_names": nav_nodes},
            {"timeout": 60.0},
        ],
    )


def _launch_navigation_nodes(context: LaunchContext) -> list[Action]:
    """Launch Nav2 navigation servers for each robot, phased sequentially."""
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower() == "true"
    agent_ids = ast.literal_eval(context.perform_substitution(LaunchConfiguration("agent_ids")))

    pkg_share = FindPackageShare("multi_agent_search")
    params_file = context.perform_substitution(PathJoinSubstitution([pkg_share, "config", "nav2_params.yaml"]))

    if not agent_ids:
        return []

    # Launch the first robot's nav stack immediately with its monitor
    first_id = agent_ids[0]
    first_group = _create_nav_group(first_id, use_known_map, params_file)
    first_monitor = _create_nav_monitor(first_id)
    actions: list[Action] = [first_group, first_monitor]

    # Chain subsequent robots: each waits for the previous monitor to exit
    prev_monitor = first_monitor
    for agent_id in agent_ids[1:]:
        nav_group = _create_nav_group(agent_id, use_known_map, params_file)
        nav_monitor = _create_nav_monitor(agent_id)
        actions.append(
            RegisterEventHandler(OnProcessExit(target_action=prev_monitor, on_exit=[nav_group, nav_monitor]))
        )
        prev_monitor = nav_monitor

    return actions


def generate_launch_description() -> LaunchDescription:
    """Generate the launch description for the Nav2 navigation nodes."""
    return LaunchDescription(
        [
            SetEnvironmentVariable("RCUTILS_LOGGING_BUFFERED_STREAM", "1"),
            DeclareLaunchArgument(
                "use_known_map",
                default_value="true",
                description="If true, use ground truth map; if false, use per-agent SLAM map",
            ),
            DeclareLaunchArgument(
                "agent_ids",
                description="Python list of agent IDs, e.g. \"['robot_0', 'robot_1']\"",
            ),
            OpaqueFunction(function=_launch_navigation_nodes),
        ]
    )
