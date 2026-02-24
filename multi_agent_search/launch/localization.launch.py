import ast

from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessStart
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode
from launch_ros.event_handlers import OnStateTransition
from launch_ros.events.lifecycle import ChangeState
from launch_ros.substitutions import FindPackageShare
from lifecycle_msgs.msg import Transition

from launch import LaunchDescription


def _launch_localization_nodes(context):
    """Generate AMCL or slam_toolbox nodes for each robot based on use_known_map."""
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower() == "true"
    known_initial_poses = context.perform_substitution(LaunchConfiguration("known_initial_poses")).lower() == "true"
    agent_ids = ast.literal_eval(context.perform_substitution(LaunchConfiguration("agent_ids")))

    pkg_share = FindPackageShare("multi_agent_search")
    amcl_params_file = context.perform_substitution(PathJoinSubstitution([pkg_share, "config", "amcl_params.yaml"]))
    slam_params_file = context.perform_substitution(
        PathJoinSubstitution([pkg_share, "config", "slam_toolbox_params.yaml"])
    )

    actions = []

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

            # When AMCL process starts, emit configure
            configure_event = RegisterEventHandler(
                OnProcessStart(
                    target_action=amcl_node,
                    on_start=[
                        EmitEvent(
                            event=ChangeState(
                                lifecycle_node_matcher=lambda msg, node=amcl_node: msg.node_name == node.node_name,
                                transition_id=Transition.TRANSITION_CONFIGURE,
                            )
                        ),
                    ],
                )
            )

            # When AMCL finishes configuring (inactive), emit activate
            activate_event = RegisterEventHandler(
                OnStateTransition(
                    target_lifecycle_node=amcl_node,
                    goal_state="inactive",
                    entities=[
                        EmitEvent(
                            event=ChangeState(
                                lifecycle_node_matcher=lambda msg, node=amcl_node: msg.node_name == node.node_name,
                                transition_id=Transition.TRANSITION_ACTIVATE,
                            )
                        ),
                    ],
                )
            )

            actions.extend([amcl_node, configure_event, activate_event])
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

            configure_event = RegisterEventHandler(
                OnProcessStart(
                    target_action=slam_node,
                    on_start=[
                        EmitEvent(
                            event=ChangeState(
                                lifecycle_node_matcher=lambda msg, node=slam_node: msg.node_name == node.node_name,
                                transition_id=Transition.TRANSITION_CONFIGURE,
                            )
                        ),
                    ],
                )
            )

            activate_event = RegisterEventHandler(
                OnStateTransition(
                    target_lifecycle_node=slam_node,
                    goal_state="inactive",
                    entities=[
                        EmitEvent(
                            event=ChangeState(
                                lifecycle_node_matcher=lambda msg, node=slam_node: msg.node_name == node.node_name,
                                transition_id=Transition.TRANSITION_ACTIVATE,
                            )
                        ),
                    ],
                )
            )

            actions.extend([slam_node, configure_event, activate_event])

    return actions


def generate_launch_description():
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
            OpaqueFunction(function=_launch_localization_nodes),
        ]
    )
