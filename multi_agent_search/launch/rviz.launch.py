"""Launch file for RViz with dynamic TF frame configuration based on agent IDs."""

import ast
import tempfile

import yaml

from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.launch_context import LaunchContext
from launch.launch_description import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _generate_rviz_config(context: LaunchContext) -> list[Node]:
    """Generate an rviz config with TF frames for each agent and launch rviz."""
    from launch_ros.substitutions import FindPackageShare

    agent_ids: list[str] = ast.literal_eval(context.perform_substitution(LaunchConfiguration("agent_ids")))
    pkg_share_path = context.perform_substitution(FindPackageShare("multi_agent_search"))

    with open(f"{pkg_share_path}/config/config.rviz") as f:
        config = yaml.safe_load(f)

    # Find the TF display and update its Frames and Tree
    for display in config["Visualization Manager"]["Displays"]:
        if display.get("Class") == "rviz_default_plugins/TF":
            # Frames: only base_link enabled for each agent, odom and laser explicitly disabled
            frames: dict[str, object] = {"All Enabled": False, "map": {"Value": False}}
            for agent_id in agent_ids:
                frames[f"{agent_id}/base_link"] = {"Value": True}
                frames[f"{agent_id}/odom"] = {"Value": False}
                frames[f"{agent_id}/laser"] = {"Value": False}
            display["Frames"] = frames

            # Tree: map -> agent_id/odom -> agent_id/base_link (no laser children)
            tree: dict[str, dict[str, dict[str, object]]] = {}
            for agent_id in agent_ids:
                tree[f"{agent_id}/odom"] = {f"{agent_id}/base_link": {}}
            display["Tree"] = {"map": tree}
            break

    with tempfile.NamedTemporaryFile(mode="w", suffix=".rviz", delete=False) as tmp:
        yaml.dump(config, tmp, default_flow_style=False, allow_unicode=True)
        tmp_path = tmp.name

    return [
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", tmp_path],
        )
    ]


def generate_launch_description() -> LaunchDescription:
    """Generate the launch description for RViz."""
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "agent_ids",
                default_value="['robot_0', 'robot_1', 'robot_2']",
                description="Python list of agent IDs to visualize (e.g. \"['robot_0', 'robot_1']\")",
            ),
            OpaqueFunction(function=_generate_rviz_config),
        ]
    )
