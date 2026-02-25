"""Launch file for testing the multi-agent search system."""

import tempfile
from pathlib import Path

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
)
from launch.event_handlers import OnProcessExit
from launch.launch_context import LaunchContext
from launch.launch_description import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import LifecycleNode, Node
from launch_ros.substitutions import FindPackageShare

PLACEHOLDER = "{{NUM_ROBOTS}}"


def _validate_args(context: LaunchContext) -> list[Node]:
    """Validate that known_initial_poses is not true when use_known_map is false."""
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower() == "true"
    known_initial_poses = context.perform_substitution(LaunchConfiguration("known_initial_poses")).lower() == "true"
    if known_initial_poses and not use_known_map:
        raise ValueError("known_initial_poses requires use_known_map to be true (AMCL must be running)")
    return []


def _resolve_config(template_path: str, num_robots: int, suffix: str) -> str:
    """Read a config template, replace {{NUM_ROBOTS}}, and write to a temp file."""
    content = Path(template_path).read_text()
    content = content.replace(PLACEHOLDER, str(num_robots))
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp.write(content)
    return tmp.name


def _launch_system(context: LaunchContext) -> list[Node | LifecycleNode]:
    """
    Launch the full system: simulation, stage monitor, and post-stage nodes.

    Resolves num_robots to generate config files from templates and create the correct number of agent nodes.
    """
    num_robots = int(context.perform_substitution(LaunchConfiguration("num_robots")))
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower()
    known_initial_poses = context.perform_substitution(LaunchConfiguration("known_initial_poses")).lower()

    agent_ids = [f"robot_{i}" for i in range(num_robots)]
    agent_ids_str = str(agent_ids)

    pkg_share = FindPackageShare("multi_agent_search")
    pkg_share_path = context.perform_substitution(pkg_share)

    # --- Simulation ---
    floorplan_config_path = _resolve_config(f"{pkg_share_path}/config/config.toml", num_robots, ".toml")
    robot_config_path = _resolve_config(f"{pkg_share_path}/config/robot_config.yaml", num_robots, ".yaml")

    floorplan_generator_launch_file = PathJoinSubstitution(
        [FindPackageShare("floorplan_generator_stage"), "launch", "floorplan_generator_stage.launch.py"]
    )
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(floorplan_generator_launch_file),
        launch_arguments={
            "floorplan_config_path": floorplan_config_path,
            "robot_config_path": robot_config_path,
            "publish_ground_truth_map": "true",
            "one_tf_tree": "true",
            "publish_initial_poses": known_initial_poses,
        }.items(),
    )

    # Stage monitor exits once /clock is received, gating downstream nodes
    stage_monitor = Node(
        package="multi_agent_search",
        executable="stage_monitor",
        name="stage_monitor",
    )

    # --- Stage 1: Localization (launched after Stage is ready) ---

    localization_launch_file = PathJoinSubstitution([pkg_share, "launch", "localization.launch.py"])
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch_file),
        launch_arguments={
            "use_known_map": use_known_map,
            "agent_ids": agent_ids_str,
            "known_initial_poses": known_initial_poses,
        }.items(),
    )

    loc_node_name = "amcl" if use_known_map == "true" else "slam_toolbox"
    localization_monitor = Node(
        package="multi_agent_search",
        executable="lifecycle_monitor",
        name="localization_monitor",
        parameters=[
            {"node_names": [f"{agent_id}/{loc_node_name}" for agent_id in agent_ids]},
            {"timeout": 60.0},
        ],
    )

    on_stage_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=stage_monitor,
            on_exit=[localization_monitor, localization_launch],
        )
    )

    # --- Stage 2: Search system (launched after localization is active) ---

    comms_manager = LifecycleNode(
        package="multi_agent_search",
        executable="comms_manager",
        name="comms_manager",
        namespace="",
        parameters=[
            {"agent_ids": agent_ids},
            {"use_known_map": use_known_map == "true"},
        ],
    )

    target_positions = "[[-6, 2]]"
    agent_nodes: list[LifecycleNode] = []
    for agent_id in agent_ids:
        remappings = []
        if use_known_map == "true":
            remappings.append((f"/{agent_id}/map", "/ground_truth_map"))
            remappings.append((f"/{agent_id}/pose", f"/{agent_id}/amcl_pose"))

        agent_nodes.append(
            LifecycleNode(
                package="multi_agent_search",
                executable="example_agent",
                name=agent_id,
                namespace="",
                remappings=remappings,
                parameters=[
                    {"agent_id": agent_id},
                    {"use_known_map": use_known_map == "true"},
                    {"known_initial_poses": known_initial_poses == "true"},
                    {"target_positions": target_positions},
                ],
            )
        )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="search_lifecycle_manager",
        parameters=[
            {
                "node_names": ["comms_manager", *agent_ids],
                "autostart": True,
                "bond_timeout": 0.0,
            }
        ],
    )

    search_monitor = Node(
        package="multi_agent_search",
        executable="lifecycle_monitor",
        name="search_monitor",
        parameters=[
            {"node_names": ["comms_manager", *agent_ids]},
            {"timeout": 60.0},
        ],
    )

    on_localization_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=localization_monitor,
            on_exit=[comms_manager, *agent_nodes, lifecycle_manager, search_monitor],
        )
    )

    # --- Stage 3: Navigation (launched after search system is active) ---

    navigation_launch_file = PathJoinSubstitution([pkg_share, "launch", "navigation.launch.py"])
    navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(navigation_launch_file),
        launch_arguments={
            "use_known_map": use_known_map,
            "agent_ids": agent_ids_str,
        }.items(),
    )

    on_search_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=search_monitor,
            on_exit=[navigation_launch],
        )
    )

    return [stage_monitor, simulation_launch, on_stage_ready, on_localization_ready, on_search_ready]


def generate_launch_description() -> LaunchDescription:
    """Generate the launch description for the test."""
    use_known_map_arg = DeclareLaunchArgument(
        "use_known_map",
        default_value="true",
        description="If true, use AMCL with known map; if false, use slam_toolbox for SLAM",
    )
    known_initial_poses_arg = DeclareLaunchArgument(
        "known_initial_poses",
        default_value="true",
        description="If true, publish initial poses from floorplan generator and disable AMCL's default initial pose",
    )
    num_robots_arg = DeclareLaunchArgument(
        "num_robots",
        default_value="3",
        description="Number of robots to spawn (named robot_0, robot_1, ...)",
    )

    floorplan_gen_world = PathJoinSubstitution([FindPackageShare("floorplan_generator_stage"), "world"])
    set_stagepath = SetEnvironmentVariable("STAGEPATH", floorplan_gen_world)

    return LaunchDescription(
        [
            use_known_map_arg,
            known_initial_poses_arg,
            num_robots_arg,
            OpaqueFunction(function=_validate_args),
            set_stagepath,
            OpaqueFunction(function=_launch_system),
        ]
    )
