"""Launch file for testing the multi-agent search system."""

import tempfile
from pathlib import Path

import yaml

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

NUM_ROBOTS_PLACEHOLDER = "{{NUM_ROBOTS}}"
NUM_TARGETS_PLACEHOLDER = "{{NUM_TARGETS}}"
TARGET_RADIUS_PLACEHOLDER = "{{TARGET_RADIUS}}"


def _validate_args(context: LaunchContext) -> list[Node]:
    """Validate that known_initial_poses is not true when use_known_map is false."""
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower() == "true"
    known_initial_poses = context.perform_substitution(LaunchConfiguration("known_initial_poses")).lower() == "true"
    if known_initial_poses and not use_known_map:
        raise ValueError("known_initial_poses requires use_known_map to be true (AMCL must be running)")
    return []


def _resolve_config(template_path: str, replacements: dict[str, str], suffix: str) -> str:
    """Read a config template, replace placeholders, and write to a temp file."""
    content = Path(template_path).read_text()
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as tmp:
        tmp.write(content)
    return tmp.name


def _launch_system(context: LaunchContext) -> list[Node | LifecycleNode]:
    """
    Launch the full system: simulation, stage monitor, and post-stage nodes.

    Resolves num_robots to generate config files from templates and create the correct number of agent nodes.
    """
    num_robots = context.perform_substitution(LaunchConfiguration("num_robots"))
    num_targets = context.perform_substitution(LaunchConfiguration("num_targets"))
    target_radius = context.perform_substitution(LaunchConfiguration("target_radius"))
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower()
    known_initial_poses = context.perform_substitution(LaunchConfiguration("known_initial_poses")).lower()

    agent_ids = [f"robot_{i}" for i in range(int(num_robots))]
    agent_ids_str = str(agent_ids)

    pkg_share = FindPackageShare("multi_agent_search")
    pkg_share_path = context.perform_substitution(pkg_share)

    # --- Simulation ---
    replacements = {
        NUM_ROBOTS_PLACEHOLDER: num_robots,
        NUM_TARGETS_PLACEHOLDER: num_targets,
        TARGET_RADIUS_PLACEHOLDER: target_radius,
    }
    floorplan_config_path = _resolve_config(f"{pkg_share_path}/config/config.toml", replacements, ".toml")
    robot_config_path = _resolve_config(f"{pkg_share_path}/config/robot_config.yaml", replacements, ".yaml")

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

    # --- Stage 2 & 3: Deferred until localization is ready (world_config.yaml exists by then) ---

    on_localization_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=localization_monitor,
            on_exit=[OpaqueFunction(function=_launch_search_and_nav)],
        )
    )

    return [stage_monitor, simulation_launch, on_stage_ready, on_localization_ready]


def _launch_search_and_nav(context: LaunchContext) -> list[Node | LifecycleNode | RegisterEventHandler]:
    """
    Launch the search system and navigation stack.

    Deferred to an OpaqueFunction so that world_config.yaml (generated by the floorplan generator)
    is available for reading target positions.
    """
    num_robots = int(context.perform_substitution(LaunchConfiguration("num_robots")))
    target_radius = float(context.perform_substitution(LaunchConfiguration("target_radius")))
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower()
    known_initial_poses = context.perform_substitution(LaunchConfiguration("known_initial_poses")).lower()

    agent_ids = [f"robot_{i}" for i in range(num_robots)]
    agent_ids_str = str(agent_ids)

    pkg_share = FindPackageShare("multi_agent_search")

    # --- Read target positions from generated world_config.yaml ---
    fg_share_path = context.perform_substitution(FindPackageShare("floorplan_generator_stage"))
    world_config_path = Path(fg_share_path) / "output" / "world_config.yaml"
    with open(world_config_path) as f:
        world_config = yaml.safe_load(f)
    extra_points = world_config.get("extra_points", [])
    target_positions = str([[p["x"], p["y"]] for p in extra_points])

    # --- Search system ---

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
                ],
            )
        )

    target_detector = LifecycleNode(
        package="multi_agent_search",
        executable="target_detector",
        name="target_detector",
        namespace="",
        parameters=[
            {"agent_ids": agent_ids},
            {"target_positions": target_positions},
            {"target_radius": target_radius},
            {"scan_topic": "base_scan"},
            {"use_known_map": use_known_map == "true"},
        ],
    )

    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="search_lifecycle_manager",
        parameters=[
            {
                "node_names": ["comms_manager", "target_detector", *agent_ids],
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
            {"node_names": ["comms_manager", "target_detector", *agent_ids]},
            {"timeout": 60.0},
        ],
    )

    # --- Navigation (launched after search system is active) ---

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

    return [comms_manager, target_detector, *agent_nodes, lifecycle_manager, search_monitor, on_search_ready]


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
    num_targets_arg = DeclareLaunchArgument(
        "num_targets",
        default_value="1",
        description="Number of target points to place in the environment",
    )
    target_radius_arg = DeclareLaunchArgument(
        "target_radius",
        default_value="0.5",
        description="Radius of each target point in meters",
    )

    floorplan_gen_world = PathJoinSubstitution([FindPackageShare("floorplan_generator_stage"), "world"])
    set_stagepath = SetEnvironmentVariable("STAGEPATH", floorplan_gen_world)

    return LaunchDescription(
        [
            use_known_map_arg,
            known_initial_poses_arg,
            num_robots_arg,
            num_targets_arg,
            target_radius_arg,
            OpaqueFunction(function=_validate_args),
            set_stagepath,
            OpaqueFunction(function=_launch_system),
        ]
    )
