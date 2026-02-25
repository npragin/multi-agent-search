"""Launch file for testing the multi-agent search system."""

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


def _validate_args(context: LaunchContext) -> list[Node]:
    """Validate that known_initial_poses is not true when use_known_map is false."""
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower() == "true"
    known_initial_poses = context.perform_substitution(LaunchConfiguration("known_initial_poses")).lower() == "true"
    if known_initial_poses and not use_known_map:
        raise ValueError("known_initial_poses requires use_known_map to be true (AMCL must be running)")
    return []


def _launch_brain_dead_agents(context: LaunchContext) -> list[LifecycleNode]:
    """Launch the brain dead agents."""
    use_known_map = context.perform_substitution(LaunchConfiguration("use_known_map")).lower()
    known_initial_poses = context.perform_substitution(LaunchConfiguration("known_initial_poses")).lower()

    agent_ids = ["robot_0", "robot_1", "robot_2"]
    target_positions = "[[-6, 2]]"
    nodes = []
    for agent_id in agent_ids:
        remappings = []
        if use_known_map == "true":
            remappings.append((f"/{agent_id}/map", "/ground_truth_map"))
            remappings.append((f"/{agent_id}/pose", f"/{agent_id}/amcl_pose"))

        nodes.append(
            LifecycleNode(
                package="multi_agent_search",
                executable="brain_dead_agent",
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
    return nodes


def generate_launch_description() -> LaunchDescription:
    """Generate the launch description for the test."""
    use_known_map_arg = DeclareLaunchArgument(
        "use_known_map",
        default_value="true",
        description="If true, use AMCL with known map; if false, use slam_toolbox for SLAM",
    )
    use_known_map = LaunchConfiguration("use_known_map")

    known_initial_poses_arg = DeclareLaunchArgument(
        "known_initial_poses",
        default_value="true",
        description="If true, publish initial poses from floorplan generator and disable AMCL's default initial pose",
    )
    known_initial_poses = LaunchConfiguration("known_initial_poses")

    pkg_share = FindPackageShare("multi_agent_search")

    # Floorplan config path for floorplan generation
    floorplan_config_path = PathJoinSubstitution([pkg_share, "config", "config.toml"])

    # Robot config path - defines header and robot templates
    robot_config_path = PathJoinSubstitution([pkg_share, "config", "robot_config.yaml"])

    # Path to the floorplan_generator_stage launch file
    floorplan_launch_file = PathJoinSubstitution(
        [
            FindPackageShare("floorplan_generator_stage"),
            "launch",
            "floorplan_generator_stage.launch.py",
        ]
    )

    # Launch the floorplan generator with one_tf_tree enabled
    floorplan_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(floorplan_launch_file),
        launch_arguments={
            "floorplan_config_path": floorplan_config_path,
            "robot_config_path": robot_config_path,
            "publish_ground_truth_map": "true",
            "one_tf_tree": "true",
            "publish_initial_poses": known_initial_poses,
        }.items(),
    )

    # Localization launch (AMCL or slam_toolbox depending on use_known_map)
    localization_launch_file = PathJoinSubstitution([pkg_share, "launch", "localization.launch.py"])
    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(localization_launch_file),
        launch_arguments={
            "use_known_map": use_known_map,
            "agent_ids": "['robot_0', 'robot_1', 'robot_2']",
            "known_initial_poses": known_initial_poses,
        }.items(),
    )

    comms_manager = LifecycleNode(
        package="multi_agent_search",
        executable="comms_manager",
        name="comms_manager",
        namespace="",
        parameters=[
            {"num_agents": 3},
            {"agent_ids": ["robot_0", "robot_1", "robot_2"]},
            {"use_known_map": use_known_map},
        ],
    )

    # nav2_lifecycle_manager handles ordered configure -> activate transitions
    lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="search_lifecycle_manager",
        parameters=[
            {
                "node_names": ["comms_manager", "robot_0", "robot_1", "robot_2"],
                "autostart": True,
                "bond_timeout": 0.0,
            }
        ],
    )

    # Stage monitor exits once /clock is received, gating downstream nodes
    stage_monitor = Node(
        package="multi_agent_search",
        executable="stage_monitor",
        name="stage_monitor",
    )

    # Stage needs STAGEPATH to find its assets (robots.inc, bitmaps, etc.)
    # All world files are consolidated in floorplan_generator_stage/world/ by the launch file
    floorplan_gen_world = PathJoinSubstitution([FindPackageShare("floorplan_generator_stage"), "world"])
    set_stagepath = SetEnvironmentVariable("STAGEPATH", floorplan_gen_world)

    # When stage_monitor exits (Stage is up), launch localization, comms, agents, and lifecycle manager
    on_stage_ready = RegisterEventHandler(
        OnProcessExit(
            target_action=stage_monitor,
            on_exit=[
                localization_launch,
                comms_manager,
                OpaqueFunction(function=_launch_brain_dead_agents),
                lifecycle_manager,
            ],
        )
    )

    return LaunchDescription(
        [
            use_known_map_arg,
            known_initial_poses_arg,
            OpaqueFunction(function=_validate_args),
            set_stagepath,
            floorplan_launch,
            stage_monitor,
            on_stage_ready,
        ]
    )
