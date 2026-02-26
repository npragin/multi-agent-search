# Multi-Agent Search

A ROS2 workspace for developing and testing multi-agent search algorithms in procedurally generated 2D environments. Built on the [Stage](https://github.com/rtv/Stage) simulator with Nav2 for navigation and slam_toolbox/AMCL for localization.

The system supports variable numbers of robots and targets, configurable environment generation, and distance-based communication constraints, making it straightforward to run experiments across different environment types, team sizes, and target counts. It supports known and unknown map modes, with known map mode additionally supporting known and unknown initial robot poses. Target detection, belief tracking, and map/belief fusion are handled automatically, letting you focus on the search algorithm itself.

This repository contains multiple packages. The core package discussed in this README is `multi_agent_search`.

## Setup

**Requirements:** ROS2 Kilted, Python 3.12

Clone with submodules:

```bash
git clone --recurse-submodules <repo-url>
```

Build from the workspace root:

```bash
cd multi_agent_search
colcon build --packages-select floorplan_generator_stage
colcon build --symlink-install --packages-select multi_agent_search_interfaces multi_agent_search
source install/setup.bash
```

> **Note:** The `floorplan_generator_stage` package **cannot** be built using the `--symlink-install` flag.

## Usage

Launch the full system:

```bash
ros2 launch multi_agent_search multi_agent_search.launch.py
```

### Launch Arguments

| Parameter | Default | Description |
|---|---|---|
| `num_robots` | `3` | Number of robots |
| `num_targets` | `1` | Number of targets placed in the environment |
| `target_radius` | `0.5` | Detection radius of each target (meters) |
| `use_known_map` | `true` | `true`: Use AMCL and ground truth map<br>`false`: Use slam_toolbox for SLAM |
| `known_initial_poses` | `true` | `true`: Publish initial poses to AMCL (requires `use_known_map:=true`)<br>`false`: Use uniform prior for initial poses |

Example with 5 robots and 3 targets:

```bash
ros2 launch multi_agent_search multi_agent_search.launch.py num_robots:=5 num_targets:=3
```

> **Note:** Unknown map mode (`use_known_map:=false`) is under active development and not yet working.

## Implementing Your Own Agent

Subclass `AgentBase` and implement the required hooks:

```python
from multi_agent_search.agent_base import AgentBase
from multi_agent_search.types import BaseCoordinationMessage, HeartbeatMessage

class MyAgent(AgentBase):
    def __init__(self) -> None:
        super().__init__("MyAgent")

    def on_heartbeat(self, msg: HeartbeatMessage) -> None: ...
    def on_coordination(self, msg: BaseCoordinationMessage) -> None: ...
    def on_lidar_scan(self, scan: LaserScan) -> None: ...
    def on_target_detected(self, target_locations: list[tuple[float, float]]) -> None: ...
```

### Required Hooks

| Hook | Description |
|---|---|
| `on_heartbeat(msg)` | Received heartbeat with another agent's pose |
| `on_coordination(msg)` | Received coordination message |
| `on_lidar_scan(scan)` | Received lidar scan, called after updating internal target belief state |
| `on_target_detected(targets)` | Target found, receives list of `(x, y)` positions in robot's map frame |

### Optional Hooks

| Hook | Description |
|---|---|
| `on_map_updated()` | Received a new map, only called once when `use_known_map` is true |
| `on_fusion_completed()` | Completed map/belief fusion with another agent |
| `on_navigation_succeeded()` | Nav2 reached the goal |
| `on_navigation_failed(reason)` | Nav2 failed to reach the goal |
| `on_navigation_feedback(feedback)` | Nav2 progress feedback |

### Useful Functions

| Function | Description |
|---|---|
| `publish_heartbeat()` | Broadcast a heartbeat message containing the current pose |
| `publish_coordination_message(msg)` | Send a custom `BaseCoordinationMessage` implementation |
| `navigate_to(goal)` | Navigate to a goal pose using Nav2 |
| `cancel_navigation()` | Cancel active navigation |

### Useful Attributes

| Attribute | Description |
|---|---|
| `self.agent_id` | This robot's ID (e.g., `robot_0`) |
| `self.current_pose` | Current estimated pose |
| `self.map` / `self.map_info` | Occupancy grid and metadata |
| `self.belief` | Log-odds belief grid over target positions |
| `self.eliminated` | Boolean grid of cells eliminated as possible target locations |
| `self.nav_status` | Current `NavStatus` (IDLE, NAVIGATING, SUCCEEDED, FAILED) |

### Wiring Your Agent

1. Add an entry point in `multi_agent_search/setup.cfg`
2. Change the `executable` in `_create_agent_nodes()` in `multi_agent_search/launch/multi_agent_search.launch.py`

See `multi_agent_search/multi_agent_search/example_agent.py` for a minimal reference implementation.

## Communication Model

All inter-agent communication is mediated by the `CommsManager`, which enforces distance-based constraints using ground truth positions:

| Zone | Distance | Behavior |
|---|---|---|
| **Close Range** | ≤ 5m | Belief/map fusion (requires LoS) and message delivery at 10 Hz |
| **Long Range** | 5-20m | Message delivery at 1 Hz |
| **Blackout** | > 20m | No communication |

Thresholds and rates are configurable via parameters to the `comms_manager.py` node.

### Custom Coordination Messages

Define your own message types by subclassing `BaseCoordinationMessage`:

```python
@dataclass
class MyCoordMsg(BaseCoordinationMessage):
    task_id: int = 0
    target_cell: tuple[int, int] = (0, 0)
```

Publish using `publish_coordination_msg(BaseCoordinationMessage(0, (5, 5)))`.

## Architecture

### Key Components

- **`CommsManager`** — Lifecycle node tracking agent positions and enforcing communication zones
- **`AgentBase`** — Abstract lifecycle node providing Nav2 integration, belief tracking, lidar processing, and communication
- **`TargetDetector`** — Monitors lidar scans and detects when robots find targets
- **`floorplan-generator-stage`** — Submodule that procedurally generates office-style environments for Stage

### Launch

The system launches in four gated phases:

1. **Simulation** — Procedurally generates the environment and spawns Stage
2. **Localization** — Starts AMCL or slam_toolbox per robot
3. **Search System** — CommsManager, TargetDetector, and agent nodes
4. **Navigation** — Nav2 servers (controller, planner, behavior, BT navigator)

Each phase waits for the previous one to be fully active before starting.

### Environment Configuration

Environment generation is controlled by `multi_agent_search/config/config.toml`. Parameters include room count, hallway width, obstacle density, and more.

To change the robot model used in the simulation, check out the [README in the floorplan-generator-stage package](https://github.com/npragin/floorplan-generator-stage/blob/main/README.md) for instructions on updating this package.