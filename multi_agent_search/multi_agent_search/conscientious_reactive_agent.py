"""
Coarse-Grid Idleness Patrolling Agent with Heartbeat Avoidance
==============================================================

Core idea ("the heart"):
    1. Divide the occupancy grid into coarse tiles (kernel_size × kernel_size pixels)
    2. Track last-visit time per tile
    3. Always move to the stalest neighboring tile
    4. Use heartbeats to avoid tiles occupied by other robots

See architecture_liam.md for the full system walkthrough.

Usage:
    ros2 launch multi_agent_search multi_agent_search.launch.py \\
        agent_executable:=cr_agent
"""

from __future__ import annotations

from collections import deque

import numpy as np
import random
import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from multi_agent_search.agent_base import AgentBase
from multi_agent_search.types import HeartbeatMessage, NavStatus

# REMOVED: from enum import IntEnum
# REMOVED: IntEnum unused — no enum types in this agent

# REMOVED: from heapq import heappop, heappush
# REMOVED: heapq unused — neighbor selection uses linear scan + random tiebreak


# REMOVED: Hardcoded node/box/graph definitions that were commented out in
# the original __init__ (lines 44-98). Replaced by dynamic coarse graph built
# from the occupancy grid at runtime. See architecture_liam.md §5 "Coarse graph"
#
# Original definitions mapped fixed world-coordinate boxes to letter-named
# nodes (A-N) with a hand-drawn adjacency graph. The dynamic approach builds
# the same structure automatically from any procedurally generated map.


class CR_Agent(AgentBase):
    """Coarse-grid idleness patroller with heartbeat-based avoidance.

    Divides the occupancy grid into large tiles, tracks per-tile staleness,
    and always navigates to the stalest neighboring tile — unless another
    robot's heartbeat says they're already there.
    """

    def __init__(self) -> None:
        super().__init__("CR_Agent")

        self.robot_cell = None

        self.declare_parameter("planning_interval", 2.0)
        self.planning_interval = self.get_parameter("planning_interval").get_parameter_value().double_value

        self.goal_viz_pub = self.create_publisher(PoseStamped, 'visualization_goal', 10)

        qos_profile = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.gt_map = None

        self.graph = None
        self.tile_timers = None
        # kernel_size is in occupancy grid cells (pixels), NOT meters.
        # 60 cells * 0.05 m/cell resolution = 3.0 m per coarse tile.
        self.kernel_size = 60
        self.threshold = 0.25
        self._graph_pruned = False  # See architecture_liam.md §5 "Reachability pruning"

        # Heartbeat avoidance state — See architecture_liam.md §5 "Heartbeat avoidance"
        self.other_robot_cells: dict[str, tuple[tuple[int, int], float]] = {}
        self.heartbeat_timeout = 5.0  # seconds — ignore stale positions

        # Goal-send cooldown — prevents rapid duplicate goal sends.
        # See architecture_liam.md §5 "Exploration loop"
        self._last_goal_send_time = None
        self._goal_cooldown = 2.0  # seconds — min interval between navigate_to calls

        self.gt_map_sub = self.create_subscription(
            OccupancyGrid,
            '/ground_truth_map',
            self.gt_map_callback,
            qos_profile,
        )

        self.locate_robot_timer = self.create_timer(1.0, self._locate_update_robot_cell)
        # ORIGINAL: # self.exploration_timer = self.create_timer(1.0, self._perform_exploration)
        # CHANGED: Timer was disabled during development. Enabling it starts the
        #          planning loop. See architecture_liam.md §5 "Exploration loop"
        self.exploration_timer = self.create_timer(1.0, self._perform_exploration)

    # ── Map & Graph ──────────────────────────────────────────────────────

    def gt_map_callback(self, msg):
        self.gt_map = msg
        self.get_logger().info(f'height: {self.gt_map.info.height}')
        self.get_logger().info(f'width: {self.gt_map.info.width}')
        self.get_logger().info(f'resolution: {self.gt_map.info.resolution}')
        if self.graph is None:
            self.graph = self.create_graph(msg, self.kernel_size, self.threshold).astype(int)
            self.tile_timers = self.create_tile_timers()
            np.savetxt('grid.txt', self.graph, fmt='%d')

    def create_tile_timers(self):
        """Create per-tile last-visit timestamps, initialized to now."""
        if self.graph is None:
            return None
        now = self.get_clock().now()
        timers = np.empty(self.graph.shape, dtype=object)
        for i in range(self.graph.shape[0]):
            for j in range(self.graph.shape[1]):
                timers[i, j] = now
        return timers

    def create_graph(self, msg, n, threshold):
        """Build coarse traversability graph from occupancy grid.

        Divides the grid into n×n pixel blocks. A block is traversable (0)
        if ≥ threshold fraction of its cells are free (value 0 in the
        occupancy grid). Unknown cells (-1, exterior) and occupied cells
        (100) both count against the free ratio.

        See architecture_liam.md §5 "Coarse graph"
        """
        grid = np.array(msg.data).reshape(msg.info.height, msg.info.width)

        h, w = grid.shape
        out_h = h // n
        out_w = w // n

        result = np.zeros((out_h, out_w), dtype=int)

        for i in range(out_h):
            for j in range(out_w):
                block = grid[i * n : (i + 1) * n, j * n : (j + 1) * n]
                zero_ratio = np.sum(block == 0) / (n * n)
                if zero_ratio >= threshold:
                    result[i, j] = 0
                else:
                    result[i, j] = 1

        return result

    def _prune_unreachable_tiles(self) -> None:
        """BFS from robot_cell — mark unreachable traversable tiles as blocked.

        Maps are procedurally generated with irregular shapes (serpentine
        hallways, rooms). Some coarse tiles may pass the free-ratio threshold
        but be isolated behind walls with no path from the robot. This BFS
        ensures only connected tiles are considered.

        See architecture_liam.md §2 "Map Generation & Shape"
        See architecture_liam.md §5 "Reachability pruning"
        """
        if self.robot_cell is None or self.graph is None:
            return
        visited = set()
        queue = deque([self.robot_cell])
        visited.add(self.robot_cell)
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in visited and 0 <= nr < self.graph.shape[0] and 0 <= nc < self.graph.shape[1]:
                    if self.graph[nr, nc] == 0:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
        # Block everything not reached
        pruned = 0
        for i in range(self.graph.shape[0]):
            for j in range(self.graph.shape[1]):
                if self.graph[i, j] == 0 and (i, j) not in visited:
                    self.graph[i, j] = 1
                    pruned += 1
        self.get_logger().info(
            f"Pruned {pruned} unreachable tiles, {len(visited)} reachable tiles remain"
        )

    # ── Exploration Loop ─────────────────────────────────────────────────

    def _perform_exploration(self):
        """Planning tick: publish heartbeat, pick stalest neighbor, navigate.

        See architecture_liam.md §5 "Exploration loop"
        """
        if self.current_pose is None:
            self.get_logger().info('Current Pose is None')
            return
        if self.graph is None:
            self.get_logger().info('Graph is None')
            return

        # ORIGINAL: # self.publish_heartbeat()
        # CHANGED: Heartbeats are required for multi-robot avoidance. Other robots'
        #          on_heartbeat() uses our pose to avoid our tile.
        #          See architecture_liam.md §4 "Communication Model"
        self.publish_heartbeat()

        # One-time reachability prune after graph + robot_cell are both ready
        if not self._graph_pruned and self.robot_cell is not None:
            self._prune_unreachable_tiles()
            self._graph_pruned = True

        if self.nav_status == NavStatus.NAVIGATING:
            return

        # Cooldown: don't send a new goal if we just sent one.
        # Between _send_nav_goal() and _on_nav_goal_response(), _current_nav_goal
        # is still None and nav_status hasn't changed — without this guard, the
        # next timer tick would send a duplicate goal.
        if self._last_goal_send_time is not None:
            elapsed = (self.get_clock().now().nanoseconds - self._last_goal_send_time) / 1e9
            if elapsed < self._goal_cooldown:
                return

        # Guard: don't send goals before Nav2 action server is available.
        # Nav2 starts in Phase 4 (after search system monitor completes).
        # Goals sent before the server exists will fail silently.
        if not self._nav_client.server_is_ready():
            self.get_logger().info('Nav2 action server not ready')
            return

        neighbor_cell = self._get_most_neglected_neighbor_box()
        if not neighbor_cell:
            self.get_logger().info('No neighbors')
            return
        goal_x, goal_y = self.graph_cell_center_to_stage(*neighbor_cell)
        self.get_logger().info(f'Navigating to cell {neighbor_cell} -> world ({goal_x:.2f}, {goal_y:.2f})')
        goal = self._pose_from_xy(goal_x, goal_y)
        self.goal_viz_pub.publish(goal)
        self._last_goal_send_time = self.get_clock().now().nanoseconds
        self.navigate_to(goal)

    # ── Neighbor Selection ───────────────────────────────────────────────

    def _get_most_neglected_neighbor_box(self):
        """Pick the stalest traversable neighbor tile, avoiding other robots.

        1. Gather 4-connected traversable neighbors of robot's current tile
        2. Build a blocked set from heartbeat data (other robot's tile + 1-hop)
        3. Prefer unblocked neighbors; fall back to all if everything blocked
        4. Among candidates, pick the one with the longest idle time
        5. Break ties randomly

        See architecture_liam.md §5 "Neighbor selection" and "Heartbeat avoidance"
        """
        if self.robot_cell is None or self.graph is None or self.tile_timers is None:
            return None
        i, j = self.robot_cell

        # 4-connected neighbors (up, down, left, right)
        neighbors = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.graph.shape[0] and 0 <= nj < self.graph.shape[1]:
                if self.graph[ni, nj] == 0:
                    neighbors.append((ni, nj))
        if not neighbors:
            return None

        # Build blocked cells from heartbeat data
        # Each other robot blocks its tile + 1-hop graph neighbors
        # See architecture_liam.md §5 "Heartbeat avoidance"
        now_sec = self.get_clock().now().nanoseconds / 1e9
        blocked: set[tuple[int, int]] = set()
        for agent_id, (cell, ts) in list(self.other_robot_cells.items()):
            if now_sec - ts > self.heartbeat_timeout:
                del self.other_robot_cells[agent_id]
                continue
            blocked.add(cell)
            ci, cj = cell
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = ci + di, cj + dj
                if 0 <= ni < self.graph.shape[0] and 0 <= nj < self.graph.shape[1]:
                    blocked.add((ni, nj))

        # Prefer unblocked neighbors; fall back if all blocked (no deadlock)
        unblocked = [n for n in neighbors if n not in blocked]
        candidates = unblocked if unblocked else neighbors

        self.get_logger().info(
            f'robot_cell=({i},{j}), {len(neighbors)} traversable neighbors, '
            f'{len(blocked)} blocked cells'
        )

        # Find the candidate(s) with the largest time since last visit
        now = self.get_clock().now()
        max_age = -1.0
        oldest_neighbors = []
        for ni, nj in candidates:
            last_visit = self.tile_timers[ni, nj]
            age = (now.nanoseconds - last_visit.nanoseconds) / 1e9
            if age > max_age:
                max_age = age
                oldest_neighbors = [(ni, nj)]
            elif age == max_age:
                oldest_neighbors.append((ni, nj))

        chosen = random.choice(oldest_neighbors)
        self.get_logger().info(
            f'Chosen neighbor: {chosen} (age={max_age:.1f}s, '
            f'{len(unblocked)}/{len(neighbors)} unblocked)'
        )
        return chosen

    # ── Coordinate Conversion ────────────────────────────────────────────

    def graph_cell_center_to_stage(self, ni, nj):
        """Convert coarse tile (ni, nj) to world (x, y) at the centroid of free cells.

        ORIGINAL: Used geometric center — (nj+0.5)*kernel_size, (ni+0.5)*kernel_size.
        CHANGED: Geometric center can land in a wall if the tile is only partially
                 free (threshold=0.25 means 75% can be walls). Now computes centroid
                 of free cells (value==0) within the tile block. Falls back to
                 geometric center if no free cells found.
                 See architecture_liam.md §5 "Coarse graph"
        """
        if self.gt_map is None:
            return None, None
        origin_x = self.gt_map.info.origin.position.x
        origin_y = self.gt_map.info.origin.position.y
        resolution = self.gt_map.info.resolution
        k = self.kernel_size

        # Extract the occupancy grid block for this tile
        grid = np.array(self.gt_map.data).reshape(
            self.gt_map.info.height, self.gt_map.info.width
        )
        row_start, row_end = ni * k, (ni + 1) * k
        col_start, col_end = nj * k, (nj + 1) * k
        block = grid[row_start:row_end, col_start:col_end]

        # Find free cells (value == 0) and compute their centroid in pixel coords
        free_rows, free_cols = np.where(block == 0)
        if len(free_rows) == 0:
            # Fallback: geometric center (shouldn't happen for traversable tiles)
            pixel_x = (nj + 0.5) * k
            pixel_y = (ni + 0.5) * k
        else:
            pixel_y = row_start + free_rows.mean()
            pixel_x = col_start + free_cols.mean()

        x = origin_x + pixel_x * resolution
        y = origin_y + pixel_y * resolution
        return x, y

    def get_kernel_indices_from_xy(self, x, y):
        """Convert world (x, y) to coarse graph cell (i, j).

        Returns (i, j) or (None, None) if out of bounds.
        """
        if self.gt_map is None or self.graph is None:
            return None, None
        origin_x = self.gt_map.info.origin.position.x
        origin_y = self.gt_map.info.origin.position.y
        resolution = self.gt_map.info.resolution
        kernel_size = self.kernel_size

        pixel_x = (x - origin_x) / resolution
        pixel_y = (y - origin_y) / resolution

        i = int(pixel_y // kernel_size)
        j = int(pixel_x // kernel_size)

        if 0 <= i < self.graph.shape[0] and 0 <= j < self.graph.shape[1]:
            return i, j
        else:
            return None, None

    # ── Robot Tracking ───────────────────────────────────────────────────

    def _locate_update_robot_cell(self):
        """Convert robot pose to coarse graph cell and stamp tile timer.

        See architecture_liam.md §5 "Robot tracking"
        """
        if self.current_pose is None:
            return None
        stage_x = self.current_pose.position.x
        stage_y = self.current_pose.position.y
        self.get_logger().info(f'robot_pose: ({stage_x}, {stage_y})')

        # ORIGINAL: if self.tile_timers is None or self.robot_cell is None:
        # CHANGED: robot_cell starts as None — checking it here prevents first
        #          initialization. Only tile_timers needs guarding.
        #          See architecture_liam.md §5 "Robot tracking"
        if self.tile_timers is None:
            return

        cell = self.get_kernel_indices_from_xy(stage_x, stage_y)

        if cell[0] is not None:
            self.robot_cell = cell
            self.tile_timers[cell[0], cell[1]] = self.get_clock().now()
            self.get_logger().info(f'robot cell: {self.robot_cell}')

    # ── Lifecycle ────────────────────────────────────────────────────────

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info("Activating CR_Agent")
        result = super().on_activate(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result
        return TransitionCallbackReturn.SUCCESS

    def _pose_from_xy(self, x: float, y: float) -> PoseStamped:
        """Create a PoseStamped goal at (x, y) in the map frame."""
        pose = PoseStamped()
        # ORIGINAL: pose.header.frame_id = "map" if self.use_known_map else self.agent_id + "/map"
        # CHANGED: AgentBase already computes this as self._map_frame (agent_base.py:229)
        pose.header.frame_id = self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        # ORIGINAL: pose.pose.orientation.x = 0.0  (and y, z)
        # REMOVED: PoseStamped defaults are already 0.0 for x/y/z
        pose.pose.orientation.w = 1.0
        return pose

    # ── Communication Hooks ──────────────────────────────────────────────

    def on_heartbeat(self, msg: HeartbeatMessage) -> None:
        """Project sender's pose onto coarse graph, store for avoidance.

        See architecture_liam.md §4 "Communication Model"
        See architecture_liam.md §5 "Heartbeat avoidance"
        """
        # ORIGINAL: pass
        # CHANGED: Now projects heartbeat sender's position onto our coarse
        #          graph so _get_most_neglected_neighbor_box() can avoid
        #          tiles occupied by other robots.
        if msg.sender_id == self.agent_id:
            return
        cell = self.get_kernel_indices_from_xy(msg.pose.position.x, msg.pose.position.y)
        if cell[0] is not None:
            self.other_robot_cells[msg.sender_id] = (cell, self.get_clock().now().nanoseconds / 1e9)

    # ── Navigation Result Hooks ─────────────────────────────────────────

    def on_navigation_succeeded(self) -> None:
        """Clear goal cooldown so next planning tick sends immediately."""
        self._last_goal_send_time = None
        self.get_logger().info('Navigation succeeded — ready for next goal')

    def on_navigation_failed(self, error_msg: str) -> None:
        """Log failure. Cooldown stays active to prevent rapid retry of bad goals."""
        self.get_logger().warning(f'Navigation failed: {error_msg}')

    # on_coordination(), on_lidar_scan(), on_target_detected() are @abstractmethod
    # in AgentBase (agent_base.py:967-1003) — must be implemented even if no-op.

    def on_coordination(self, msg) -> None:
        pass

    def on_lidar_scan(self, scan) -> None:
        pass

    def on_target_detected(self, target_locations: list[tuple[float, float]]) -> None:
        pass


# ── Entry point ──────────────────────────────────────────────────────────


def main(args: list[str] | None = None) -> None:
    """Entry point for the CR_Agent."""
    rclpy.init(args=args)
    agent = CR_Agent()
    executor = MultiThreadedExecutor()
    executor.add_node(agent)
    executor.spin()
    agent.destroy_node()
    rclpy.shutdown()