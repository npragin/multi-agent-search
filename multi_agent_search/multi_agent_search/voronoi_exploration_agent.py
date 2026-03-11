# voronoi-partitioned exploration agent for multi-agent search
# @author: conor gagliardi

from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from scipy.ndimage import uniform_filter
from sensor_msgs.msg import LaserScan

from multi_agent_search.agent_base import AgentBase
from multi_agent_search.types import BaseCoordinationMessage, HeartbeatMessage, NavStatus


@dataclass
class IntentionMessage(BaseCoordinationMessage):
    # broadcasts current nav target
    target_x: float = 0.0
    target_y: float = 0.0


class VoronoiExplorationAgent(AgentBase):
    # voronoi partition + dijkstra frontier search scored by gain over distance

    def __init__(self) -> None:
        super().__init__("VoronoiExplorationAgent")

        self.declare_parameter("planning_cooldown", 2.0)
        self.declare_parameter("num_candidates", 8)
        self.declare_parameter("min_candidate_spacing", 3.0)
        self.declare_parameter("scan_radius", 5.0)
        self.declare_parameter("rendezvous_threshold", 500)
        self.declare_parameter("max_rendezvous_distance", 20.0)

        self.planning_cooldown: float = self.get_parameter("planning_cooldown").value
        self.num_candidates: int = self.get_parameter("num_candidates").value
        self.min_candidate_spacing: float = self.get_parameter("min_candidate_spacing").value
        self.scan_radius: float = self.get_parameter("scan_radius").value
        self.rendezvous_threshold: int = self.get_parameter("rendezvous_threshold").value
        self.max_rendezvous_distance: float = self.get_parameter("max_rendezvous_distance").value

        self.other_robot_poses: dict[str, tuple[float, float]] = {}
        self.intentions: dict[str, tuple[float, float]] = {}
        self.cells_since_fusion: int = 0
        self._prev_eliminated_count: int = 0

        self.create_timer(self.planning_cooldown, self._planning_tick)
        self.get_logger().info("Initialized")

    # multi-source bfs voronoi partition using path distance
    def _compute_voronoi_partition(self) -> tuple[np.ndarray, int]:
        assert self.map is not None and self.current_pose is not None

        my_pos = (self.current_pose.position.x, self.current_pose.position.y)
        robots: list[tuple[str, tuple[int, int]]] = [
            (self.agent_id, self._world_to_cell(my_pos))
        ]
        for agent_id, pos in self.other_robot_poses.items():
            robots.append((agent_id, self._world_to_cell(pos)))

        my_idx = 0
        rows, cols = self.map.shape
        partition = np.full((rows, cols), -1, dtype=np.int8)

        # seed bfs from all robots
        queue: deque[tuple[int, int]] = deque()
        for i, (_, cell) in enumerate(robots):
            r, c = cell
            if 0 <= r < rows and 0 <= c < cols and self.map[r, c] != 100:
                partition[r, c] = i
                queue.append((r, c))

        # first to reach a cell claims it
        while queue:
            r, c = queue.popleft()
            owner = partition[r, c]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and partition[nr, nc] == -1
                    and self.map[nr, nc] != 100
                ):
                    partition[nr, nc] = owner
                    queue.append((nr, nc))

        return partition, my_idx

    # unsearched free cell density around each cell
    def _compute_gain_map(self) -> np.ndarray:
        assert self.eliminated is not None and self.map is not None and self.map_info is not None

        searchable = (~self.eliminated) & (self.map != 100) & (self.map != -1)
        kernel = max(3, int(self.scan_radius / self.map_info.resolution))
        return uniform_filter(searchable.astype(np.float64), size=2 * kernel + 1, mode="constant")

    # dijkstra from robot to find unsearched cells in our territory
    # tries territory first then searches globally
    def _find_frontier_candidates(
        self,
        partition: np.ndarray,
        my_idx: int,
    ) -> list[tuple[float, tuple[float, float]]]:
        assert self.map is not None and self.map_info is not None
        assert self.eliminated is not None and self.current_pose is not None

        robot_world = (self.current_pose.position.x, self.current_pose.position.y)
        robot_cell = self._world_to_cell(robot_world)
        resolution = self.map_info.resolution
        rows, cols = self.map.shape

        for territory_only in (True, False):
            pq: list[tuple[float, tuple[int, int]]] = [(0.0, robot_cell)]
            visited: set[tuple[int, int]] = set()
            candidates: list[tuple[float, tuple[float, float]]] = []

            while pq and len(candidates) < self.num_candidates:
                dist, cell = heappop(pq)
                if cell in visited or self.map[cell] == 100:
                    continue
                visited.add(cell)

                in_territory = (not territory_only) or (partition[cell] == my_idx)
                if not self.eliminated[cell] and dist > 0.5 and in_territory:
                    world = self._cell_to_world(cell)
                    too_close = any(
                        math.hypot(world[0] - px, world[1] - py) < self.min_candidate_spacing
                        for _, (px, py) in candidates
                    )
                    if not too_close:
                        candidates.append((dist, world))

                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cell[0] + dr, cell[1] + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        heappush(pq, (dist + resolution, (nr, nc)))

            if candidates:
                return candidates

        return []

    # meet nearest teammate if we have enough unshared coverage
    def _should_rendezvous(self) -> tuple[bool, str | None, tuple[float, float] | None]:
        if self.cells_since_fusion < self.rendezvous_threshold:
            return False, None, None
        if not self.other_robot_poses or self.current_pose is None:
            return False, None, None

        mx = self.current_pose.position.x
        my = self.current_pose.position.y

        best_dist = float("inf")
        best_id: str | None = None
        best_pos: tuple[float, float] | None = None

        for agent_id, (rx, ry) in self.other_robot_poses.items():
            d = math.hypot(mx - rx, my - ry)
            if d < best_dist:
                best_dist = d
                best_id = agent_id
                best_pos = (rx, ry)

        if best_dist > self.max_rendezvous_distance:
            return False, None, None

        return True, best_id, best_pos

    # main planning loop
    def _planning_tick(self) -> None:
        if not self._ready_to_plan():
            return

        self.publish_heartbeat()

        # scatter if robots are too close together
        if self._robots_are_clustered():
            target = self._pick_scatter_target()
            if target is not None:
                self.get_logger().info(f"Dispersing to ({target[0]:.1f}, {target[1]:.1f})")
                self.navigate_to(self._pose_from_xy(target[0], target[1]))
            return

        partition, my_idx = self._compute_voronoi_partition()
        n_robots = len(self.other_robot_poses) + 1
        my_territory_size = int(np.sum(partition == my_idx))

        should_rv, rv_agent, rv_pos = self._should_rendezvous()
        if should_rv and rv_pos is not None:
            self.get_logger().info(
                f"Rendezvous with {rv_agent} ({self.cells_since_fusion} unshared cells)"
            )
            self.publish_coordination_message(
                IntentionMessage(target_x=rv_pos[0], target_y=rv_pos[1])
            )
            self.navigate_to(self._pose_from_xy(rv_pos[0], rv_pos[1]))
            return

        gain_map = self._compute_gain_map()

        candidates = self._find_frontier_candidates(partition, my_idx)
        if not candidates:
            self.get_logger().warn("No frontier candidates")
            return

        # pick candidate with best gain/distance ratio
        best_score = -1.0
        best_candidate: tuple[float, float] | None = None

        for dijkstra_dist, world_xy in candidates:
            cell = self._world_to_cell(world_xy)
            r, c = cell
            gain = float(gain_map[r, c]) if 0 <= r < gain_map.shape[0] and 0 <= c < gain_map.shape[1] else 0.0
            score = gain / max(dijkstra_dist, 0.1)
            if score > best_score:
                best_score = score
                best_candidate = world_xy

        if best_candidate is None:
            return

        x, y = best_candidate
        self.publish_coordination_message(IntentionMessage(target_x=x, target_y=y))
        self.get_logger().info(
            f"Frontier ({x:.1f}, {y:.1f}) score={best_score:.4f} "
            f"candidates={len(candidates)} robots={n_robots} territory={my_territory_size}"
        )
        self.navigate_to(self._pose_from_xy(x, y))

    def _ready_to_plan(self) -> bool:
        return all(a is not None for a in [self.eliminated, self.current_pose, self.map, self.map_info]) \
            and self.nav_status != NavStatus.NAVIGATING

    def on_heartbeat(self, msg: HeartbeatMessage) -> None:
        self.other_robot_poses[msg.sender_id] = (msg.pose.position.x, msg.pose.position.y)

    def on_coordination(self, msg: BaseCoordinationMessage) -> None:
        if isinstance(msg, IntentionMessage):
            self.intentions[msg.sender_id] = (msg.target_x, msg.target_y)

    def on_lidar_scan(self, scan: LaserScan) -> None:
        # track new eliminations for rendezvous decision
        if self.eliminated is not None:
            current_count = int(np.sum(self.eliminated))
            new_cells = current_count - self._prev_eliminated_count
            if new_cells > 0:
                self.cells_since_fusion += new_cells
                self._prev_eliminated_count = current_count

    def on_fusion_completed(self) -> None:
        # replan if fusion added new coverage
        if self.eliminated is None:
            return
        new_count = int(np.sum(self.eliminated))
        gained = new_count - self._prev_eliminated_count
        self.cells_since_fusion = 0
        self._prev_eliminated_count = new_count
        if gained > 0:
            self.cancel_navigation()
            self.get_logger().info(f"Fusion gained {gained} cells, replanning")

    def on_target_detected(self, target_locations: list[tuple[float, float]]) -> None:
        for x, y in target_locations:
            self.get_logger().info(f"Target at ({x:.2f}, {y:.2f})")

    def on_navigation_succeeded(self) -> None:
        self.get_logger().info("Reached target")

    def on_navigation_failed(self, reason: str) -> None:
        self.get_logger().warn(f"Nav failed {reason}")

    # check if all known robots are within threshold
    def _robots_are_clustered(self, threshold: float = 5.0) -> bool:
        if not self.other_robot_poses or self.current_pose is None:
            return False
        mx = self.current_pose.position.x
        my = self.current_pose.position.y
        return all(
            math.hypot(mx - rx, my - ry) <= threshold
            for rx, ry in self.other_robot_poses.values()
        )

    # pick a random free cell far from us
    def _pick_scatter_target(self, min_dist: float = 10.0, n_samples: int = 200) -> tuple[float, float] | None:
        assert self.map is not None and self.map_info is not None and self.current_pose is not None

        free_rows, free_cols = np.where((self.map != 100) & (self.map != -1))
        if len(free_rows) == 0:
            return None

        indices = np.random.choice(len(free_rows), size=min(n_samples, len(free_rows)), replace=False)
        rx = self.current_pose.position.x
        ry = self.current_pose.position.y
        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y

        best: tuple[float, float] | None = None
        best_dist = 0.0
        far_enough: list[tuple[float, float]] = []

        for idx in indices:
            wx = float(free_cols[idx]) * res + ox
            wy = float(free_rows[idx]) * res + oy
            d = math.hypot(wx - rx, wy - ry)
            if d >= min_dist:
                far_enough.append((wx, wy))
            if d > best_dist:
                best_dist = d
                best = (wx, wy)

        return random.choice(far_enough) if far_enough else best

    def _pose_from_xy(self, x: float, y: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self._map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def _cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        assert self.map_info is not None
        row, col = cell
        x = col * self.map_info.resolution + self.map_info.origin.position.x
        y = row * self.map_info.resolution + self.map_info.origin.position.y
        return x, y

    def _world_to_cell(self, world: tuple[float, float]) -> tuple[int, int]:
        assert self.map_info is not None
        x, y = world
        col = int((x - self.map_info.origin.position.x) / self.map_info.resolution)
        row = int((y - self.map_info.origin.position.y) / self.map_info.resolution)
        return row, col


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    agent = VoronoiExplorationAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(agent)
    executor.spin()
    agent.destroy_node()
    rclpy.shutdown()
