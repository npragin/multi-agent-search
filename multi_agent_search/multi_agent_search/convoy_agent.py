"""AgentBase subclass implementing convoy behavior with leader election and breadcrumb following."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from heapq import heappop, heappush

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn
from sensor_msgs.msg import LaserScan

from multi_agent_search.agent_base import AgentBase
from multi_agent_search.types import BaseCoordinationMessage, HeartbeatMessage, NavStatus


@dataclass
class LeaderNegotiationMessage(BaseCoordinationMessage):
    """Message broadcast during leader negotiation to share agent IDs."""

    pass


@dataclass
class HandshakeMessage(BaseCoordinationMessage):
    """Coordination message used to confirm agents are within coordination range before re-entering negotiation."""

    pass


class ConvoyPhase(IntEnum):
    """Phase of the convoy agent."""

    LEADER_NEGOTIATION = 0
    LEADING = 1
    FOLLOWING = 2


class ConvoyAgent(AgentBase):
    """AgentBase subclass implementing convoy behavior with leader election and breadcrumb following."""

    def __init__(self) -> None:
        """Initialize the Convoy Agent."""
        super().__init__("ConvoyAgent")

        self.declare_parameter("negotiation_timeout", 5.0)
        self.declare_parameter("heartbeat_timeout", 10.0)
        self.declare_parameter("planning_cooldown", 1.0)
        self.declare_parameter("follower_nav_cooldown", 2.0)
        self.declare_parameter("minimum_frontier_distance", 1.0)
        self.declare_parameter("handshake_timeout", 5.0)

        self.negotiation_timeout: float = 0.0
        self.heartbeat_timeout: float = 0.0
        self.planning_cooldown: float = 0.0
        self.follower_nav_cooldown: float = 0.0
        self.minimum_frontier_distance: float = 0.0
        self.handshake_timeout: float = 0.0

        self.phase = ConvoyPhase.LEADER_NEGOTIATION
        self.known_agent_ids: set[str] = set()
        self.leader_id: str | None = None
        self.last_leader_heartbeat_time = self.get_clock().now()
        self._last_breadcrumb_pos: tuple[float, float] | None = None
        self._negotiation_start_time = self.get_clock().now()
        self._handshake_pending = False
        self._handshake_start_time = self.get_clock().now()

        self.negotiation_timer: rclpy.timer.Timer | None = None
        self.planning_timer: rclpy.timer.Timer | None = None
        self.follower_timeout_timer: rclpy.timer.Timer | None = None
        self.handshake_timer: rclpy.timer.Timer | None = None

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate: read parameters and start leader negotiation."""
        result = super().on_activate(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result

        self.negotiation_timeout = self.get_parameter("negotiation_timeout").value
        self.heartbeat_timeout = self.get_parameter("heartbeat_timeout").value
        self.planning_cooldown = self.get_parameter("planning_cooldown").value
        self.follower_nav_cooldown = self.get_parameter("follower_nav_cooldown").value
        self.minimum_frontier_distance = self.get_parameter("minimum_frontier_distance").value
        self.handshake_timeout = self.get_parameter("handshake_timeout").value

        # Start negotiation phase
        self.known_agent_ids = {self.agent_id}
        self._negotiation_start_time = self.get_clock().now()
        self.negotiation_timer = self.create_timer(0.5, self._negotiation_tick)

        self.get_logger().info("Activated. Entering leader negotiation phase")
        return TransitionCallbackReturn.SUCCESS

    # ------------------------------------------------------------------
    # Leader negotiation
    # ------------------------------------------------------------------

    def _negotiation_tick(self) -> None:
        """Periodically broadcast our agent_id and check if the negotiation timeout has elapsed."""
        if self.phase != ConvoyPhase.LEADER_NEGOTIATION:
            return

        self.publish_coordination_message(LeaderNegotiationMessage(sender_id=self.agent_id))

        elapsed = (self.get_clock().now() - self._negotiation_start_time).nanoseconds / 1e9
        if elapsed >= self.negotiation_timeout:
            self._elect_leader()

    def _elect_leader(self) -> None:
        """Choose the leader as the first agent_id in sorted order."""
        self.leader_id = sorted(self.known_agent_ids)[0]
        self.get_logger().info(f"Leader elected: {self.leader_id} (from candidates: {sorted(self.known_agent_ids)})")

        if self.negotiation_timer is not None:
            self.negotiation_timer.cancel()
            self.negotiation_timer = None

        if self.leader_id == self.agent_id:
            self._become_leader()
        else:
            self._become_follower()

    def _become_leader(self) -> None:
        """Transition to the LEADING phase."""
        self.phase = ConvoyPhase.LEADING
        self.get_logger().info("I am the leader. Starting frontier exploration")

        self.planning_timer = self.create_timer(self.planning_cooldown, self._planning_tick)

    def _become_follower(self) -> None:
        """Transition to the FOLLOWING phase."""
        self.phase = ConvoyPhase.FOLLOWING
        self.last_leader_heartbeat_time = self.get_clock().now()
        self._last_breadcrumb_pos = None
        self.get_logger().info(f"I am a follower. Following leader {self.leader_id}")

        self.follower_timeout_timer = self.create_timer(1.0, self._check_breadcrumb_timeout)
        self.planning_timer = self.create_timer(self.follower_nav_cooldown, self._follower_nav_tick)

    # ------------------------------------------------------------------
    # Leader behavior
    # ------------------------------------------------------------------

    def _planning_tick(self) -> None:
        """Find the nearest frontier and navigate to it (leader only)."""
        if self.phase != ConvoyPhase.LEADING or self.current_pose is None or self.eliminated is None:
            return

        self.publish_heartbeat()

        if self.nav_status != NavStatus.NAVIGATING:
            nearest = self._find_nearest_frontier()
            if nearest is not None:
                goal = self._pose_from_xy(nearest[0], nearest[1])
                self.get_logger().info(f"Leader navigating to frontier ({nearest[0]:.2f}, {nearest[1]:.2f})")
                self.navigate_to(goal)
            else:
                self.get_logger().info("No frontier found. Did I explore the entire map?")

    # ------------------------------------------------------------------
    # Follower behavior
    # ------------------------------------------------------------------

    def _follower_nav_tick(self) -> None:
        """Navigate to the latest breadcrumb position if it has changed."""
        if self.phase != ConvoyPhase.FOLLOWING or self._last_breadcrumb_pos is None:
            return

        x, y = self._last_breadcrumb_pos
        goal = self._pose_from_xy(x, y)
        self.get_logger().debug(f"Following leader to ({x:.2f}, {y:.2f})")
        self.navigate_to(goal)

    def _check_breadcrumb_timeout(self) -> None:
        """If no breadcrumb received within timeout, re-enter leader negotiation."""
        if self.phase != ConvoyPhase.FOLLOWING:
            return

        elapsed = (self.get_clock().now() - self.last_leader_heartbeat_time).nanoseconds / 1e9
        if elapsed >= self.heartbeat_timeout:
            self.get_logger().warn(f"No heartbeat from leader for {elapsed:.1f}s. Re-entering leader negotiation")
            self._enter_negotiation()

    def _start_handshake(self) -> None:
        """Send a handshake coordination message and start the handshake timeout."""
        self._handshake_pending = True
        self._handshake_start_time = self.get_clock().now()
        self.publish_coordination_message(HandshakeMessage(sender_id=self.agent_id))
        self.handshake_timer = self.create_timer(1.0, self._check_handshake_timeout)

    def _check_handshake_timeout(self) -> None:
        """Cancel the handshake if no response is received within the timeout."""
        if not self._handshake_pending:
            return

        elapsed = (self.get_clock().now() - self._handshake_start_time).nanoseconds / 1e9
        if elapsed >= self.handshake_timeout:
            self.get_logger().info("Handshake timed out. Agent not in coordination range")
            self._cancel_handshake()

    def _cancel_handshake(self) -> None:
        """Clear handshake state and cancel the handshake timer."""
        self._handshake_pending = False
        if self.handshake_timer is not None:
            self.handshake_timer.cancel()
            self.handshake_timer = None

    def _enter_negotiation(self) -> None:
        """Reset state and re-enter the leader negotiation phase."""
        # Cancel role-specific timers
        if self.planning_timer is not None:
            self.planning_timer.cancel()
            self.planning_timer = None
        if self.follower_timeout_timer is not None:
            self.follower_timeout_timer.cancel()
            self.follower_timeout_timer = None
        self._cancel_handshake()

        self.cancel_navigation()

        # Reset negotiation state
        self.phase = ConvoyPhase.LEADER_NEGOTIATION
        self.known_agent_ids = {self.agent_id}
        self.leader_id = None
        self._negotiation_start_time = self.get_clock().now()

        self.negotiation_timer = self.create_timer(0.5, self._negotiation_tick)

        self.get_logger().info("Re-entered leader negotiation phase")

    # ------------------------------------------------------------------
    # AgentBase hooks
    # ------------------------------------------------------------------

    def on_heartbeat(self, msg: HeartbeatMessage) -> None:
        """Handle a heartbeat. Followers store the leader's position as a breadcrumb."""
        if self.phase != ConvoyPhase.LEADER_NEGOTIATION and msg.sender_id != self.leader_id:
            if not self._handshake_pending:
                self.get_logger().info(f"Received heartbeat from unknown agent {msg.sender_id}. Initiating handshake")
                self._start_handshake()
            return

        if self.phase == ConvoyPhase.FOLLOWING and msg.sender_id == self.leader_id:
            self.last_leader_heartbeat_time = self.get_clock().now()
            self._last_breadcrumb_pos = (msg.pose.position.x, msg.pose.position.y)

    def on_coordination(self, msg: BaseCoordinationMessage) -> None:
        """Handle a coordination message."""
        if isinstance(msg, LeaderNegotiationMessage) and self.phase == ConvoyPhase.LEADER_NEGOTIATION:
            self.known_agent_ids.add(msg.sender_id)
        elif isinstance(msg, HandshakeMessage) and self.phase != ConvoyPhase.LEADER_NEGOTIATION:
            self.get_logger().info(f"Received handshake from {msg.sender_id}. Re-entering negotiation")
            self._cancel_handshake()
            self._enter_negotiation()

    def on_lidar_scan(self, scan: LaserScan) -> None:
        """Handle a lidar scan."""
        pass

    def on_target_detected(self, target_locations: list[tuple[float, float]]) -> None:
        """Handle a target detection."""
        pass

    # ------------------------------------------------------------------
    # Frontier finding
    # ------------------------------------------------------------------

    def _pose_from_xy(self, x: float, y: float) -> PoseStamped:
        """Create a pose from (x, y) coordinates."""
        pose = PoseStamped()
        pose.header.frame_id = "map" if self.use_known_map else self.agent_id + "/map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def _find_nearest_frontier(self) -> tuple[float, float] | None:
        """Find the nearest frontier and return the (x, y) coordinates."""
        if self.eliminated is None:
            raise ValueError("Eliminated is not set")
        if self.current_pose is None:
            raise ValueError("Current pose is not set")
        if self.map is None:
            raise ValueError("Map is not set")
        if self.map_info is None:
            raise ValueError("Map info is not set")

        self.get_logger().info("Finding nearest frontier")

        robot_world = (self.current_pose.position.x, self.current_pose.position.y)
        robot_cell = self._world_to_cell(robot_world)
        robot_cell_tuple: tuple[int, tuple[int, int]] = (0, robot_cell)

        pq: list[tuple[int, tuple[int, int]]] = [robot_cell_tuple]
        visited: set[tuple[int, int]] = set()
        while pq:
            distance, cell = heappop(pq)
            if cell in visited or self.map[cell] == 100:
                continue
            visited.add(cell)
            if not self.eliminated[cell] and distance > self.minimum_frontier_distance:
                return self._cell_to_world(cell)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor_cell = (cell[0] + dx, cell[1] + dy)
                if 0 <= neighbor_cell[0] < self.map.shape[0] and 0 <= neighbor_cell[1] < self.map.shape[1]:
                    heappush(pq, (distance + self.map_info.resolution, neighbor_cell))

        return None

    def _cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Convert a (row, col) cell to a world (x, y) coordinate."""
        if self.map_info is None:
            raise ValueError("Map info is not set")

        row, col = cell
        world_x = col * self.map_info.resolution + self.map_info.origin.position.x
        world_y = row * self.map_info.resolution + self.map_info.origin.position.y
        return world_x, world_y

    def _world_to_cell(self, world: tuple[float, float]) -> tuple[int, int]:
        """Convert a world (x, y) coordinate to a (row, col) cell."""
        if self.map_info is None:
            raise ValueError("Map info is not set")

        world_x, world_y = world
        col = int((world_x - self.map_info.origin.position.x) / self.map_info.resolution)
        row = int((world_y - self.map_info.origin.position.y) / self.map_info.resolution)
        return row, col


def main(args: list[str] | None = None) -> None:
    """Entry point for the Convoy Agent."""
    rclpy.init(args=args)
    agent = ConvoyAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(agent)
    executor.spin()
    agent.destroy_node()
    rclpy.shutdown()
