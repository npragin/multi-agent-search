"""AgentBase subclass that doesn't coordinate with other agents and drives to the nearest frontier."""

from heapq import heappop, heappush

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import LaserScan

from multi_agent_search.agent_base import AgentBase
from multi_agent_search.types import BaseCoordinationMessage, HeartbeatMessage, NavStatus


class UncoordinatedAgent(AgentBase):
    """AgentBase subclass that doesn't coordinate with other agents and drives to the nearest frontier."""

    def __init__(self) -> None:
        """Initialize the Uncoordinated Agent."""
        super().__init__("UncoordinatedAgent")

        self.declare_parameter("planning_cooldown", 5.0)
        self.declare_parameter("minimum_frontier_distance", 1.0)

        self.planning_cooldown = self.get_parameter("planning_cooldown").value
        self.minimum_frontier_distance = self.get_parameter("minimum_frontier_distance").value

        self.last_planning_time = self.get_clock().now()

        self.planning_timer = self.create_timer(self.planning_cooldown, self.planning_timer_callback)

        self.get_logger().info("Initialized")

    def on_heartbeat(self, msg: HeartbeatMessage) -> None:
        """Handle a heartbeat."""
        pass

    def on_coordination(self, msg: BaseCoordinationMessage) -> None:
        """Handle a coordination message."""
        pass

    def on_lidar_scan(self, scan: LaserScan) -> None:
        """Handle a lidar scan."""
        pass

    def planning_timer_callback(self) -> None:
        """Find the nearest frontier and navigate to it."""
        if self.ready_to_plan():
            nearest_frontier_point = self._find_nearest_frontier()
            if nearest_frontier_point is not None:
                goal_pose_stamped = self._pose_from_xy(nearest_frontier_point[0], nearest_frontier_point[1])
                self.get_logger().info(
                    f"Navigating to {nearest_frontier_point[0]:.2f}, {nearest_frontier_point[1]:.2f}"
                )
                self.navigate_to(goal_pose_stamped)
                self.last_planning_time = self.get_clock().now()
            else:
                self.get_logger().info("No frontier found. Did I explore the entire map?")

    def on_target_detected(self, target_locations: list[tuple[float, float]]) -> None:
        """Handle a target detection."""
        pass

    def on_map_updated(self) -> None:
        """Handle a map update."""
        pass

    def on_fusion_completed(self) -> None:
        """Handle a fusion completion."""
        pass

    def on_navigation_feedback(self, feedback: NavigateToPose.Feedback) -> None:
        """Handle a navigation feedback."""
        pass

    def on_navigation_succeeded(self) -> None:
        """Handle a navigation success."""
        pass

    def on_navigation_failed(self, reason: str) -> None:
        """Handle a navigation failure."""
        pass

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

    def ready_to_plan(self) -> bool:
        """Check if the agent is ready to plan."""
        attributes_to_check = [self.eliminated, self.current_pose, self.map, self.map_info]
        relevant_attributes_are_not_none = all(attr is not None for attr in attributes_to_check)
        return relevant_attributes_are_not_none and self.nav_status != NavStatus.NAVIGATING

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
    """Entry point for the Uncoordinated Agent."""
    rclpy.init(args=args)
    agent = UncoordinatedAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(agent)
    executor.spin()
    agent.destroy_node()
    rclpy.shutdown()
