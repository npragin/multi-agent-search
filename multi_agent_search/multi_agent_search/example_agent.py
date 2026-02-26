"""Brain Dead Agent for testing the multi-agent search system."""

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Header

from multi_agent_search.agent_base import AgentBase
from multi_agent_search.types import BaseCoordinationMessage, HeartbeatMessage


class ExampleAgent(AgentBase):
    """
    Example Agent for testing the multi-agent search system.

    Some other functions you might use:
    - publish_heartbeat
    - publish_coordination_message (define your own coordination message that inherits from BaseCoordinationMessage)
    - navigate_to
    - cancel_navigation

    Some useful attributes for you to access:
    - self.agent_id
    - self.current_pose
    - self.belief
    - self.eliminated
    - self.map
    - self.use_known_map
    - self.known_initial_poses
    - self.map_info
    - self.nav_status
    """

    def __init__(self) -> None:
        """Initialize the Example Agent."""
        super().__init__("ExampleAgent")

        self.time = self.get_clock().now()
        self.mode = 0

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

    def on_target_detected(self, target_locations: list[tuple[float, float]]) -> None:
        """Handle a target detection."""
        pass

    def on_map_updated(self) -> None:
        """Handle a map update."""
        pass

    def on_fusion_completed(self) -> None:
        """Handle a fusion completion."""
        if self.agent_id == "robot_0":
            if self.mode == 0 and self.get_clock().now() - self.time > Duration(seconds=10):
                header = Header()
                header.stamp = self.get_clock().now().to_msg()
                header.frame_id = "map"
                pose = Pose()
                pose.position.x = 0.0
                pose.position.y = 0.0
                pose.position.z = 0.0
                pose.orientation.x = 0.0
                pose.orientation.y = 0.0
                pose.orientation.z = 0.0
                pose.orientation.w = 1.0
                pose_stamped = PoseStamped()
                pose_stamped.header = header
                pose_stamped.pose = pose
                self.navigate_to(pose_stamped)
                self.mode += 1
            if self.mode == 1:
                pass

    def on_navigation_feedback(self, feedback: NavigateToPose.Feedback) -> None:
        """Handle a navigation feedback."""
        self.get_logger().info(f"Navigation feedback: {feedback}")

    def on_navigation_succeeded(self) -> None:
        """Handle a navigation success."""
        pass

    def on_navigation_failed(self, reason: str) -> None:
        """Handle a navigation failure."""
        pass


def main(args: list[str] | None = None) -> None:
    """Entry point for the Example Agent."""
    rclpy.init(args=args)
    agent = ExampleAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(agent)
    executor.spin()
    agent.destroy_node()
    rclpy.shutdown()
