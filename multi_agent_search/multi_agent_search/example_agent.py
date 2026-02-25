"""Brain Dead Agent for testing the multi-agent search system."""

import rclpy
from nav2_msgs.action import NavigateToPose
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import LaserScan

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
    """

    def __init__(self) -> None:
        """Initialize the Example Agent."""
        super().__init__("ExampleAgent")

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


def main(args: list[str] | None = None) -> None:
    """Entry point for the Example Agent."""
    rclpy.init(args=args)
    agent = ExampleAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(agent)
    executor.spin()
    agent.destroy_node()
    rclpy.shutdown()
