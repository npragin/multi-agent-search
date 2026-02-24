"""Brain Dead Agent for testing the multi-agent search system."""

import rclpy
from sensor_msgs.msg import LaserScan

from multi_agent_search.agent_base import AgentBase
from multi_agent_search.types import BaseCoordinationMessage, HeartbeatMessage


class BrainDeadAgent(AgentBase):
    """Brain Dead Agent for testing the multi-agent search system."""

    def __init__(self):
        """Initialize the Brain Dead Agent."""
        super().__init__("BrainDeadAgent")

        self.get_logger().info("Brain Dead Agent initialized")

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

    def on_navigation_succeeded(self) -> None:
        """Handle a navigation success."""
        pass


def main(args=None):
    """Main function for the Brain Dead Agent."""
    rclpy.init(args=args)
    agent = BrainDeadAgent()
    rclpy.spin(agent)
    agent.destroy_node()
    rclpy.shutdown()
