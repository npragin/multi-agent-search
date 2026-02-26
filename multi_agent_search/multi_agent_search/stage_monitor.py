"""Stage Monitor Node - waits for Stage simulator to be ready."""

from rosgraph_msgs.msg import Clock

import rclpy
from rclpy.node import Node


class StageMonitor(Node):
    """Subscribes to /clock and exits when the first message is received."""

    def __init__(self) -> None:
        """Initialize the stage monitor node."""
        super().__init__("stage_monitor")
        self.create_subscription(Clock, "/clock", self._on_clock, 1)
        self.get_logger().info("Waiting for Stage /clock...")

    def _on_clock(self, msg: Clock) -> None:
        self.get_logger().info("Stage is up (received /clock)")
        raise SystemExit(0)


def main(args: list[str] | None = None) -> None:
    """Entry point for the stage monitor node."""
    rclpy.init(args=args)
    node = StageMonitor()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
