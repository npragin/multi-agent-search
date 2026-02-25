"""Lifecycle Monitor Node - waits for specified lifecycle nodes to reach active state."""

import rclpy
from lifecycle_msgs.msg import State, TransitionEvent
from lifecycle_msgs.srv import GetState
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.node import Node


class LifecycleMonitor(Node):
    """Subscribes to transition events and exits when all monitored nodes are active."""

    def __init__(self) -> None:
        """Initialize the lifecycle monitor node."""
        super().__init__("lifecycle_monitor")

        self.declare_parameter("node_names", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("timeout", 30.0)

        node_names: list[str] = self.get_parameter("node_names").get_parameter_value().string_array_value
        timeout: float = self.get_parameter("timeout").get_parameter_value().double_value

        if not node_names:
            self.get_logger().error("No node_names provided")
            raise SystemExit(1)

        self._pending: set[str] = set(node_names)
        self._cb_group = ReentrantCallbackGroup()

        for name in node_names:
            topic = f"/{name}/transition_event"
            self.create_subscription(
                TransitionEvent,
                topic,
                lambda msg, n=name: self._on_transition(msg, n),
                10,
            )

        self.get_logger().info(f"Waiting for the following nodes to become active: {node_names}")

        self._poll_current_states()

        self._timeout_timer = self.create_timer(timeout, self._on_timeout)

    def _mark_active(self, node_name: str) -> None:
        """Mark a node as active and exit if all nodes are active."""
        if node_name in self._pending:
            self._pending.discard(node_name)
            if not self._pending:
                self.get_logger().info("All monitored nodes are active")
                raise SystemExit(0)
            else:
                self.get_logger().info(f"{node_name} is active. Remaining nodes: {sorted(self._pending)})")

    def _on_transition(self, msg: TransitionEvent, node_name: str) -> None:
        """Handle a transition event from a monitored node."""
        if msg.goal_state.id == State.PRIMARY_STATE_ACTIVE:
            self._mark_active(node_name)

    def _poll_current_states(self) -> None:
        """Query the current state of all still-pending nodes via GetState service."""
        for name in list(self._pending):
            service_name = f"/{name}/get_state"
            client: Client[GetState.Request, GetState.Response] = self.create_client(
                GetState, service_name, callback_group=self._cb_group
            )
            if not client.wait_for_service(timeout_sec=0.1):
                self.destroy_client(client)
                continue
            future = client.call_async(GetState.Request())
            rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)
            if (
                future.done()
                and future.result() is not None
                and future.result().current_state.id == State.PRIMARY_STATE_ACTIVE  # type: ignore[union-attr]
            ):
                self._mark_active(name)
            self.destroy_client(client)

    def _on_timeout(self) -> None:
        """Handle timeout expiry."""
        self.get_logger().error(f"Timed out waiting for nodes: {sorted(self._pending)}")
        raise SystemExit(1)


def main(args: list[str] | None = None) -> None:
    """Entry point for the lifecycle monitor node."""
    rclpy.init(args=args)
    node = LifecycleMonitor()
    try:
        rclpy.spin(node)
    except SystemExit:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
