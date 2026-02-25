"""
Agent Base Class for the multi-agent search system.

Provides common communication infrastructure, Nav2 integration, belief grid
management, and target detection for concrete agent implementations.
"""

from __future__ import annotations

import ast
import math
import pickle
import time
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from action_msgs.msg import GoalStatus
from skimage.draw import line as skimage_line

from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import SetInitialPose
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.task import Future
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Empty, Trigger

from multi_agent_search.types import (
    BaseCoordinationMessage,
    HeartbeatMessage,
    NavStatus,
)
from multi_agent_search_interfaces.msg import AgentMessage
from multi_agent_search_interfaces.srv import GetMap, SetMap

LOG_ODDS_MIN, LOG_ODDS_MAX = -7.0, 7.0  # TODO: Magic number
SCALE = 127.0 / LOG_ODDS_MAX  # TODO: Magic number
ELIMINATED_VALUE = -128  # TODO: Magic number


class AgentBase(LifecycleNode, ABC):
    """
    Abstract lifecycle base class for search agents.

    Provides the communication interface with the CommsManager, Nav2 action
    client for navigation, belief grid management from lidar data, and target
    detection. Concrete subclasses implement the search and coordination logic.
    """

    def __init__(self, agent_id: str) -> None:
        """Initialize base agent with ID and declare parameters."""
        super().__init__(agent_id)
        self._set_up_parameters()
        self._set_up_state_defaults()

    # -------------------------------------------------------------------------
    # Lifecycle Callbacks
    # -------------------------------------------------------------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure: read params, create all interfaces (pubs/subs/services/clients)."""
        self._set_up_state()
        self._set_up_publishers()
        self._set_up_action_clients()
        self._set_up_service_clients()
        self._set_up_subscribers()
        self._set_up_service_servers()
        self.get_logger().info("Configured")
        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate: initialize localization, start timers, enable lifecycle publishers."""
        if self._use_known_map:
            if self._known_initial_poses:
                result = self._wait_and_set_initial_pose()
            else:
                result = self._wait_and_reinitialize_global_localization()
            if result != TransitionCallbackReturn.SUCCESS:
                return result
        self.get_logger().info("Activated")
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivate: cancel timers and navigation, disable lifecycle publishers."""
        for timer in self._managed_timers:
            timer.cancel()
        self._managed_timers.clear()
        self.cancel_navigation()
        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Cleanup: destroy all interfaces, reset state."""
        self._destroy_interfaces()
        self._set_up_state_defaults()
        self.get_logger().info("Cleaned up")
        return super().on_cleanup(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Shutdown: same cleanup as on_cleanup."""
        self._destroy_interfaces()
        self.get_logger().info("Shut down")
        return super().on_shutdown(state)

    def _destroy_interfaces(self) -> None:
        """Destroy all ROS2 interfaces created during configure."""
        for timer in self._managed_timers:
            timer.cancel()
        for sub in self._managed_subscriptions:
            self.destroy_subscription(sub)
        for pub in self._managed_publishers:
            self.destroy_publisher(pub)
        for srv in self._managed_service_servers:
            self.destroy_service(srv)
        for client in self._managed_service_clients:
            self.destroy_client(client)
        for action_client in self._managed_action_clients:
            action_client.destroy()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _set_up_parameters(self) -> None:
        """Set up parameters for the agent."""
        self.declare_parameter("agent_id", "robot_0")

        self.declare_parameter("use_known_map", False)
        self.declare_parameter("known_initial_poses", False)

        self.declare_parameter("target_positions", "[]")
        self.declare_parameter("target_radius", 1.0)

        self.declare_parameter("amcl_initialization_service_call_timeout", 5.0)
        self.declare_parameter("amcl_initialization_service_call_max_attempts", 3)

    def _set_up_state_defaults(self) -> None:
        """Set all instance attributes to safe defaults before on_configure."""
        self._agent_id: str = ""
        self._use_known_map: bool = False
        self._known_initial_poses: bool = False
        self._map: npt.NDArray[np.int8] | None = None
        self._belief: npt.NDArray[np.float32] | None = None
        self._eliminated: npt.NDArray[np.bool_] | None = None
        self._map_info: MapMetaData | None = None
        self._target_positions: list[tuple[float, float]] = []
        self._target_radius: float = 1.0
        self._found_targets: set[int] = set()
        self._amcl_initialization_service_call_timeout: float = 5.0
        self._amcl_initialization_service_call_max_attempts: int = 3
        self._initial_pose_msg: PoseWithCovarianceStamped | None = None
        self._nav_status: NavStatus = NavStatus.IDLE
        self._current_nav_goal: ClientGoalHandle | None = None
        self._pending_goal: PoseStamped | None = None
        self._current_pose: Pose | None = None

        # Lifecycle-managed interface lists for clean teardown
        self._managed_timers: list[Timer] = []
        self._managed_subscriptions: list[Subscription] = []
        self._managed_publishers: list[Publisher] = []
        self._managed_service_servers: list[Service] = []
        self._managed_service_clients: list[Client] = []
        self._managed_action_clients: list[ActionClient] = []

    def _set_up_state(self) -> None:
        """Set up state for the agent."""
        # Identity
        self._agent_id: str = self.get_parameter("agent_id").value
        self._use_known_map: bool = self.get_parameter("use_known_map").value
        self._known_initial_poses: bool = self.get_parameter("known_initial_poses").value

        if self._known_initial_poses and not self._use_known_map:
            raise ValueError("known_initial_poses requires use_known_map to be true (AMCL must be running)")

        # Map and belief state
        self._map: npt.NDArray[np.int8] | None = None
        self._belief: npt.NDArray[np.float32] | None = None
        self._eliminated: npt.NDArray[np.bool_] | None = None
        self._map_info: MapMetaData | None = None

        # Target detection state
        raw = self.get_parameter("target_positions").value
        parsed = ast.literal_eval(raw)
        self._target_positions: list[tuple[float, float]] = [(float(p[0]), float(p[1])) for p in parsed]
        self._target_radius: float = self.get_parameter("target_radius").value
        self._found_targets: set[int] = set()

        # Service call retry parameters
        self._amcl_initialization_service_call_timeout: float = self.get_parameter(
            "amcl_initialization_service_call_timeout"
        ).value
        self._amcl_initialization_service_call_max_attempts: int = self.get_parameter(
            "amcl_initialization_service_call_max_attempts"
        ).value

        # Localization state
        self._initial_pose_msg: PoseWithCovarianceStamped | None = None

        # Navigation state
        self._nav_status: NavStatus = NavStatus.IDLE
        self._current_nav_goal: ClientGoalHandle | None = None
        self._pending_goal: PoseStamped | None = None

        # Pose state
        self._current_pose: Pose | None = None

    def _set_up_subscribers(self) -> None:
        """Set up subscribers for the agent."""
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.sub_incoming = self.create_subscription(
            AgentMessage, f"/comms/input/{self._agent_id}", self._on_incoming_message, 10
        )
        self.sub_lidar = self.create_subscription(
            LaserScan, f"/{self._agent_id}/base_scan", self._on_lidar_callback, 10
        )
        self.sub_map = self.create_subscription(
            OccupancyGrid, f"/{self._agent_id}/map", self._on_map_updated, latched_qos
        )
        self.sub_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            f"/{self._agent_id}/pose",
            self._on_pose_updated,
            latched_qos if self._use_known_map else 10,  # AMCL publishes a latched topic
        )
        self._managed_subscriptions.extend([self.sub_incoming, self.sub_lidar, self.sub_map, self.sub_pose])

        if self._known_initial_poses:
            self.sub_initial_pose = self.create_subscription(
                PoseWithCovarianceStamped,
                f"/{self._agent_id}/initialpose",
                self._on_initial_pose,
                latched_qos,
                callback_group=self._localization_cbg,
            )
            self._managed_subscriptions.append(self.sub_initial_pose)

    def _set_up_publishers(self) -> None:
        """Set up lifecycle publishers for the agent (only transmit when active)."""
        self.pub_outgoing = self.create_lifecycle_publisher(AgentMessage, f"/comms/output/{self._agent_id}", 10)
        self.pub_found_targets = self.create_lifecycle_publisher(
            Int32MultiArray, f"/{self._agent_id}/found_targets", 10
        )
        self._managed_publishers.extend([self.pub_outgoing, self.pub_found_targets])

    def _set_up_service_servers(self) -> None:
        """Create map/belief get/set services and fusion_complete service."""
        self.srv_get_map = self.create_service(GetMap, f"/{self._agent_id}/get_map", self._handle_get_map)
        self.srv_set_map = self.create_service(SetMap, f"/{self._agent_id}/set_map", self._handle_set_map)
        self.srv_get_belief = self.create_service(GetMap, f"/{self._agent_id}/get_belief", self._handle_get_belief)
        self.srv_set_belief = self.create_service(SetMap, f"/{self._agent_id}/set_belief", self._handle_set_belief)
        self.srv_fusion_complete = self.create_service(
            Trigger, f"/{self._agent_id}/fusion_complete", self._handle_fusion_complete
        )
        self._managed_service_servers.extend(
            [
                self.srv_get_map,
                self.srv_set_map,
                self.srv_get_belief,
                self.srv_set_belief,
                self.srv_fusion_complete,
            ]
        )

    def _set_up_action_clients(self) -> None:
        """Create NavigateToPose action client at /{agent_id}/navigate_to_pose."""
        self._nav_client: ActionClient[NavigateToPose.Goal, NavigateToPose.Result, NavigateToPose.Feedback] = (
            ActionClient(self, NavigateToPose, f"/{self._agent_id}/navigate_to_pose")
        )
        self._managed_action_clients.append(self._nav_client)

    def _set_up_service_clients(self) -> None:
        """
        Create service clients.

        Localization clients use a ReentrantCallbackGroup so on_activate can poll-wait for responses without deadlocking.
        """
        self._localization_cbg = ReentrantCallbackGroup()
        if self._use_known_map and not self._known_initial_poses:
            self._reinit_global_loc_client: Client[Empty.Request, Empty.Response] = self.create_client(
                Empty, f"/{self._agent_id}/reinitialize_global_localization", callback_group=self._localization_cbg
            )
            self._managed_service_clients.append(self._reinit_global_loc_client)
        if self._known_initial_poses:
            self._set_initial_pose_client: Client[SetInitialPose.Request, SetInitialPose.Response] = self.create_client(
                SetInitialPose, f"/{self._agent_id}/set_initial_pose", callback_group=self._localization_cbg
            )
            self._managed_service_clients.append(self._set_initial_pose_client)

    # -------------------------------------------------------------------------
    # Publishing Interface
    # -------------------------------------------------------------------------

    def _publish_message(
        self,
        msg_type: int,
        payload: bytes,
        recipient: str = "",
        overwrite_targeted: bool = True,
    ) -> None:
        """
        Publish message to the comms manager.

        Args:
            msg_type: Message type enum value (HEARTBEAT, COORDINATION)
            payload: Serialized message data
            recipient: Specific recipient ID, or "" for broadcast
            overwrite_targeted: For broadcasts, whether to clear recipient-specific
                               messages in comms manager buffer

        Note: The comms manager will deliver this message to close-range agents at
        close-range rate. For long-range agents, only HEARTBEAT
        messages will be delivered (at long-range rate). COORDINATION messages are
        never delivered to long-range agents.

        Important: When you publish a COORDINATION message, long-range agents will
        not receive your previous heartbeat - they simply receive nothing until you
        publish another HEARTBEAT message.

        """
        if recipient == self._agent_id:
            self.get_logger().info("Skipping publishing message to self")
            return
        self.pub_outgoing.publish(
            AgentMessage(
                msg_type=msg_type,
                sender_id=self._agent_id,
                recipient_id=recipient,
                overwrite_targeted=overwrite_targeted,
                timestamp=self.get_clock().now().nanoseconds,
                payload=payload,
            )
        )

    def publish_heartbeat(self, recipient: str = "", overwrite_targeted: bool = True) -> None:
        """
        Publish a heartbeat carrying this agent's current pose.

        Uses the most recent pose cached from the /{agent_id}/amcl_pose
        subscription.  Logs a warning and returns without publishing if no
        pose has been received yet.
        """
        heartbeat_message = HeartbeatMessage(
            sender_id=self._agent_id, timestamp=self.get_clock().now().nanoseconds / 1e9, pose=self._current_pose
        )
        self._publish_message(AgentMessage.HEARTBEAT, pickle.dumps(heartbeat_message), recipient, overwrite_targeted)

    def publish_coordination_message(
        self,
        msg: BaseCoordinationMessage,
        recipient: str = "",
        overwrite_targeted: bool = True,
    ) -> None:
        """
        Pickle and publish an algorithm-specific coordination message.

        Serializes msg with pickle.dumps() and publishes it as a COORDINATION
        message via publish_message().

        Args:
            msg: Coordination message instance to send.
            recipient: Specific recipient agent ID, or "" for broadcast.
            overwrite_targeted: For broadcasts, whether to clear any pending
                               recipient-specific message in the comms manager
                               buffer for this agent.

        """
        self._publish_message(AgentMessage.COORDINATION, pickle.dumps(msg), recipient, overwrite_targeted)

    # -------------------------------------------------------------------------
    # Message Receiving
    # -------------------------------------------------------------------------

    def _on_incoming_message(self, msg: AgentMessage) -> None:
        """
        Comms manager message callback.

        Deserializes payload and dispatches to appropriate handler.
        """
        if msg.msg_type == AgentMessage.HEARTBEAT:
            self.on_heartbeat(pickle.loads(msg.payload))
        elif msg.msg_type == AgentMessage.COORDINATION:
            self.on_coordination(pickle.loads(msg.payload))

    # -------------------------------------------------------------------------
    # Navigation Interface
    # -------------------------------------------------------------------------

    def navigate_to(self, goal: PoseStamped) -> None:
        """
        Send a NavigateToPose goal to Nav2.

        Cancels any active goal before sending the new one.
        Sets _nav_status to NAVIGATING.

        Rapid calls honor the last goal sent.
        """
        self._pending_goal = goal

        if self._current_nav_goal is not None:
            self._current_nav_goal.cancel_goal_async()
        else:
            self._send_nav_goal(goal)

    def cancel_navigation(self) -> None:
        """
        Cancel the active navigation goal if one exists.

        Sets _nav_status to IDLE.
        """
        self._pending_goal = None
        if self._current_nav_goal is not None:
            self._current_nav_goal.cancel_goal_async()
        else:
            self._nav_status = NavStatus.IDLE

    def _send_nav_goal(self, goal: PoseStamped) -> None:
        """Send a NavigateToPose goal to Nav2."""
        send_goal_future = self._nav_client.send_goal_async(
            NavigateToPose.Goal(pose=goal),
            feedback_callback=self._on_nav_feedback,
        )
        send_goal_future.add_done_callback(self._on_nav_goal_response)

    def _on_nav_goal_response(self, future: Future[ClientGoalHandle]) -> None:
        """Handle Nav2 goal acceptance or rejection."""
        goal_handle = future.result()
        if goal_handle is None:
            self.get_logger().warn("Navigation goal handle is None")
            self._nav_status = NavStatus.FAILED
            self._current_nav_goal = None
            self.on_navigation_failed("Goal handle is None")
            return
        elif goal_handle.accepted:
            self._current_nav_goal = goal_handle
            self._nav_status = NavStatus.NAVIGATING
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._on_nav_result)
        else:
            self.get_logger().warn("Navigation goal rejected")
            self._nav_status = NavStatus.FAILED
            self._current_nav_goal = None
            self.on_navigation_failed("Goal rejected")

    def _on_nav_feedback(self, feedback_msg: NavigateToPose.Impl.FeedbackMessage) -> None:
        """Forward Nav2 feedback to the virtual hook."""
        self.on_navigation_feedback(feedback_msg.feedback)

    def _on_nav_result(self, future: Future) -> None:
        """Handle Nav2 result, dispatching to succeeded/failed hooks."""
        result = future.result()
        self._current_nav_goal = None
        if result is None:
            self.get_logger().warn("Navigation result is None")
            self._nav_status = NavStatus.FAILED
            self.on_navigation_failed("Result is None")
            return
        status = result.status

        # 1. Determine outcome and update status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self._nav_status = NavStatus.SUCCEEDED
            self.on_navigation_succeeded()
        elif status == GoalStatus.STATUS_CANCELED:
            self._nav_status = NavStatus.IDLE
        else:
            self._nav_status = NavStatus.FAILED
            self.on_navigation_failed(f"Navigation ended with status {status}")

        # 2. If a new goal was queued, send it (overrides status above)
        if self._pending_goal is not None:
            goal = self._pending_goal
            self._pending_goal = None
            self._send_nav_goal(goal)

    # -------------------------------------------------------------------------
    # Map Subscription Handler
    # -------------------------------------------------------------------------

    def _on_pose_updated(self, msg: PoseWithCovarianceStamped) -> None:
        """
        Cache the latest pose from /{agent_id}/amcl_pose.

        Updates _current_pose so publish_heartbeat() always has a fresh value.
        """
        self._current_pose = msg.pose.pose

    # -------------------------------------------------------------------------
    # Localization Initialization
    # -------------------------------------------------------------------------

    def _on_initial_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """Cache initial pose message for use during on_activate."""
        self._initial_pose_msg = msg

    def _wait_and_set_initial_pose(self) -> TransitionCallbackReturn:
        """Wait for initial pose message, then call set_initial_pose service synchronously."""
        timeout = 30.0  # TODO: Magic number
        elapsed = 0.0
        while self._initial_pose_msg is None and elapsed < timeout:
            self.get_logger().info("Waiting for initial pose message...", once=True)
            time.sleep(0.01)
            elapsed += 0.01
        if self._initial_pose_msg is None:
            self.get_logger().error(f"Initial pose not received after {timeout}s")
            return TransitionCallbackReturn.FAILURE

        if not self._set_initial_pose_client.wait_for_service(timeout_sec=10.0):  # TODO: Magic number
            self.get_logger().error("set_initial_pose service not available after 10s")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info("Calling set_initial_pose service")
        request = SetInitialPose.Request(pose=self._initial_pose_msg)
        for attempt in range(1, self._amcl_initialization_service_call_max_attempts + 1):
            future = self._set_initial_pose_client.call_async(request)
            elapsed = 0.0
            while not future.done() and elapsed < self._amcl_initialization_service_call_timeout:
                time.sleep(0.01)
                elapsed += 0.01
            if future.done() and future.result() is not None:
                self.get_logger().info("Initial pose set successfully")
                return TransitionCallbackReturn.SUCCESS
            self.get_logger().warn(
                f"set_initial_pose attempt {attempt}/{self._amcl_initialization_service_call_max_attempts} failed, retrying..."
            )
        self.get_logger().error(
            f"set_initial_pose failed after {self._amcl_initialization_service_call_max_attempts} attempts"
        )
        return TransitionCallbackReturn.FAILURE

    def _wait_and_reinitialize_global_localization(self) -> TransitionCallbackReturn:
        """Call reinitialize_global_localization service synchronously."""
        if not self._reinit_global_loc_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error("reinitialize_global_localization service not available after 10s")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info("Calling reinitialize_global_localization service")
        request = Empty.Request()
        for attempt in range(1, self._service_call_max_attempts + 1):
            future = self._reinit_global_loc_client.call_async(request)
            elapsed = 0.0
            while not future.done() and elapsed < self._service_call_timeout:
                time.sleep(0.01)
                elapsed += 0.01
            if future.done():
                self.get_logger().info("Global localization reinitialized")
                return TransitionCallbackReturn.SUCCESS
            self.get_logger().warn(
                f"reinitialize_global_localization attempt {attempt}/{self._service_call_max_attempts} "
                "failed, retrying..."
            )
        self.get_logger().error(
            f"reinitialize_global_localization failed after {self._service_call_max_attempts} attempts"
        )
        return TransitionCallbackReturn.SUCCESS

    # -------------------------------------------------------------------------
    # Map Subscription Handler
    # -------------------------------------------------------------------------

    def _on_map_updated(self, msg: OccupancyGrid) -> None:
        """
        Map update callback.

        Updates _map with the latest occupancy grid from slam_toolbox or the
        known-map node. Expands the belief grid to cover the union of old and
        new map extents. Calls on_map_updated() hook after updating.
        """
        old_map_info = self._map_info
        self._map = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self._map_info = msg.info

        if self._belief is None:
            self._belief = np.zeros((msg.info.height, msg.info.width), dtype=np.float32)
            self._eliminated = np.zeros((msg.info.height, msg.info.width), dtype=np.bool_)
        elif old_map_info is not None:
            self._expand_belief(old_map_info, msg.info)

        self.on_map_updated()

    def _expand_grid(
        self,
        grid: npt.NDArray,
        old_info: MapMetaData,
        new_info: MapMetaData,
    ) -> npt.NDArray:
        """
        Expand a grid array to cover the union of old and new map extents.

        Returns the expanded grid (or the original if no expansion needed).
        """
        res = old_info.resolution

        ox_old = old_info.origin.position.x
        oy_old = old_info.origin.position.y
        ox_new = new_info.origin.position.x
        oy_new = new_info.origin.position.y

        # Union bounding box
        out_ox = min(ox_old, ox_new)
        out_oy = min(oy_old, oy_new)
        out_x_max = max(ox_old + old_info.width * res, ox_new + new_info.width * res)
        out_y_max = max(oy_old + old_info.height * res, oy_new + new_info.height * res)

        out_h = math.ceil((out_y_max - out_oy) / res)
        out_w = math.ceil((out_x_max - out_ox) / res)

        # Early exit if no expansion needed
        if out_h == old_info.height and out_w == old_info.width and out_ox == ox_old and out_oy == oy_old:
            return grid

        # Create expanded array, copy old data into correct position
        expanded = np.zeros((out_h, out_w), dtype=grid.dtype)
        row_off = round((oy_old - out_oy) / res)
        col_off = round((ox_old - out_ox) / res)
        expanded[row_off : row_off + old_info.height, col_off : col_off + old_info.width] = grid

        return expanded

    def _expand_belief(self, old_info: MapMetaData, new_info: MapMetaData) -> None:
        """Expand belief and eliminated arrays to cover the union of old and new map extents."""
        if self._belief is None or self._eliminated is None:
            raise ValueError("Belief and eliminated arrays must be initialized before expanding")
        self._belief = self._expand_grid(self._belief, old_info, new_info)
        self._eliminated = self._expand_grid(self._eliminated, old_info, new_info)

    # -------------------------------------------------------------------------
    # Service Handlers
    # -------------------------------------------------------------------------

    def _handle_get_map(self, request: GetMap.Request, response: GetMap.Response) -> object:
        """Service handler: return current environment map."""
        if self._map is not None:
            response.map = self._map_to_occupancy_grid(self._map)
        return response

    def _handle_set_map(self, request: SetMap.Request, response: SetMap.Response) -> object:
        self._map = np.array(request.map.data, dtype=np.int8).reshape(request.map.info.height, request.map.info.width)
        return response

    def _handle_get_belief(self, request: GetMap.Request, response: GetMap.Response) -> object:
        """Service handler: return current belief/coverage grid."""
        if self._belief is not None:
            response.map = self._belief_to_occupancy_grid(self._belief)
        return response

    def _handle_set_belief(self, request: SetMap.Request, response: SetMap.Response) -> object:
        self._belief, self._eliminated = self._occupancy_grid_to_belief_and_eliminated(request.map)
        return response

    def _handle_fusion_complete(self, request: object, response: object) -> object:
        """
        Service handler: called by comms manager after both map and belief are set.

        Calls on_fusion_completed() hook with new map and belief.
        """
        self.on_fusion_completed()
        return response

    def _occupancy_grid_to_belief_and_eliminated(
        self, grid: OccupancyGrid
    ) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        raw = np.array(grid.data, dtype=np.int8).reshape(grid.info.height, grid.info.width)
        eliminated = raw == ELIMINATED_VALUE
        belief = raw.astype(np.float32) / SCALE
        belief[eliminated] = -np.inf
        return belief, eliminated

    def _copy_map_info(self) -> MapMetaData:
        """Create a fresh MapMetaData from cached fields to avoid C-level assertion failures."""
        src = self._map_info
        info = MapMetaData()
        info.resolution = src.resolution
        info.width = src.width
        info.height = src.height
        info.origin = src.origin
        return info

    def _belief_to_occupancy_grid(self, belief: npt.NDArray[np.float32]) -> OccupancyGrid:
        grid = (np.clip(belief, LOG_ODDS_MIN, LOG_ODDS_MAX) * SCALE).astype(np.int8)
        if self._eliminated is not None:
            grid[self._eliminated] = ELIMINATED_VALUE
        data = grid.flatten(order="C").tolist()
        msg = OccupancyGrid(data=data)
        msg.info = self._copy_map_info()
        return msg

    def _map_to_occupancy_grid(self, map: npt.NDArray[np.int8]) -> OccupancyGrid:
        data = map.flatten(order="C").tolist()
        msg = OccupancyGrid(data=data)
        msg.info = self._copy_map_info()
        return msg

    # -------------------------------------------------------------------------
    # Belief Update and Target Detection
    # -------------------------------------------------------------------------

    def _on_lidar_callback(self, scan: LaserScan) -> None:
        """
        Handle lidar scan internally.

        1. Traces all rays via Bresenham's algorithm (once)
        2. Updates belief/coverage grid via _update_belief_from_scan
        3. Checks for target detection via _check_for_targets
        4. Forwards to subclass via on_lidar_scan for algorithm-specific processing
        """
        if self._current_pose is None or self._belief is None or self._map_info is None or self._eliminated is None:
            warning = "_on_lidar_callback is exiting early: "
            if self._current_pose is None:
                warning += "pose, "
            if self._belief is None or self._eliminated is None or self._map_info is None:
                warning += "map data, "
            warning = warning.rstrip(", ")
            warning += " not set."
            self.get_logger().warn(warning, throttle_duration_sec=1.0)
            return

        all_rr, all_cc = self._trace_scan_rays(scan)
        self._update_belief_from_scan(all_rr, all_cc)
        self._check_for_targets(all_rr, all_cc)
        self.on_lidar_scan(scan)

    def _trace_scan_rays(self, scan: LaserScan) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        Trace all lidar rays using Bresenham's algorithm.

        Return the concatenated (row, col) arrays for every cell touched by
        every finite-range beam in the scan.
        """
        if self._current_pose is None or self._map_info is None:
            raise ValueError("Current pose and map info must be set before tracing scan rays")

        pose = self._current_pose
        robot_x = pose.position.x
        robot_y = pose.position.y
        robot_yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)

        res = self._map_info.resolution
        ox = self._map_info.origin.position.x
        oy = self._map_info.origin.position.y

        robot_row = int((robot_y - oy) / res)
        robot_col = int((robot_x - ox) / res)

        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)

        valid_mask = np.isfinite(ranges)
        valid_angles = angles[valid_mask] + robot_yaw
        valid_ranges = ranges[valid_mask]

        end_rows = ((robot_y + valid_ranges * np.sin(valid_angles) - oy) / res).astype(int)
        end_cols = ((robot_x + valid_ranges * np.cos(valid_angles) - ox) / res).astype(int)

        all_rr: list[npt.NDArray[np.intp]] = []
        all_cc: list[npt.NDArray[np.intp]] = []
        for i in range(len(valid_ranges)):
            rr, cc = skimage_line(robot_row, robot_col, end_rows[i], end_cols[i])
            all_rr.append(rr)
            all_cc.append(cc)

        rr = np.concatenate(all_rr)
        cc = np.concatenate(all_cc)

        # Filter out-of-bounds indices (rays can extend beyond the map)
        rows = self._map_info.height
        cols = self._map_info.width
        mask = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        return rr[mask], cc[mask]

    def _update_belief_from_scan(self, all_rr: npt.NDArray[np.intp], all_cc: npt.NDArray[np.intp]) -> None:
        """
        Eliminate all cells visible in the scan from the target belief grid.

        Sets belief to -inf and marks cells as eliminated for every cell
        touched by the pre-traced rays.
        """
        if self._eliminated is None or self._belief is None:
            raise ValueError("Eliminated and belief arrays must be initialized before updating")
        self._eliminated[all_rr, all_cc] = True
        self._belief[all_rr, all_cc] = -np.inf

    def _check_for_targets(self, all_rr: npt.NDArray[np.intp], all_cc: npt.NDArray[np.intp]) -> None:
        """
        Check if any traced ray cell is within target_radius of an unfound target.

        Converts cell indices to world coordinates and checks proximity.
        When a new target is found, updates _found_targets, publishes, and
        calls on_target_detected().
        """
        if not self._target_positions:
            self.get_logger().warn("Target positions are not set, skipping target detection", once=True)
            return

        if self._map_info is None:
            self.get_logger().warn("Map info is not set, skipping target detection", throttle_duration_sec=1.0)
            return

        unfound = [(i, pos) for i, pos in enumerate(self._target_positions) if i not in self._found_targets]
        if not unfound:
            self.get_logger().info("All targets found, skipping target detection", once=True)
            return

        res = self._map_info.resolution
        ox = self._map_info.origin.position.x
        oy = self._map_info.origin.position.y

        # Convert cell indices to world coordinates (cell centers)
        world_x = all_cc.astype(np.float64) * res + ox + res / 2.0
        world_y = all_rr.astype(np.float64) * res + oy + res / 2.0

        # Check each unfound target against all ray points
        newly_found = []
        radius_sq = self._target_radius**2
        for target_idx, (tx, ty) in unfound:
            dist_sq = (world_x - tx) ** 2 + (world_y - ty) ** 2
            if np.any(dist_sq <= radius_sq):
                self._found_targets.add(target_idx)
                newly_found.append((target_idx, (tx, ty)))

        if newly_found:
            self._publish_found_targets()
            self.on_target_detected([pos for _, pos in newly_found])

    def _publish_found_targets(self) -> None:
        """
        Publish the current list of found target indices to /{agent_id}/found_targets.

        Message contains sorted list of indices corresponding to _target_positions.
        This topic is for external success criteria evaluation only.
        """
        self.pub_found_targets.publish(Int32MultiArray(data=sorted(self._found_targets)))

    # -------------------------------------------------------------------------
    # Abstract Callbacks (Must Implement)
    # -------------------------------------------------------------------------

    @abstractmethod
    def on_heartbeat(self, msg: HeartbeatMessage) -> None:
        """Override to handle a heartbeat received from another agent."""

    @abstractmethod
    def on_coordination(self, msg: BaseCoordinationMessage) -> None:
        """
        Override to handle a coordination message received from another agent.

        The base class deserializes the payload automatically; msg is the
        unpickled BaseCoordinationMessage subclass instance sent by the peer.
        Use isinstance() to determine the concrete type and read its fields.

        This is where algorithm-specific messages are handled, including:
        - Task assignments
        - Rendezvous commands
        - ACK / NACK
        - Exploration intentions
        - Any other coordination messages

        Note: Coordination messages are only received from close-range agents.
        """

    @abstractmethod
    def on_lidar_scan(self, scan: LaserScan) -> None:
        """
        Override to handle a lidar scan received. Called after base class has updated belief and checked for targets.

        Subclass can use this for algorithm-specific processing (e.g., frontier
        identification) but does NOT need to handle belief updates or target detection.
        """

    @abstractmethod
    def on_target_detected(self, target_locations: list[tuple[float, float]]) -> None:
        """
        Override to handle target detections. Called once per scan when new targets are found.

        Args:
            target_locations: List of (x, y) positions for each newly detected target.

        Subclass should implement this to handle target discovery, typically by:
        - Publishing a COORDINATION message to notify other agents
        - Updating internal search state
        - Potentially triggering a rendezvous or task reassignment

        Note: The base class has already updated _found_targets and published
        to the found_targets topic before calling this hook.

        """

    # -------------------------------------------------------------------------
    # Virtual Hooks (Optional Override)
    # -------------------------------------------------------------------------

    def on_map_updated(self) -> None:
        """
        Override to handle map updates. Called after _map is updated from the map topic subscription.

        NOT called during fusion - use on_fusion_completed for that.
        Override to trigger replanning or other responses.
        Default: no-op.
        """

    def on_fusion_completed(self) -> None:
        """
        Override to handle fusion completion. Called after map and belief are updated via fusion with another agent.

        on_map_updated and on_belief_updated are NOT called when fusion occurs;
        this hook is the sole notification for fusion-driven grid updates.
        Override to trigger replanning or respond to newly discovered information.
        Default: no-op.
        """

    def on_navigation_feedback(self, feedback: NavigateToPose.Feedback) -> None:
        """Override to handle Nav2 navigation feedback. Default: no-op."""

    def on_navigation_succeeded(self) -> None:
        """Override to handle when Nav2 reports goal reached. Default: no-op."""

    def on_navigation_failed(self, reason: str) -> None:
        """
        Override to handle when Nav2 reports failure or the goal is aborted.

        Subclass should typically select a new goal or trigger recovery.
        Default: no-op.
        """
