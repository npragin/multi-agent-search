"""
Agent Base Class for the multi-agent search system.

Provides common communication infrastructure, Nav2 integration, belief grid
management, and target detection for concrete agent implementations.
"""

from __future__ import annotations

import math
import pickle
import time
from abc import ABC, abstractmethod
from typing import overload

import numpy as np
import numpy.typing as npt
from scipy.ndimage import affine_transform as scipy_affine_transform
from skimage.draw import line as skimage_line

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import SetInitialPose
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.client import Client
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.publisher import Publisher
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.service import Service
from rclpy.subscription import Subscription
from rclpy.task import Future
from rclpy.timer import Timer
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty, Trigger

from multi_agent_search.types import (
    BaseCoordinationMessage,
    HeartbeatMessage,
    NavStatus,
)
from multi_agent_search_interfaces.msg import AgentMessage
from multi_agent_search_interfaces.srv import GetMap, SetMap, TargetDetected

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
        if self.use_known_map:
            if self.known_initial_poses:
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

        self.declare_parameter("amcl_initialization_service_call_timeout", 5.0)
        self.declare_parameter("amcl_initialization_service_call_max_attempts", 3)

    def _set_up_state_defaults(self) -> None:
        """Set all instance attributes to safe defaults before on_configure."""
        self.agent_id: str = ""
        self.use_known_map: bool = False
        self.known_initial_poses: bool = False
        self.map: npt.NDArray[np.int8] | None = None
        self.belief: npt.NDArray[np.float32] | None = None
        self.eliminated: npt.NDArray[np.bool_] | None = None
        self.map_info: MapMetaData | None = None
        self._amcl_initialization_service_call_timeout: float = 5.0
        self._amcl_initialization_service_call_max_attempts: int = 3
        self._initial_pose_msg: PoseWithCovarianceStamped | None = None
        self.nav_status: NavStatus = NavStatus.IDLE
        self._current_nav_goal: (
            ClientGoalHandle[NavigateToPose.Goal, NavigateToPose.Result, NavigateToPose.Feedback] | None
        ) = None
        self._pending_goal: PoseStamped | None = None
        self.current_pose: Pose | None = None

        # Lifecycle-managed interface lists for clean teardown
        self._managed_timers: list[Timer] = []
        self._managed_subscriptions: list[
            Subscription[AgentMessage | LaserScan | OccupancyGrid | PoseWithCovarianceStamped]
        ] = []
        self._managed_publishers: list[Publisher[AgentMessage]] = []
        self._managed_service_servers: list[
            Service[GetMap.Request, GetMap.Response]
            | Service[SetMap.Request, SetMap.Response]
            | Service[GetMap.Request, GetMap.Response]
            | Service[SetMap.Request, SetMap.Response]
            | Service[Trigger.Request, Trigger.Response]
            | Service[TargetDetected.Request, TargetDetected.Response]
        ] = []
        self._managed_service_clients: list[
            Client[Empty.Request, Empty.Response] | Client[SetInitialPose.Request, SetInitialPose.Response]
        ] = []
        self._managed_action_clients: list[
            ActionClient[NavigateToPose.Goal, NavigateToPose.Result, NavigateToPose.Feedback]
        ] = []

    def _set_up_state(self) -> None:
        """Set up state for the agent."""
        # Identity
        self.agent_id = self.get_parameter("agent_id").value
        self.use_known_map = self.get_parameter("use_known_map").value
        self.known_initial_poses = self.get_parameter("known_initial_poses").value

        if self.known_initial_poses and not self.use_known_map:
            raise ValueError("known_initial_poses requires use_known_map to be true (AMCL must be running)")

        # Map and belief state
        self.map = None
        self.belief = None
        self.eliminated = None
        self.map_info = None

        # Service call retry parameters
        self._amcl_initialization_service_call_timeout = self.get_parameter(
            "amcl_initialization_service_call_timeout"
        ).value
        self._amcl_initialization_service_call_max_attempts = self.get_parameter(
            "amcl_initialization_service_call_max_attempts"
        ).value

        # Localization state
        self._initial_pose_msg = None

        # Navigation state
        self.nav_status = NavStatus.IDLE
        self._current_nav_goal = None
        self._pending_goal = None

        # Pose state
        self.current_pose = None

    def _set_up_subscribers(self) -> None:
        """Set up subscribers for the agent."""
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self._sub_incoming = self.create_subscription(
            AgentMessage, f"/comms/input/{self.agent_id}", self._on_incoming_message, 10
        )
        self._sub_lidar = self.create_subscription(
            LaserScan, f"/{self.agent_id}/base_scan", self._on_lidar_callback, 10
        )
        self._sub_map = self.create_subscription(
            OccupancyGrid, f"/{self.agent_id}/map", self._on_map_updated, latched_qos
        )
        self._sub_pose = self.create_subscription(
            PoseWithCovarianceStamped,
            f"/{self.agent_id}/pose",
            self._on_pose_updated,
            latched_qos if self.use_known_map else 10,  # AMCL publishes a latched topic
        )
        self._managed_subscriptions.extend([self._sub_incoming, self._sub_lidar, self._sub_map, self._sub_pose])

        if self.known_initial_poses:
            self._sub_initial_pose = self.create_subscription(
                PoseWithCovarianceStamped,
                f"/{self.agent_id}/initialpose",
                self._on_initial_pose,
                latched_qos,
                callback_group=self._localization_cbg,
            )
            self._managed_subscriptions.append(self._sub_initial_pose)

    def _set_up_publishers(self) -> None:
        """Set up lifecycle publishers for the agent (only transmit when active)."""
        self._pub_outgoing = self.create_lifecycle_publisher(AgentMessage, f"/comms/output/{self.agent_id}", 10)
        self._managed_publishers.append(self._pub_outgoing)

    def _set_up_service_servers(self) -> None:
        """Create map/belief get/set services and fusion_complete service."""
        self._srv_get_map = self.create_service(GetMap, f"/{self.agent_id}/get_map", self._handle_get_map)
        self._srv_set_map = self.create_service(SetMap, f"/{self.agent_id}/set_map", self._handle_set_map)
        self._srv_get_belief = self.create_service(GetMap, f"/{self.agent_id}/get_belief", self._handle_get_belief)
        self._srv_set_belief = self.create_service(SetMap, f"/{self.agent_id}/set_belief", self._handle_set_belief)
        self._srv_fusion_complete = self.create_service(
            Trigger, f"/{self.agent_id}/fusion_complete", self._handle_fusion_complete
        )
        self._srv_target_detected = self.create_service(
            TargetDetected, f"/{self.agent_id}/target_detected", self._handle_target_detected
        )
        self._managed_service_servers.extend(
            [
                self._srv_get_map,
                self._srv_set_map,
                self._srv_get_belief,
                self._srv_set_belief,
                self._srv_fusion_complete,
                self._srv_target_detected,
            ]
        )

    def _set_up_action_clients(self) -> None:
        """Create NavigateToPose action client at /{agent_id}/navigate_to_pose."""
        self._nav_client: ActionClient[NavigateToPose.Goal, NavigateToPose.Result, NavigateToPose.Feedback] = (
            ActionClient(self, NavigateToPose, f"/{self.agent_id}/navigate_to_pose")
        )
        self._managed_action_clients.append(self._nav_client)

    def _set_up_service_clients(self) -> None:
        """
        Create service clients.

        Localization clients use a ReentrantCallbackGroup so on_activate can poll-wait without deadlocking.
        """
        self._localization_cbg = ReentrantCallbackGroup()
        if self.use_known_map and not self.known_initial_poses:
            self._reinit_global_loc_client: Client[Empty.Request, Empty.Response] = self.create_client(
                Empty, f"/{self.agent_id}/reinitialize_global_localization", callback_group=self._localization_cbg
            )
            self._managed_service_clients.append(self._reinit_global_loc_client)
        if self.known_initial_poses:
            self._set_initial_pose_client: Client[SetInitialPose.Request, SetInitialPose.Response] = self.create_client(
                SetInitialPose, f"/{self.agent_id}/set_initial_pose", callback_group=self._localization_cbg
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
        if recipient == self.agent_id:
            self.get_logger().info("Skipping publishing message to self")
            return
        self._pub_outgoing.publish(
            AgentMessage(
                msg_type=msg_type,
                sender_id=self.agent_id,
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
            sender_id=self.agent_id, timestamp=self.get_clock().now().nanoseconds / 1e9, pose=self.current_pose
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
            self.nav_status = NavStatus.IDLE

    def _send_nav_goal(self, goal: PoseStamped) -> None:
        """Send a NavigateToPose goal to Nav2."""
        send_goal_future = self._nav_client.send_goal_async(
            NavigateToPose.Goal(pose=goal),
            feedback_callback=self._on_nav_feedback,
        )
        send_goal_future.add_done_callback(self._on_nav_goal_response)

    def _on_nav_goal_response(
        self, future: Future[ClientGoalHandle[NavigateToPose.Goal, NavigateToPose.Result, NavigateToPose.Feedback]]
    ) -> None:
        """Handle Nav2 goal acceptance or rejection."""
        goal_handle = future.result()
        if goal_handle is None:
            self.get_logger().warn("Navigation goal handle is None")
            self.nav_status = NavStatus.FAILED
            self._current_nav_goal = None
            self.on_navigation_failed("Goal handle is None")
            return
        elif goal_handle.accepted:
            self._current_nav_goal = goal_handle
            self.nav_status = NavStatus.NAVIGATING
            result_future = goal_handle.get_result_async()
            result_future.add_done_callback(self._on_nav_result)
        else:
            self.get_logger().warn("Navigation goal rejected")
            self.nav_status = NavStatus.FAILED
            self._current_nav_goal = None
            self.on_navigation_failed("Navigation goal rejected")

    def _on_nav_feedback(self, feedback_msg: NavigateToPose.Feedback) -> None:
        """Forward Nav2 feedback to the virtual hook."""
        self.on_navigation_feedback(feedback_msg.feedback)

    def _on_nav_result(self, future: Future[NavigateToPose.Result]) -> None:
        """Handle Nav2 result, dispatching to succeeded/failed hooks."""
        result = future.result()
        self._current_nav_goal = None
        if result is None:
            self.get_logger().warn("Navigation result is None")
            self.nav_status = NavStatus.FAILED
            self.on_navigation_failed("Navigation result is None")
            return
        status = result.status

        # 1. Determine outcome and update status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.nav_status = NavStatus.SUCCEEDED
            self.on_navigation_succeeded()
        elif status == GoalStatus.STATUS_CANCELED:
            self.nav_status = NavStatus.IDLE
        else:
            self.nav_status = NavStatus.FAILED
            self.on_navigation_failed(result.result.error_msg)

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
        When a previous pose exists, applies a rigid-body transform to the
        belief and eliminated grids to compensate for AMCL pose corrections.
        """
        new_pose = msg.pose.pose
        if self.current_pose is not None:
            self._transform_belief_grids(self.current_pose, new_pose)
        self.current_pose = new_pose

    def _transform_belief_grids(self, old_pose: Pose, new_pose: Pose) -> None:
        """
        Apply a rigid-body transform to belief and eliminated grids to compensate for pose corrections.

        Computes the delta (dx, dy, dyaw) between old and new pose, converts to
        grid coordinates, and applies an affine transform using nearest-neighbor
        interpolation. Out-of-bounds cells are treated as unobserved.
        """
        if self.belief is None or self.eliminated is None or self.map_info is None:
            return

        # Compute pose delta in world coordinates
        dx = new_pose.position.x - old_pose.position.x
        dy = new_pose.position.y - old_pose.position.y
        old_yaw = 2.0 * math.atan2(old_pose.orientation.z, old_pose.orientation.w)
        new_yaw = 2.0 * math.atan2(new_pose.orientation.z, new_pose.orientation.w)
        dyaw = math.atan2(math.sin(new_yaw - old_yaw), math.cos(new_yaw - old_yaw))

        # Convert translation to grid cells (row = y, col = x)
        res = self.map_info.resolution
        dr = dy / res
        dc = dx / res

        # Skip if negligible
        if abs(dr) < 0.5 and abs(dc) < 0.5 and abs(dyaw) < 1e-3:
            return

        # Rotation center = old robot position in grid coords
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y
        cr = (old_pose.position.y - oy) / res
        cc = (old_pose.position.x - ox) / res

        # Inverse rotation matrix in (row, col) space for scipy's inverse mapping.
        # In grid coords (row=y, col=x), the forward rotation has flipped off-diagonal
        # signs vs standard (x,y) rotation, so R_grid_inv = [[cos, -sin], [sin, cos]]
        # which is the standard form with dyaw (not -dyaw).
        cos_d = math.cos(dyaw)
        sin_d = math.sin(dyaw)
        r_inv = np.array([[cos_d, -sin_d], [sin_d, cos_d]], dtype=np.float64)

        # Offset: R_inv @ [-(cr+dr), -(cc+dc)]^T + [cr, cc]^T
        shifted_center = np.array([-(cr + dr), -(cc + dc)], dtype=np.float64)
        offset = r_inv @ shifted_center + np.array([cr, cc], dtype=np.float64)

        # Transform belief grid (float32, cval=0.0 means unobserved)
        self.belief = scipy_affine_transform(
            self.belief, r_inv, offset=offset, order=0, mode="constant", cval=0.0, output=np.float32
        )

        # Transform eliminated grid (cast to float32, threshold back to bool)
        elim_float = self.eliminated.astype(np.float32)
        elim_transformed = scipy_affine_transform(
            elim_float, r_inv, offset=offset, order=0, mode="constant", cval=0.0, output=np.float32
        )
        self.eliminated = elim_transformed > 0.5

        # Ensure consistency: eliminated cells must have -inf belief
        self.belief[self.eliminated] = -np.inf

    # -------------------------------------------------------------------------
    # Localization Initialization
    # -------------------------------------------------------------------------

    def _on_initial_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """Cache initial pose message for use during on_activate."""
        self._initial_pose_msg = msg

    def _wait_and_set_initial_pose(self) -> TransitionCallbackReturn:
        """Wait for initial pose message, then call set_initial_pose service synchronously."""
        elapsed = 0.0
        attempts = self._amcl_initialization_service_call_max_attempts
        timeout = self._amcl_initialization_service_call_timeout
        while self._initial_pose_msg is None and elapsed < timeout:
            self.get_logger().info("Waiting for initial pose message...", once=True)
            time.sleep(0.01)
            elapsed += 0.01
        if self._initial_pose_msg is None:
            self.get_logger().error(f"Initial pose not received after {timeout}s")
            return TransitionCallbackReturn.FAILURE

        if not self._set_initial_pose_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"set_initial_pose service not available after {timeout}s")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info("Calling set_initial_pose service")
        request = SetInitialPose.Request(pose=self._initial_pose_msg)
        for attempt in range(1, attempts + 1):
            future = self._set_initial_pose_client.call_async(request)
            elapsed = 0.0
            while not future.done() and elapsed < timeout:
                time.sleep(0.01)
                elapsed += 0.01
            if future.done() and future.result() is not None:
                self.get_logger().info("Initial pose set successfully")
                return TransitionCallbackReturn.SUCCESS
            self.get_logger().warn(f"set_initial_pose attempt {attempt}/{attempts} failed, retrying...")
        self.get_logger().error(f"set_initial_pose failed after {attempts} attempts")
        return TransitionCallbackReturn.FAILURE

    def _wait_and_reinitialize_global_localization(self) -> TransitionCallbackReturn:
        """Call reinitialize_global_localization service synchronously."""
        timeout = self._amcl_initialization_service_call_timeout
        attempts = self._amcl_initialization_service_call_max_attempts
        if not self._reinit_global_loc_client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(f"reinitialize_global_localization service not available after {timeout}s")
            return TransitionCallbackReturn.FAILURE

        self.get_logger().info("Calling reinitialize_global_localization service")
        request = Empty.Request()
        for attempt in range(1, attempts + 1):
            future = self._reinit_global_loc_client.call_async(request)
            elapsed = 0.0
            while not future.done() and elapsed < timeout:
                time.sleep(0.01)
                elapsed += 0.01
            if future.done():
                self.get_logger().info("Global localization reinitialized")
                return TransitionCallbackReturn.SUCCESS
            self.get_logger().warn(f"reinitialize_global_localization attempt {attempt}/{attempts} failed, retrying...")
        self.get_logger().error(f"reinitialize_global_localization failed after {attempts} attempts")
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
        old_map_info = self.map_info
        self.map = np.array(msg.data, dtype=np.int8).reshape(msg.info.height, msg.info.width)
        self.map_info = msg.info

        if self.belief is None:
            self.belief = np.zeros((msg.info.height, msg.info.width), dtype=np.float32)
            self.eliminated = np.zeros((msg.info.height, msg.info.width), dtype=np.bool_)
        elif old_map_info is not None:
            self._expand_belief(old_map_info, msg.info)

        self.on_map_updated()

    def _expand_belief(self, old_info: MapMetaData, new_info: MapMetaData) -> None:
        """Expand belief and eliminated arrays to cover the union of old and new map extents."""
        if self.belief is None or self.eliminated is None:
            raise ValueError("Belief and eliminated arrays must be initialized before expanding")
        self.belief = self._expand_grid(self.belief, old_info, new_info)
        self.eliminated = self._expand_grid(self.eliminated, old_info, new_info)

    @overload
    def _expand_grid(
        self,
        grid: npt.NDArray[np.float32],
        old_info: MapMetaData,
        new_info: MapMetaData,
    ) -> npt.NDArray[np.float32]: ...

    @overload
    def _expand_grid(
        self,
        grid: npt.NDArray[np.bool_],
        old_info: MapMetaData,
        new_info: MapMetaData,
    ) -> npt.NDArray[np.bool_]: ...

    def _expand_grid(
        self,
        grid: npt.NDArray[np.bool_] | npt.NDArray[np.float32],
        old_info: MapMetaData,
        new_info: MapMetaData,
    ) -> npt.NDArray[np.bool_] | npt.NDArray[np.float32]:
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

    # -------------------------------------------------------------------------
    # Service Handlers
    # -------------------------------------------------------------------------

    def _handle_get_map(self, request: GetMap.Request, response: GetMap.Response) -> object:
        """Service handler: return current environment map."""
        if self.map is not None:
            response.map = self._map_to_occupancy_grid(self.map)
        return response

    def _handle_set_map(self, request: SetMap.Request, response: SetMap.Response) -> object:
        self.map = np.array(request.map.data, dtype=np.int8).reshape(request.map.info.height, request.map.info.width)
        return response

    def _handle_get_belief(self, request: GetMap.Request, response: GetMap.Response) -> object:
        """Service handler: return current belief/coverage grid."""
        if self.belief is not None and self.eliminated is not None:
            response.map = self._belief_and_eliminated_to_occupancy_grid(self.belief, self.eliminated)
        return response

    def _handle_set_belief(self, request: SetMap.Request, response: SetMap.Response) -> object:
        self.belief, self.eliminated = self._occupancy_grid_to_belief_and_eliminated(request.map)
        return response

    def _handle_fusion_complete(self, request: object, response: object) -> object:
        """
        Service handler: called by comms manager after both map and belief are set.

        Calls on_fusion_completed() hook with new map and belief.
        """
        self.on_fusion_completed()
        return response

    def _handle_target_detected(
        self, request: TargetDetected.Request, response: TargetDetected.Response
    ) -> TargetDetected.Response:
        """Service handler: convert TargetDetected request to on_target_detected hook call."""
        target_locations = [(p.x, p.y) for p in request.targets]
        self.on_target_detected(target_locations)
        response.success = True
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
        src = self.map_info
        info = MapMetaData()
        if src is not None:
            info.resolution = src.resolution
            info.width = src.width
            info.height = src.height
            info.origin = src.origin
        return info

    def _belief_and_eliminated_to_occupancy_grid(
        self, belief: npt.NDArray[np.float32], eliminated: npt.NDArray[np.bool_]
    ) -> OccupancyGrid:
        grid = (np.clip(belief, LOG_ODDS_MIN, LOG_ODDS_MAX) * SCALE).astype(np.int8)
        if eliminated is not None:
            grid[eliminated] = ELIMINATED_VALUE
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

        1. Updates belief/coverage grid via _update_belief_from_scan
        2. Forwards to subclass via on_lidar_scan for algorithm-specific processing
        """
        if self.current_pose is None or self.belief is None or self.map_info is None or self.eliminated is None:
            warning = "_on_lidar_callback is exiting early: "
            if self.current_pose is None:
                warning += "pose, "
            if self.belief is None or self.eliminated is None or self.map_info is None:
                warning += "map data, "
            warning = warning.rstrip(", ")
            warning += " not set."
            self.get_logger().warn(warning, throttle_duration_sec=1.0)
            return

        self.interrupted = True

        self._update_belief_from_scan(scan)
        self.on_lidar_scan(scan)

    def _update_belief_from_scan(self, scan: LaserScan) -> None:
        """
        Eliminate all cells visible in the scan from the target belief grid.

        Sets belief to -inf and marks cells as eliminated for every cell
        touched by the pre-traced rays.
        """
        if self.eliminated is None or self.belief is None:
            raise ValueError("Eliminated and belief arrays must be initialized before updating")

        all_rr, all_cc = self._trace_scan_rays(scan)
        self.eliminated[all_rr, all_cc] = True
        self.belief[all_rr, all_cc] = -np.inf

    def _trace_scan_rays(self, scan: LaserScan) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        Trace all lidar rays using Bresenham's algorithm.

        Return the concatenated (row, col) arrays for every cell touched by
        every finite-range beam in the scan.
        """
        if self.current_pose is None or self.map_info is None:
            raise ValueError("Current pose and map info must be set before tracing scan rays")

        pose = self.current_pose
        robot_x = pose.position.x
        robot_y = pose.position.y
        robot_yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)

        res = self.map_info.resolution
        ox = self.map_info.origin.position.x
        oy = self.map_info.origin.position.y

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
            rr, cc = skimage_line(robot_row, robot_col, end_rows[i], end_cols[i])  # type: ignore[no-untyped-call]
            all_rr.append(rr)
            all_cc.append(cc)

        rr = np.concatenate(all_rr)
        cc = np.concatenate(all_cc)

        # Filter out-of-bounds indices (rays can extend beyond the map)
        rows = self.map_info.height
        cols = self.map_info.width
        mask = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        return rr[mask], cc[mask]

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
        Override to handle a lidar scan received. Called after base class has updated belief.

        Subclass can use this for algorithm-specific processing (e.g., frontier
        identification) but does NOT need to handle belief updates.
        """

    @abstractmethod
    def on_target_detected(self, target_locations: list[tuple[float, float]]) -> None:
        """
        Override to handle target detections. Called when the TargetDetector notifies this agent discovered a target(s).

        Args:
            target_locations: List of (x, y) world positions for each newly detected target.

        Subclass should implement this to handle target discovery, typically by:
        - Publishing a COORDINATION message to notify other agents
        - Updating internal search state
        - Potentially triggering a rendezvous or task reassignment

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
