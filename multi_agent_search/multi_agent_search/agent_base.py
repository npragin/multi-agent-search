"""
Agent Base Class for the multi-agent search system.

Provides common communication infrastructure, Nav2 integration, belief grid
management, and target detection for concrete agent implementations.
"""

from __future__ import annotations

import ast
import math
import pickle
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from action_msgs.msg import GoalStatus
from skimage.draw import line as skimage_line

from geometry_msgs.msg import Pose, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle
from rclpy.node import Node
from rclpy.task import Future
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray
from std_srvs.srv import Trigger

from multi_agent_search.types import BaseCoordinationMessage, HeartbeatMessage, NavStatus
from multi_agent_search_interfaces.msg import AgentMessage
from multi_agent_search_interfaces.srv import GetMap, SetMap

LOG_ODDS_MIN, LOG_ODDS_MAX = -7.0, 7.0  # TODO: Magic number
SCALE = 127.0 / LOG_ODDS_MAX  # TODO: Magic number
ELIMINATED_VALUE = -128  # TODO: Magic number


class AgentBase(Node, ABC):
    """
    Abstract base class for search agents.

    Provides the communication interface with the CommsManager, Nav2 action
    client for navigation, belief grid management from lidar data, and target
    detection. Concrete subclasses implement the search and coordination logic.
    """

    def __init__(self, agent_id: str) -> None:
        """
        Initialize base agent with ID.

        Sets up publisher, subscriber, services, nav client, and timers.
        Declares and reads ROS2 parameters for use_known_map, target_positions,
        and target_radius.
        """
        super().__init__(agent_id)
        # TODO: Known map handling, seems we can just publish it on the same topic that slam_toolbox is using?
        # TODO: Verify slam_toolbox map topic name
        # TODO: Verify Nav2 navigation server name
        self._set_up_parameters()
        self._set_up_state()
        self._set_up_publishers()
        self._set_up_action_clients()
        self._set_up_subscribers()
        self._set_up_service_servers()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _set_up_parameters(self) -> None:
        """Set up parameters for the agent."""
        self.declare_parameter("agent_id", "robot_0")

        self.declare_parameter("use_known_map", False)

        self.declare_parameter("target_positions", "[]")
        self.declare_parameter("target_radius", 1.0)

    def _set_up_state(self) -> None:
        """Set up state for the agent."""
        # Identity
        self._agent_id: str = self.get_parameter("agent_id").value

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

        # Navigation state
        self._nav_status: NavStatus = NavStatus.IDLE
        self._current_nav_goal: ClientGoalHandle | None = None
        self._pending_goal: PoseStamped | None = None

        # Pose state
        self._current_pose: Pose | None = None

    def _set_up_subscribers(self) -> None:
        """Set up subscribers for the agent."""
        self.sub_incoming = self.create_subscription(
            AgentMessage, f"/comms/input/{self._agent_id}", self._on_incoming_message, 10
        )
        self.sub_lidar = self.create_subscription(
            LaserScan, f"/{self._agent_id}/base_scan", self._on_lidar_callback, 10
        )
        self.sub_map = self.create_subscription(OccupancyGrid, f"/{self._agent_id}/map", self._on_map_updated, 10)
        self.sub_pose = self.create_subscription(
            PoseWithCovarianceStamped, f"/{self._agent_id}/amcl_pose", self._on_pose_updated, 10
        )

    def _set_up_publishers(self) -> None:
        """Set up publishers for the agent."""
        self.pub_outgoing = self.create_publisher(AgentMessage, f"/comms/output/{self._agent_id}", 10)
        self.pub_found_targets = self.create_publisher(Int32MultiArray, f"/{self._agent_id}/found_targets", 10)

    def _set_up_service_servers(self) -> None:
        """Create map/belief get/set services and fusion_complete service."""
        self.srv_get_map = self.create_service(GetMap, f"/{self._agent_id}/get_map", self._handle_get_map)
        self.srv_set_map = self.create_service(SetMap, f"/{self._agent_id}/set_map", self._handle_set_map)
        self.srv_get_belief = self.create_service(GetMap, f"/{self._agent_id}/get_belief", self._handle_get_belief)
        self.srv_set_belief = self.create_service(SetMap, f"/{self._agent_id}/set_belief", self._handle_set_belief)
        self.srv_fusion_complete = self.create_service(
            Trigger, f"/{self._agent_id}/fusion_complete", self._handle_fusion_complete
        )

    def _set_up_action_clients(self) -> None:
        """Create NavigateToPose action client at /{agent_id}/navigate_to_pose."""
        self._nav_client: ActionClient[NavigateToPose.Goal, NavigateToPose.Result, NavigateToPose.Feedback] = (
            ActionClient(self, NavigateToPose, f"/{self._agent_id}/navigate_to_pose")
        )

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
            self.get_logger().info("_expand_grid is exiting early, no expansion needed")
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

    def _belief_to_occupancy_grid(self, belief: npt.NDArray[np.float32]) -> OccupancyGrid:
        grid = (np.clip(belief, LOG_ODDS_MIN, LOG_ODDS_MAX) * SCALE).astype(np.int8)
        if self._eliminated is not None:
            grid[self._eliminated] = ELIMINATED_VALUE
        data = grid.flatten(order="C").tolist()
        return OccupancyGrid(data=data, info=self._map_info)

    def _map_to_occupancy_grid(self, map: npt.NDArray[np.int8]) -> OccupancyGrid:
        data = map.flatten(order="C").tolist()
        return OccupancyGrid(data=data, info=self._map_info)

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
            self.get_logger().warn(
                "_on_lidar_callback is exiting early, current pose, belief, map info, or eliminated is not set"
            )
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

        return np.concatenate(all_rr), np.concatenate(all_cc)

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
            self.get_logger().warn("Target positions are not set, skipping target detection")
            return

        if self._map_info is None:
            self.get_logger().warn("Map info is not set, skipping target detection")
            return

        unfound = [(i, pos) for i, pos in enumerate(self._target_positions) if i not in self._found_targets]
        if not unfound:
            self.get_logger().warn("No unfound targets, skipping target detection")
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
