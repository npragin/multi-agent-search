"""
Communications Manager Node for the multi-agent search system.

Mediates all inter-agent communication with distance-based zone filtering,
rate limiting, and map/belief fusion (unknown map mode only).
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
import numpy.typing as npt
from skimage.draw import line as skimage_line
from std_srvs.srv import Trigger

import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.client import Client
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.time import Duration, Time

from multi_agent_search.types import CommsConfig, CommsZone
from multi_agent_search_interfaces.msg import AgentMessage
from multi_agent_search_interfaces.srv import GetMap, SetMap


class CommsManager(Node):
    """
    Central communications manager that mediates all inter-agent messaging.

    Receives messages from agents, determines pairwise communication zones
    based on distance, and propagates messages at the appropriate rate with
    message-type filtering. Also orchestrates map/belief fusion when agents
    are within fusion range with line-of-sight (unknown map mode only).
    """

    def __init__(self, agent_ids: list[str], config: CommsConfig) -> None:
        """
        Initialize comms manager with known agent IDs and configuration.

        Sets up all subscriptions (including ground truth pose), publishers,
        timers, and service clients.
        """
        super().__init__("comms_manager")

        self._set_up_parameters()
        self._set_up_state()
        self._set_up_subscribers()

        for agent_id in self.agent_ids:
            self._set_up_agent(agent_id)

        self._set_up_timers()

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def _set_up_parameters(self) -> None:
        """Set up parameters for the comms manager."""
        # Agent
        self.declare_parameter("num_agents", 0)

        # Problem settings
        self.declare_parameter("use_known_map", False)

        # Communication settings
        self.declare_parameter("close_range_threshold", 5.0)
        self.declare_parameter("long_range_threshold", 20.0)
        self.declare_parameter("fusion_range_threshold", 5.0)
        self.declare_parameter("close_range_rate", 10.0)
        self.declare_parameter("long_range_rate", 1.0)
        self.declare_parameter("fusion_cooldown", 10.0)

        # Updation rate settings
        self.declare_parameter("check_fusion_eligibility_rate", 1.0)
        self.declare_parameter("update_pairwise_zones_rate", 2.0)

    def _set_up_subscribers(self) -> None:
        """Set up subscribers for the comms manager."""
        self.create_subscription(OccupancyGrid, "/known_map", self._on_known_map_update, 10)

    def _set_up_state(self) -> None:
        """Set up state for the comms manager."""
        # Agent tracking
        num_agents = self.get_parameter("num_agents").value
        self.agent_ids: set[str] = {f"robot_{i}" for i in range(num_agents)}
        self.agent_positions: dict[str, Point | None] = dict.fromkeys(self.agent_ids, None)
        self.known_map: npt.NDArray[np.uint8] | None = None
        self.known_map_resolution: float | None = None
        self.known_map_origin: Point | None = None

        # Problem settings
        self.use_known_map = self.get_parameter("use_known_map").value

        # Communication configuration
        self.comms_config = CommsConfig(
            close_range_threshold=self.get_parameter("close_range_threshold").value,
            long_range_threshold=self.get_parameter("long_range_threshold").value,
            fusion_range_threshold=self.get_parameter("fusion_range_threshold").value,
            close_range_rate=self.get_parameter("close_range_rate").value,
            long_range_rate=self.get_parameter("long_range_rate").value,
            fusion_cooldown=self.get_parameter("fusion_cooldown").value,
        )

        # Message buffer: sender_id -> recipient_id -> message
        # recipient_id of "" (empty string) represents broadcast
        self.message_buffer: dict[str, dict[str, AgentMessage]] = {}

        # Communication zone cache: (agent_a, agent_b) sorted tuple -> zone
        self.pairwise_zones: dict[tuple[str, str], CommsZone] = {}

        # Map fusion tracking: (agent_a, agent_b) sorted tuple -> time
        self.last_fusion_time: dict[tuple[str, str], Time] = {}

        # Delivery rate tracking: (sender_id, recipient_id) -> time
        self.last_delivery_time: dict[tuple[str, str], Time] = {}

        # Per-agent ROS2 interfaces (populated by _setup_agent)
        self._message_publishers: dict[str, Publisher] = {}
        self._message_subscriptions: dict[str, Subscription] = {}
        self._pose_subscriptions: dict[str, Subscription] = {}
        self._map_get_clients: dict[str, Client] = {}
        self._map_set_clients: dict[str, Client] = {}
        self._belief_get_clients: dict[str, Client] = {}
        self._belief_set_clients: dict[str, Client] = {}
        self._fusion_complete_clients: dict[str, Client] = {}
        self._load_map_clients: dict[str, Client] = {}

    def _set_up_agent(self, agent_id: str) -> None:
        """
        Set up ROS2 interfaces for a single agent.

        Creates subscription to agent's ground truth pose topic and publisher
        for the comms manager interface.
        """
        self._message_publishers[agent_id] = self.create_publisher(AgentMessage, f"/comms/input/{agent_id}", 10)
        self._message_subscriptions[agent_id] = self.create_subscription(
            AgentMessage, f"/comms/output/{agent_id}", self._on_message, 10
        )
        self._pose_subscriptions[agent_id] = self.create_subscription(
            Odometry, f"/{agent_id}/ground_truth", lambda msg: self._on_ground_truth_pose(msg, agent_id), 10
        )
        self._map_get_clients[agent_id] = self.create_client(GetMap, f"/{agent_id}/get_map")
        self._map_set_clients[agent_id] = self.create_client(SetMap, f"/{agent_id}/set_map")
        self._belief_get_clients[agent_id] = self.create_client(GetMap, f"/{agent_id}/get_belief")
        self._belief_set_clients[agent_id] = self.create_client(SetMap, f"/{agent_id}/set_belief")
        self._fusion_complete_clients[agent_id] = self.create_client(Trigger, f"/{agent_id}/fusion_complete")

    def _set_up_timers(self) -> None:
        """Set up timers for the comms manager."""
        close_period = 1.0 / self.comms_config.close_range_rate
        self.create_timer(close_period, self._propagate_close_range)

        long_period = 1.0 / self.comms_config.long_range_rate
        self.create_timer(long_period, self._propagate_long_range)

        check_fusion_period = 1.0 / self.get_parameter("check_fusion_eligibility_rate").value
        self.create_timer(check_fusion_period, self._check_fusion_eligibility)

        update_pairwise_zones_period = 1.0 / self.get_parameter("update_pairwise_zones_rate").value
        self.create_timer(update_pairwise_zones_period, self._update_pairwise_zones)

    # -------------------------------------------------------------------------
    # Ground Truth Tracking
    # -------------------------------------------------------------------------

    def _on_ground_truth_pose(self, msg: Odometry, agent_id: str) -> None:
        """
        Handle ground truth pose message from Stage simulator.

        Updates agent_poses dict for use in distance calculations and LOS checks.
        The comms manager uses ground truth rather than relying on heartbeat
        messages to ensure accurate zone computation independent of communication.
        """
        self.agent_positions[agent_id] = msg.pose.pose.position

    def _on_known_map_update(self, msg: OccupancyGrid) -> None:
        """Handle known map update message from Stage simulator."""
        self.known_map = np.array(msg.data, dtype=np.uint8).reshape(msg.info.height, msg.info.width)
        self.known_map_resolution = msg.info.resolution
        self.known_map_origin = msg.info.origin.position

    # -------------------------------------------------------------------------
    # Message Handling
    # -------------------------------------------------------------------------

    def _on_message(self, msg: AgentMessage) -> None:
        """
        Handle outgoing message from an agent. Updates message_buffer respecting overwrite_targeted flag.

        If message is targeted (recipient_id != ""):
            - Simply store/overwrite at message_buffer[sender][recipient]

        If message is broadcast (recipient_id == ""):
            - Store at message_buffer[sender][""]
            - If overwrite_targeted is True: clear all targeted messages for this sender
            - If overwrite_targeted is False: preserve existing targeted messages
        """
        sender_id: str = msg.sender_id
        recipient_id: str = msg.recipient_id
        overwrite_targeted: bool = msg.overwrite_targeted

        if recipient_id != "":
            self.message_buffer[sender_id][recipient_id] = msg
        else:
            self.message_buffer[sender_id][""] = msg

            if overwrite_targeted:
                self.message_buffer[sender_id].clear()
            else:
                self.message_buffer[sender_id][""] = msg

    def _propagate_close_range(self) -> None:
        """
        Timer callback (close-range rate).

        For each sender, determine what to send to each recipient in CLOSE_RANGE zone:
        - All message types are delivered (heartbeat, coordination)
        - If targeted message exists for recipient: send targeted
        - Else if broadcast exists: send broadcast
        """
        for sender_id, messages in self.message_buffer.items():
            broadcast_recipients: set[str] = self.agent_ids - {sender_id}
            for recipient_id, msg in messages.items():
                if recipient_id == "":
                    broadcast_msg = msg
                    continue
                else:
                    broadcast_recipients.remove(recipient_id)

                if self._get_pairwise_zone(sender_id, recipient_id) != CommsZone.CLOSE_RANGE:
                    continue

                self._publish_to_agent(recipient_id, msg)

            for recipient_id in broadcast_recipients:
                if self._get_pairwise_zone(sender_id, recipient_id) != CommsZone.CLOSE_RANGE:
                    continue

                self._publish_to_agent(recipient_id, broadcast_msg)

    def _propagate_long_range(self) -> None:
        """
        Timer callback (long-range rate).

        For each sender, determine what to send to each recipient in LONG_RANGE zone.

        Filtering logic:
        - Skip recipients in CLOSE_RANGE (close-range handler covers them)
        - Skip recipients in BLACKOUT
        - For LONG_RANGE recipients: only deliver if msg_type is HEARTBEAT
        - Coordination messages are NOT delivered to long-range recipients

        Note: If the sender's current message is a coordination message, long-range
        recipients receive nothing (the previous heartbeat is not preserved).
        """
        for sender_id, messages in self.message_buffer.items():
            broadcast_recipients: set[str] = self.agent_ids - {sender_id}
            for recipient_id, msg in messages.items():
                if recipient_id == "":
                    broadcast_msg = msg
                    continue
                else:
                    broadcast_recipients.remove(recipient_id)

                if self._get_pairwise_zone(sender_id, recipient_id) != CommsZone.LONG_RANGE:
                    continue

                if self._should_deliver(sender_id, recipient_id, msg):
                    self._publish_to_agent(recipient_id, msg)

            for recipient_id in broadcast_recipients:
                if self._get_pairwise_zone(sender_id, recipient_id) != CommsZone.LONG_RANGE:
                    continue

                if self._should_deliver(sender_id, recipient_id, broadcast_msg):
                    self._publish_to_agent(recipient_id, broadcast_msg)

    def _publish_to_agent(self, recipient_id: str, msg: AgentMessage) -> None:
        """Publish message to specific agent's output topic."""
        self._message_publishers[recipient_id].publish(msg)

    def _should_deliver(self, sender: str, recipient: str, msg: AgentMessage) -> bool:
        """
        Check if message from sender should be delivered to recipient.

        Returns True if:
        - Pairwise zone is CLOSE_RANGE, OR
        - Pairwise zone is LONG_RANGE AND msg_type is HEARTBEAT

        Returns False if:
        - Pairwise zone is BLACKOUT, OR
        - Pairwise zone is LONG_RANGE AND msg_type is COORDINATION
        """
        return self._get_pairwise_zone(sender, recipient) == CommsZone.CLOSE_RANGE or (
            self._get_pairwise_zone(sender, recipient) == CommsZone.LONG_RANGE
            and msg.msg_type == AgentMessage.HEARTBEAT
        )

    def _get_pairwise_zone(self, agent_a: str, agent_b: str) -> CommsZone:
        """Look up cached pairwise zone."""
        return self.pairwise_zones[self._get_pair_key(agent_a, agent_b)]

    # -------------------------------------------------------------------------
    # Communication Zone Management
    # -------------------------------------------------------------------------
    def _update_pairwise_zones(self) -> None:
        """
        Recompute communication zones for all agent pairs.

        Called periodically or when poses update.
        Uses distance and line-of-sight checks.
        """
        for agent_a, agent_b in combinations(self.agent_ids, 2):
            zone = self._compute_zone(agent_a, agent_b)
            self.pairwise_zones[self._get_pair_key(agent_a, agent_b)] = zone

    def _compute_zone(self, agent_a: str, agent_b: str) -> CommsZone:
        """
        Determine comms zone between two agents.

        1. Compute distance from poses
        2. If distance > long_range_threshold: BLACKOUT
        3. If distance > close_range_threshold: LONG_RANGE
        4. Otherwise: CLOSE_RANGE

        Note: Line-of-sight is only checked for fusion eligibility, not communication.
        """
        distance = self._get_distance(agent_a, agent_b)
        if distance > self.comms_config.long_range_threshold:
            return CommsZone.BLACKOUT
        elif distance > self.comms_config.close_range_threshold:
            return CommsZone.LONG_RANGE
        else:
            return CommsZone.CLOSE_RANGE

    def _has_line_of_sight(self, agent_a: str, agent_b: str) -> bool:
        """Ray-cast between two poses to check for obstacles."""
        agent_a_indices = self._agent_map_indices(agent_a)
        agent_b_indices = self._agent_map_indices(agent_b)
        known_map = self.known_map
        if agent_a_indices is None or agent_b_indices is None or known_map is None:
            return False

        r0, c0 = skimage_line(agent_a_indices[0], agent_a_indices[1], agent_b_indices[0], agent_b_indices[1])
        obstacle_in_los = np.any(known_map[r0, c0] == 100)  # TODO: Magic number
        return not obstacle_in_los

    # -------------------------------------------------------------------------
    # Map Fusion (Unknown Map Mode Only)
    # -------------------------------------------------------------------------

    def _check_fusion_eligibility(self) -> None:
        """
        Timer callback. Initiates fusion for agent pairs within fusion range, with line-of-sight and cooldown elapsed.

        When use_known_map is True, map fusion is skipped and only belief fusion is performed.
        """
        for agent_a, agent_b in combinations(self.agent_ids, 2):
            if self._should_fuse(agent_a, agent_b):
                self._perform_fusion(agent_a, agent_b)

    def _should_fuse(self, agent_a: str, agent_b: str) -> bool:
        """
        Check if pair is eligible for fusion.

        Returns True if all of the following hold:
        - Distance is within fusion_range_threshold
        - Agents have line-of-sight
        - Cooldown period has elapsed since last fusion
        """
        distance_check = self._get_distance(agent_a, agent_b) <= self.comms_config.fusion_range_threshold
        line_of_sight_check = self._has_line_of_sight(agent_a, agent_b)
        cooldown_check = (
            self.last_fusion_time[self._get_pair_key(agent_a, agent_b)]
            + Duration(seconds=self.comms_config.fusion_cooldown)
            < self.get_clock().now()
        )
        return distance_check and line_of_sight_check and cooldown_check

    def _perform_fusion(self, agent_a: str, agent_b: str) -> None:
        """
        Execute belief (and optionally map) fusion for an agent pair.

        When use_known_map is False (unknown map mode):
        1. Call get_map and get_belief on both agents
        2. Fuse environment maps (cell-wise max or Bayesian update)
        3. Fuse belief/coverage grids (cell-wise max or Bayesian update)
        4. Call set_map and set_belief on both agents
        5. Push fused map to /{agent_id}/map_server/load_map on both agents
           so Nav2's global costmap static layer reflects the updated map
        6. Call fusion_complete on both agents to trigger hooks
        7. Update last_fusion_time for pair

        When use_known_map is True (known map mode), only belief fusion is performed.

        Note: slam_toolbox's pose graph is not modified. The fused map improves
        Nav2's global planning without affecting localization.
        """
        # TODO: Nav2 load map
        fuse_maps = not self.use_known_map

        get_futures: list = []
        if fuse_maps:
            map_a_future = self._map_get_clients[agent_a].call_async(GetMap.Request())
            map_b_future = self._map_get_clients[agent_b].call_async(GetMap.Request())
            get_futures += [map_a_future, map_b_future]
        belief_a_future = self._belief_get_clients[agent_a].call_async(GetMap.Request())
        belief_b_future = self._belief_get_clients[agent_b].call_async(GetMap.Request())
        get_futures += [belief_a_future, belief_b_future]

        while not all(f.done() for f in get_futures):
            self.get_logger().info("Waiting for get requests to complete for fusion...")
            rclpy.spin_once(self)

        if any(f.result() is None for f in get_futures):
            self.get_logger().error("Failed to get map or belief messages")
            return

        belief_a: OccupancyGrid = belief_a_future.result().map  # type: ignore[union-attr]
        belief_b: OccupancyGrid = belief_b_future.result().map  # type: ignore[union-attr]
        fused_belief = self._fuse_beliefs(belief_a, belief_b)

        fused_map = None
        if fuse_maps:
            map_a: OccupancyGrid = map_a_future.result().map  # type: ignore[union-attr]
            map_b: OccupancyGrid = map_b_future.result().map  # type: ignore[union-attr]
            fused_map = self._fuse_maps(map_a, map_b)

        set_futures: list = []
        if fuse_maps:
            set_futures.append(self._map_set_clients[agent_a].call_async(SetMap.Request(map=fused_map)))
            set_futures.append(self._map_set_clients[agent_b].call_async(SetMap.Request(map=fused_map)))
        set_futures.append(self._belief_set_clients[agent_a].call_async(SetMap.Request(map=fused_belief)))
        set_futures.append(self._belief_set_clients[agent_b].call_async(SetMap.Request(map=fused_belief)))

        while not all(f.done() for f in set_futures):
            self.get_logger().info("Waiting for set requests to complete for fusion...")
            rclpy.spin_once(self)

        self._fusion_complete_clients[agent_a].call_async(Trigger.Request())
        self._fusion_complete_clients[agent_b].call_async(Trigger.Request())
        self.last_fusion_time[self._get_pair_key(agent_a, agent_b)] = self.get_clock().now()

    def _fuse_maps(self, map_a: OccupancyGrid, map_b: OccupancyGrid) -> OccupancyGrid:
        """
        Combine two environment occupancy grids using Bayesian log-odds fusion.

        Data format: int8 values in [0, 100] (probability * 100), with -1 for unseen cells.
        Maps may have different origins and sizes; the output covers the union bounding box.
        Cells where neither map has information are output as -1 (unseen).
        """
        res = map_a.info.resolution

        ox_a = map_a.info.origin.position.x
        oy_a = map_a.info.origin.position.y
        ox_b = map_b.info.origin.position.x
        oy_b = map_b.info.origin.position.y

        out_ox = min(ox_a, ox_b)
        out_oy = min(oy_a, oy_b)
        out_x_max = max(ox_a + map_a.info.width * res, ox_b + map_b.info.width * res)
        out_y_max = max(oy_a + map_a.info.height * res, oy_b + map_b.info.height * res)

        out_h = math.ceil((out_y_max - out_oy) / res)
        out_w = math.ceil((out_x_max - out_ox) / res)

        row_a = round((oy_a - out_oy) / res)
        col_a = round((ox_a - out_ox) / res)
        row_b = round((oy_b - out_oy) / res)
        col_b = round((ox_b - out_ox) / res)

        raw_a = np.array(map_a.data, dtype=np.int8).reshape(map_a.info.height, map_a.info.width)
        raw_b = np.array(map_b.data, dtype=np.int8).reshape(map_b.info.height, map_b.info.width)

        known_a = raw_a != -1
        known_b = raw_b != -1

        log_a = self._to_log_odds(raw_a, known_a)
        log_b = self._to_log_odds(raw_b, known_b)

        # Initialize to 0.0 (unknown prior). Unseen cells contribute 0 log-odds
        # so only cells with real observations affect the fused result.
        log_fused = np.zeros((out_h, out_w), dtype=np.float64)
        log_fused[row_a : row_a + map_a.info.height, col_a : col_a + map_a.info.width] += log_a
        log_fused[row_b : row_b + map_b.info.height, col_b : col_b + map_b.info.width] += log_b

        has_info = np.zeros((out_h, out_w), dtype=bool)
        has_info[row_a : row_a + map_a.info.height, col_a : col_a + map_a.info.width] |= known_a
        has_info[row_b : row_b + map_b.info.height, col_b : col_b + map_b.info.width] |= known_b

        p_fused = 1.0 / (1.0 + np.exp(-log_fused))
        val_fused = np.clip(np.round(p_fused * 100.0), 0, 100).astype(np.int8)  # TODO: Magic number (also in docstring)
        output = np.where(has_info, val_fused, np.int8(-1))

        fused_map = OccupancyGrid()
        fused_map.header = map_a.header
        fused_map.info = map_a.info
        fused_map.info.width = out_w
        fused_map.info.height = out_h
        fused_map.info.origin.position.x = out_ox
        fused_map.info.origin.position.y = out_oy
        fused_map.data = output.flatten().tolist()
        return fused_map

    def _fuse_beliefs(self, belief_a: OccupancyGrid, belief_b: OccupancyGrid) -> OccupancyGrid:
        """
        Combine belief grids (search coverage) via Bayesian log-odds fusion.

        Data format: int8 values already in log-odds form, range [-127, 127].
        Fusion is addition in log-odds space, clamped back to [-127, 127].
        Cells not covered by either map stay at 0 (neutral/unknown prior).
        """
        res = belief_a.info.resolution

        ox_a = belief_a.info.origin.position.x
        oy_a = belief_a.info.origin.position.y
        ox_b = belief_b.info.origin.position.x
        oy_b = belief_b.info.origin.position.y

        out_ox = min(ox_a, ox_b)
        out_oy = min(oy_a, oy_b)
        out_x_max = max(ox_a + belief_a.info.width * res, ox_b + belief_b.info.width * res)
        out_y_max = max(oy_a + belief_a.info.height * res, oy_b + belief_b.info.height * res)

        out_h = math.ceil((out_y_max - out_oy) / res)
        out_w = math.ceil((out_x_max - out_ox) / res)

        row_a = round((oy_a - out_oy) / res)
        col_a = round((ox_a - out_ox) / res)
        row_b = round((oy_b - out_oy) / res)
        col_b = round((ox_b - out_ox) / res)

        # Use int16 as intermediate to avoid overflow before clamping (e.g. 127 + 127 = 254)
        data_a = np.array(belief_a.data, dtype=np.int16).reshape(belief_a.info.height, belief_a.info.width)
        data_b = np.array(belief_b.data, dtype=np.int16).reshape(belief_b.info.height, belief_b.info.width)

        fused = np.zeros((out_h, out_w), dtype=np.int16)
        fused[row_a : row_a + belief_a.info.height, col_a : col_a + belief_a.info.width] += data_a
        fused[row_b : row_b + belief_b.info.height, col_b : col_b + belief_b.info.width] += data_b

        output = np.clip(fused, -127, 127).astype(np.int8)  # TODO: Magic number (also in docstring)

        fused_map = OccupancyGrid()
        fused_map.header = belief_a.header
        fused_map.info = belief_a.info
        fused_map.info.width = out_w
        fused_map.info.height = out_h
        fused_map.info.origin.position.x = out_ox
        fused_map.info.origin.position.y = out_oy
        fused_map.data = output.flatten().tolist()
        return fused_map

    # -------------------------------------------------------------------------
    # Utility
    # -------------------------------------------------------------------------

    def _get_pair_key(self, agent_a: str, agent_b: str) -> tuple[str, str]:
        """Return sorted tuple for consistent dictionary keys."""
        return tuple(sorted([agent_a, agent_b]))  # type: ignore[return-value]

    def _get_distance(self, agent_a: str, agent_b: str) -> float:
        """Get distance between two agents."""
        agent_a_pos = self.agent_positions[agent_a]
        agent_b_pos = self.agent_positions[agent_b]
        if agent_a_pos is None or agent_b_pos is None:
            return float("inf")

        return math.sqrt((agent_a_pos.x - agent_b_pos.x) ** 2 + (agent_a_pos.y - agent_b_pos.y) ** 2)

    def _agent_map_indices(self, agent_id: str) -> tuple[int, int] | None:
        """Convert coordinates to indices."""
        agent_pos = self.agent_positions[agent_id]
        map_origin = self.known_map_origin
        map_resolution = self.known_map_resolution
        if agent_pos is None or map_origin is None or map_resolution is None:
            return None

        return (
            int((agent_pos.y - map_origin.y) / map_resolution),
            int((agent_pos.x - map_origin.x) / map_resolution),
        )

    def _to_log_odds(self, raw: npt.NDArray[np.int8], known: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
        eps = 1e-6  # TODO: Magic number
        p = np.where(known, np.clip(raw.astype(np.float64) / 100.0, eps, 1.0 - eps), 0.5)  # TODO: Magic number
        return np.where(known, np.log(p / (1.0 - p)), 0.0)


def main(args: list[str] | None = None) -> None:
    """Entry point for the comms manager node."""
    rclpy.init(args=args)
    # agent_ids and config would be loaded from parameters or launch arguments
    pass


if __name__ == "__main__":
    main()
