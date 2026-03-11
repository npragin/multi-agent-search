"""AgentBase subclass implementing sequential room and hallway junction auction for known-map environments."""

from __future__ import annotations

import hashlib
import heapq
import random
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, center_of_mass, label
from scipy.signal import fftconvolve

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.lifecycle import LifecycleState, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from multi_agent_search.agent_base import AgentBase
from multi_agent_search.types import BaseCoordinationMessage, HeartbeatMessage, NavStatus


@dataclass
class AuctioneerNegotiationMessage(BaseCoordinationMessage):
    """Message broadcast during auctioneer negotiation to share agent IDs."""

    pass


@dataclass
class AuctionStartMessage(BaseCoordinationMessage):
    """Sent by the auctioneer to announce a waypoint up for bid."""

    waypoint_index: int = 0
    waypoint: tuple[float, float] = (0.0, 0.0)


@dataclass
class BidMessage(BaseCoordinationMessage):
    """Sent by agents (including the auctioneer) to bid on the current waypoint."""

    waypoint_index: int = 0
    bid_distance: float = float("inf")


@dataclass
class AwardMessage(BaseCoordinationMessage):
    """Sent by the auctioneer to the winner of a waypoint auction."""

    waypoint_index: int = 0
    waypoint: tuple[float, float] = (0.0, 0.0)


@dataclass
class AwardAckMessage(BaseCoordinationMessage):
    """Sent by the winner back to the auctioneer to confirm receipt of an award."""

    pass


@dataclass
class AuctionTerminatedMessage(BaseCoordinationMessage):
    """Sent by the auctioneer when all waypoints have been auctioned off."""

    pass


class AuctionPhase(IntEnum):
    """Phase of the auction agent."""

    AUCTIONEER_NEGOTIATION = 0
    WAITING_FOR_AUCTION = 1
    AUCTIONING = 2
    BIDDING = 3
    EXECUTING = 4


class SequentialAuctionAgent(AgentBase):
    """AgentBase subclass implementing sequential room and hallway junction auction for known-map environments."""

    def __init__(self) -> None:
        """Initialize the Sequential Auction Agent."""
        super().__init__("SequentialAuctionAgent")

        self.declare_parameter("negotiation_timeout", 5.0)
        self.declare_parameter("room_wall_length", 6.9)
        self.declare_parameter("hallway_width", 3.6)
        self.declare_parameter("obstacle_length", 0.2)
        self.declare_parameter("bid_timeout", 5.0)
        self.declare_parameter("ack_timeout", 5.0)
        self.declare_parameter("action_server_poll_rate", 2.0)

        self.negotiation_timeout: float = 0.0
        self.room_wall_length: float = 0.0
        self.hallway_width: float = 0.0
        self.obstacle_length: float = 0.0
        self.bid_timeout: float = 0.0
        self.ack_timeout: float = 0.0
        self.action_server_poll_rate: float = 0.0

        self._nav_poll_timer: rclpy.timer.Timer | None = None

        self.phase = AuctionPhase.AUCTIONEER_NEGOTIATION
        self.known_agent_ids: set[str] = set()
        self.auctioneer_id: str | None = None
        self._negotiation_start_time = self.get_clock().now()

        self.negotiation_timer: rclpy.timer.Timer | None = None
        self.auction_timer: rclpy.timer.Timer | None = None

        self._waypoints: list[tuple[float, float]] = []
        self._current_waypoint_index: int = 0
        self._bids: dict[str, float] = {}
        self._bid_start_time = self.get_clock().now()
        self._bid_submitted: bool = False
        self._award_received: bool = False
        self._awaiting_ack: bool = False
        self._ack_start_time = self.get_clock().now()

        self._my_assignments: list[tuple[float, float]] = []
        self._last_assigned_center: tuple[float, float] | None = None
        self._cumulative_distance: float = 0.0
        self._current_execution_index: int = 0

        self._marker_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._marker_pub: rclpy.publisher.Publisher[MarkerArray] | None = None

        # Separate callback groups prevent blocking BFS from starving message callbacks
        self._waypoint_detection_cbg = ReentrantCallbackGroup()
        self._bid_generation_cbg = ReentrantCallbackGroup()

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate: read parameters and start auctioneer negotiation."""
        result = super().on_activate(state)
        if result != TransitionCallbackReturn.SUCCESS:
            return result

        seed = int(hashlib.md5(self.agent_id.encode()).hexdigest(), 16)
        rng = random.Random(seed)
        self._marker_color = (rng.random(), rng.random(), rng.random())
        self._marker_pub = self.create_publisher(
            MarkerArray,
            "/auction_assignments",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE),
        )

        self.negotiation_timeout = self.get_parameter("negotiation_timeout").value
        self.room_wall_length = self.get_parameter("room_wall_length").value
        self.hallway_width = self.get_parameter("hallway_width").value
        self.obstacle_length = self.get_parameter("obstacle_length").value
        self.bid_timeout = self.get_parameter("bid_timeout").value
        self.ack_timeout = self.get_parameter("ack_timeout").value
        self.action_server_poll_rate = self.get_parameter("action_server_poll_rate").value

        self.known_agent_ids = {self.agent_id}
        self._negotiation_start_time = self.get_clock().now()
        self.negotiation_timer = self.create_timer(0.5, self._negotiation_tick)

        self.get_logger().info("Activated. Entering auctioneer negotiation phase")
        return TransitionCallbackReturn.SUCCESS

    def _negotiation_tick(self) -> None:
        """Broadcast agent_id and check if negotiation timeout has elapsed."""
        if self.phase != AuctionPhase.AUCTIONEER_NEGOTIATION:
            return

        self.publish_coordination_message(AuctioneerNegotiationMessage(sender_id=self.agent_id))

        elapsed = (self.get_clock().now() - self._negotiation_start_time).nanoseconds / 1e9
        if elapsed >= self.negotiation_timeout:
            self._elect_auctioneer()

    def _elect_auctioneer(self) -> None:
        """Elect the auctioneer as the lowest sorted agent_id."""
        self.auctioneer_id = sorted(self.known_agent_ids)[0]
        self.get_logger().info(
            f"Auctioneer elected: {self.auctioneer_id} (from candidates: {sorted(self.known_agent_ids)})"
        )

        if self.negotiation_timer is not None:
            self.negotiation_timer.cancel()
            self.negotiation_timer = None

        if self.auctioneer_id == self.agent_id:
            self._become_auctioneer()
        else:
            self.phase = AuctionPhase.WAITING_FOR_AUCTION
            self.get_logger().info(f"Waiting for auctioneer {self.auctioneer_id} to start auction")

    def _schedule_on_cbg(self, cbg: ReentrantCallbackGroup, fn: object, *args: object) -> None:
        """Run *fn(*args)* once on a one-shot timer in the given callback group."""
        executed = False

        def _once() -> None:
            nonlocal executed
            if executed:
                return
            executed = True
            timer.cancel()
            fn(*args)  # type: ignore[operator]

        timer = self.create_timer(0.0, _once, callback_group=cbg)

    def _become_auctioneer(self) -> None:
        """Detect waypoints (rooms + hallway junctions), then begin the sequential auction."""
        self.phase = AuctionPhase.AUCTIONING
        self.get_logger().info("I am the auctioneer. Detecting waypoints...")
        self._schedule_on_cbg(self._waypoint_detection_cbg, self._detect_waypoints_and_start_auction)

    def _detect_waypoints_and_start_auction(self) -> None:
        """Long-running detection of rooms and hallway junctions, then start the auction."""
        rooms, room_interior_mask = self._detect_rooms()
        hallway_junctions = self._detect_hallway_corners(room_interior_mask)

        self.get_logger().info(f"Detected {len(rooms)} rooms and {len(hallway_junctions)} hallway junctions")
        self._waypoints = rooms + hallway_junctions
        if not self._waypoints:
            self.get_logger().warn("No waypoints detected! Terminating auction immediately")
            self._terminate_auction()
            return

        self._waypoints = self._sort_waypoints_by_distance(self._waypoints)
        self.get_logger().info(f"Auctioning {len(self._waypoints)} waypoints")

        self._current_waypoint_index = 0
        self._announce_waypoint()

    def _detect_rooms(self) -> tuple[list[tuple[float, float]], NDArray[np.bool_]]:
        """Detect rooms via free-space kernel convolution. Returns room centers and interior mask."""
        if self.map is None or self.map_info is None:
            return [], np.zeros((0, 0), dtype=bool)

        resolution = self.map_info.resolution
        wall_cells = max(1, round(self.room_wall_length / resolution)) - 2

        # Erase freestanding obstacles from a working copy so rooms with obstacles are still detected
        working_map = self.map.astype(np.float32).copy()
        obs_cells = max(1, round(self.obstacle_length / resolution))
        if obs_cells > 0:
            obstacle_mask = (self.map == 100).astype(np.float32)
            obs_kernel = np.ones((obs_cells, obs_cells), dtype=np.float32)
            obs_conv = fftconvolve(obstacle_mask, obs_kernel, mode="same")
            max_val = obs_cells * obs_cells
            obs_center_mask = obs_conv >= (max_val - 0.5)
            # Ring kernel checks that cells surrounding the obstacle are all free space,
            # distinguishing freestanding obstacles from wall segments
            border = obs_cells + 4
            ring_kernel = np.ones((border, border), dtype=np.float32)
            inner = obs_cells + 2
            pad = (border - inner) // 2
            ring_kernel[pad : pad + inner, pad : pad + inner] = 0.0
            not_free = (self.map != 0).astype(np.float32)
            ring_conv = fftconvolve(not_free, ring_kernel, mode="same")
            obs_center_mask &= ring_conv < 0.5
            obs_rows, obs_cols = np.where(obs_center_mask)
            for r, c in zip(obs_rows, obs_cols, strict=True):
                r_lo = max(0, r - obs_cells // 2)
                r_hi = min(self.map.shape[0], r + obs_cells // 2 + 1)
                c_lo = max(0, c - obs_cells // 2)
                c_hi = min(self.map.shape[1], c + obs_cells // 2 + 1)
                working_map[r_lo:r_hi, c_lo:c_hi] = 0.0

        # Convolve: entirely free windows sum to 0
        kernel = np.ones((wall_cells, wall_cells), dtype=np.float32)
        conv = fftconvolve(working_map, kernel, mode="same")
        center_mask = np.abs(conv) < 0.5

        centers = self._group_candidates_to_world_centers(center_mask, suppression_size=wall_cells)

        # Dilate center_mask to cover full room footprints
        room_interior_struct = np.ones((wall_cells, wall_cells), dtype=bool)
        room_interior_mask = binary_dilation(center_mask, structure=room_interior_struct)

        return centers, room_interior_mask

    def _detect_hallway_corners(self, room_interior_mask: NDArray[np.bool_]) -> list[tuple[float, float]]:
        """Detect hallway corners and dead ends using L-shaped convolution kernels in all 4 rotations."""
        if self.map is None or self.map_info is None:
            return []

        resolution = self.map_info.resolution
        width_cells = max(3, round(self.hallway_width / resolution))
        side_len = width_cells + 1

        obstacle_mask = (self.map == 100).astype(np.float32)
        if room_interior_mask.size > 0 and room_interior_mask.shape == self.map.shape:
            union_mask = np.maximum(obstacle_mask, room_interior_mask.astype(np.float32))
        else:
            union_mask = obstacle_mask

        l_kernel = np.zeros((side_len, side_len), dtype=np.float32)
        l_kernel[0, :] = 1.0
        l_kernel[:, 0] = 1.0
        max_val = float(np.sum(l_kernel))

        free_space = self.map == 0
        rows, cols = self.map.shape

        # Cell just inside the L corner for each rotation
        elbow_positions = [
            (1, 1),
            (side_len - 2, 1),
            (side_len - 2, side_len - 2),
            (1, side_len - 2),
        ]

        combined_mask = np.zeros(self.map.shape, dtype=bool)

        for k in range(4):
            rotated = np.rot90(l_kernel, k=k)
            conv = fftconvolve(union_mask, rotated, mode="same")

            activated = conv >= (max_val - 0.5)

            half_r = rotated.shape[0] // 2
            half_c = rotated.shape[1] // 2
            elbow_r, elbow_c = elbow_positions[k]
            off_r = elbow_r - half_r
            off_c = elbow_c - half_c

            elbow_free = np.zeros(self.map.shape, dtype=bool)
            r_lo = max(0, -off_r)
            r_hi = min(rows, rows - off_r)
            c_lo = max(0, -off_c)
            c_hi = min(cols, cols - off_c)
            elbow_free[r_lo:r_hi, c_lo:c_hi] = free_space[r_lo + off_r : r_hi + off_r, c_lo + off_c : c_hi + off_c]

            combined_mask |= activated & elbow_free

        if room_interior_mask.size > 0 and room_interior_mask.shape == self.map.shape:
            combined_mask &= ~room_interior_mask

        if not np.any(combined_mask):
            return []

        all_centers = self._group_candidates_to_world_centers(combined_mask, suppression_size=width_cells)
        merged: list[tuple[float, float]] = []
        for c in all_centers:
            if not any(((c[0] - m[0]) ** 2 + (c[1] - m[1]) ** 2) ** 0.5 < self.hallway_width for m in merged):
                merged.append(c)

        return merged

    def _sort_waypoints_by_distance(self, waypoints: list[tuple[float, float]]) -> list[tuple[float, float]]:
        """Sort waypoints by BFS travel distance from the auctioneer's position."""
        if self.current_pose is None:
            return waypoints

        start = (self.current_pose.position.x, self.current_pose.position.y)
        distances: list[tuple[float, tuple[float, float]]] = []
        for wp in waypoints:
            d = self._compute_travel_distance(start, wp)
            distances.append((d, wp))
        distances.sort(key=lambda x: x[0])
        return [wp for _, wp in distances]

    def _announce_waypoint(self) -> None:
        """Announce the current waypoint for bidding."""
        if self._current_waypoint_index >= len(self._waypoints):
            self._terminate_auction()
            return

        wp = self._waypoints[self._current_waypoint_index]
        self.get_logger().info(
            f"Announcing waypoint {self._current_waypoint_index + 1}/{len(self._waypoints)} "
            f"at ({wp[0]:.2f}, {wp[1]:.2f})"
        )

        self._bids = {}

        self.publish_coordination_message(
            AuctionStartMessage(
                sender_id=self.agent_id,
                waypoint_index=self._current_waypoint_index,
                waypoint=wp,
            )
        )

        self._schedule_on_cbg(self._bid_generation_cbg, self._auctioneer_submit_bid, wp)

    def _auctioneer_submit_bid(self, wp: tuple[float, float]) -> None:
        """Compute the auctioneer's own bid, then start the bid collection timer."""
        self._submit_bid(wp)

        # Set bid start time AFTER blocking BFS to avoid sim-time advancing
        # during computation and triggering an immediate timeout
        self._bid_start_time = self.get_clock().now()

        if self.auction_timer is not None:
            self.auction_timer.cancel()
        self.auction_timer = self.create_timer(0.5, self._check_bids)

    def _submit_bid(self, waypoint: tuple[float, float]) -> None:
        """Compute cumulative travel distance bid and publish it."""
        if self._last_assigned_center is not None:
            start = self._last_assigned_center
        elif self.current_pose is not None:
            start = (self.current_pose.position.x, self.current_pose.position.y)
        else:
            return

        distance = self._compute_travel_distance(start, waypoint)
        cumulative_bid = self._cumulative_distance + distance

        if self.auctioneer_id == self.agent_id:
            self._bids[self.agent_id] = cumulative_bid
        else:
            bid = BidMessage(
                sender_id=self.agent_id,
                waypoint_index=self._current_waypoint_index,
                bid_distance=cumulative_bid,
            )
            self.publish_coordination_message(bid)
        self.get_logger().info(f"Bid {cumulative_bid:.2f} for waypoint {self._current_waypoint_index + 1}")

    def _check_bids(self) -> None:
        """Check if all bids have been received or timeout has elapsed."""
        if self.phase != AuctionPhase.AUCTIONING:
            return

        elapsed = (self.get_clock().now() - self._bid_start_time).nanoseconds / 1e9
        all_received = len(self._bids) >= len(self.known_agent_ids)

        if all_received or elapsed >= self.bid_timeout:
            if self.auction_timer is not None:
                self.auction_timer.cancel()
                self.auction_timer = None
            self._resolve_bid()

    def _resolve_bid(self) -> None:
        """Award the current waypoint to the lowest bidder."""
        if not self._bids:
            self.get_logger().warn(f"No bids for waypoint {self._current_waypoint_index}. Skipping")
            self._current_waypoint_index += 1
            self._announce_waypoint()
            return

        winner_id = min(self._bids, key=self._bids.__getitem__)
        wp = self._waypoints[self._current_waypoint_index]

        self.get_logger().info(f"Waypoint {self._current_waypoint_index} awarded to {winner_id}")

        if winner_id == self.agent_id:
            self._my_assignments.append(wp)
            self._last_assigned_center = wp
            self._cumulative_distance = self._bids[winner_id]
            self._current_waypoint_index += 1
            self._announce_waypoint()
        else:
            self.publish_coordination_message(
                AwardMessage(
                    sender_id=self.agent_id,
                    waypoint_index=self._current_waypoint_index,
                    waypoint=wp,
                ),
                recipient=winner_id,
            )
            self._awaiting_ack = True
            self._ack_start_time = self.get_clock().now()
            self._current_waypoint_index += 1
            if self.auction_timer is not None:
                self.auction_timer.cancel()
            self.auction_timer = self.create_timer(0.5, self._check_award_ack_timeout)

    def _check_award_ack_timeout(self) -> None:
        """If no ack received within ack_timeout, move on to the next waypoint."""
        if not self._awaiting_ack:
            return

        elapsed = (self.get_clock().now() - self._ack_start_time).nanoseconds / 1e9
        if elapsed >= self.ack_timeout:
            self.get_logger().warn("Award ack timed out. Moving to next waypoint")
            self._awaiting_ack = False
            if self.auction_timer is not None:
                self.auction_timer.cancel()
                self.auction_timer = None
            self._announce_waypoint()

    def _terminate_auction(self) -> None:
        """Broadcast termination and enter execution phase."""
        self.get_logger().info("Auction complete. Executing assignments")
        self.publish_coordination_message(AuctionTerminatedMessage(sender_id=self.agent_id))

        if self.auction_timer is not None:
            self.auction_timer.cancel()
            self.auction_timer = None

        self._start_execution()

    def _publish_assignment_markers(self) -> None:
        """Publish cylinder + text markers for each assigned waypoint."""
        marker_array = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        r, g, b = self._marker_color

        for i, (wx, wy) in enumerate(self._my_assignments):
            cylinder = Marker()
            cylinder.header.frame_id = "map"
            cylinder.header.stamp = stamp
            cylinder.ns = f"{self.agent_id}_assignments"
            cylinder.id = i * 2
            cylinder.type = Marker.CYLINDER
            cylinder.action = Marker.ADD
            cylinder.pose.position.x = wx
            cylinder.pose.position.y = wy
            cylinder.pose.position.z = 0.0
            cylinder.pose.orientation.w = 1.0
            cylinder.scale.x = 1.0
            cylinder.scale.y = 1.0
            cylinder.scale.z = 0.05
            cylinder.color.r = r
            cylinder.color.g = g
            cylinder.color.b = b
            cylinder.color.a = 0.8
            marker_array.markers.append(cylinder)

            label = Marker()
            label.header.frame_id = "map"
            label.header.stamp = stamp
            label.ns = f"{self.agent_id}_assignments"
            label.id = i * 2 + 1
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = wx
            label.pose.position.y = wy
            label.pose.position.z = 0.5
            label.pose.orientation.w = 1.0
            label.scale.z = 0.4
            label.color.r = r
            label.color.g = g
            label.color.b = b
            label.color.a = 1.0
            label.text = f"{self.agent_id} #{i + 1}"
            marker_array.markers.append(label)

        if self._marker_pub is not None:
            self._marker_pub.publish(marker_array)

    def _start_execution(self) -> None:
        """Begin sequential navigation to assigned waypoints."""
        self.phase = AuctionPhase.EXECUTING
        self._current_execution_index = 0
        self._publish_assignment_markers()
        self.get_logger().info(f"Executing {len(self._my_assignments)} assignments: {self._my_assignments}")
        self._navigate_to_next_waypoint()

    def _navigate_to_next_waypoint(self) -> None:
        if self._current_execution_index >= len(self._my_assignments):
            self.get_logger().info("All assigned waypoints visited")
            return

        if self.nav_status == NavStatus.NAVIGATING:
            return

        if not self._nav_client.server_is_ready():
            self.get_logger().info("Action server not ready, waiting...")
            if self._nav_poll_timer is None:
                period = 1.0 / self.action_server_poll_rate if self.action_server_poll_rate > 0 else 0.5
                self._nav_poll_timer = self.create_timer(period, self._poll_action_server)
            return

        self._send_current_waypoint_goal()

    def _poll_action_server(self) -> None:
        """Check if the Nav2 action server is ready and send the goal once available."""
        if self._nav_client.server_is_ready():
            if self._nav_poll_timer is not None:
                self._nav_poll_timer.cancel()
                self._nav_poll_timer = None
            self._send_current_waypoint_goal()

    def _send_current_waypoint_goal(self) -> None:
        """Send the navigation goal for the current execution waypoint."""
        if self._current_execution_index >= len(self._my_assignments):
            return

        wp = self._my_assignments[self._current_execution_index]
        self.get_logger().info(
            f"Navigating to waypoint {self._current_execution_index + 1}/{len(self._my_assignments)} "
            f"at ({wp[0]:.2f}, {wp[1]:.2f})"
        )
        goal = self._pose_from_xy(wp[0], wp[1])
        self.navigate_to(goal)

    def on_navigation_succeeded(self) -> None:
        """Move to the next assigned waypoint when navigation succeeds."""
        if self.phase == AuctionPhase.EXECUTING:
            self._current_execution_index += 1
            self._navigate_to_next_waypoint()

    def on_navigation_failed(self, reason: str) -> None:
        """Skip the current waypoint on navigation failure."""
        if self.phase == AuctionPhase.EXECUTING:
            self.get_logger().warn(f"Navigation failed: {reason}. Skipping to next waypoint")
            self._current_execution_index += 1
            self._navigate_to_next_waypoint()

    def on_heartbeat(self, msg: HeartbeatMessage) -> None:
        """Handle a heartbeat (unused in auction agent)."""
        pass

    def on_coordination(self, msg: BaseCoordinationMessage) -> None:
        """Handle coordination messages for negotiation, bidding, and awards."""
        if isinstance(msg, AuctioneerNegotiationMessage) and self.phase == AuctionPhase.AUCTIONEER_NEGOTIATION:
            self.known_agent_ids.add(msg.sender_id)

        elif isinstance(msg, AuctionStartMessage):
            new_round = msg.waypoint_index != self._current_waypoint_index
            if self.phase in (AuctionPhase.WAITING_FOR_AUCTION, AuctionPhase.BIDDING) and (
                new_round or not self._bid_submitted
            ):
                self.phase = AuctionPhase.BIDDING
                self._award_received = False
                self._bid_submitted = True
                self._current_waypoint_index = msg.waypoint_index
                self._schedule_on_cbg(self._bid_generation_cbg, self._submit_bid, msg.waypoint)

        elif isinstance(msg, BidMessage) and self.phase == AuctionPhase.AUCTIONING:
            if msg.waypoint_index == self._current_waypoint_index:
                self._bids[msg.sender_id] = msg.bid_distance
                if len(self._bids) >= len(self.known_agent_ids):
                    if self.auction_timer is not None:
                        self.auction_timer.cancel()
                        self.auction_timer = None
                    self._resolve_bid()

        elif isinstance(msg, AwardMessage) and not self._award_received:
            self._award_received = True
            self._bid_submitted = False
            start = (
                self._last_assigned_center
                if self._last_assigned_center is not None
                else (
                    (self.current_pose.position.x, self.current_pose.position.y)
                    if self.current_pose is not None
                    else msg.waypoint
                )
            )
            self._cumulative_distance += self._compute_travel_distance(start, msg.waypoint)
            self._my_assignments.append(msg.waypoint)
            self._last_assigned_center = msg.waypoint
            self.get_logger().info(
                f"Won waypoint {msg.waypoint_index + 1} at ({msg.waypoint[0]:.2f}, {msg.waypoint[1]:.2f})"
            )
            self.publish_coordination_message(
                AwardAckMessage(sender_id=self.agent_id),
                recipient=self.auctioneer_id or "",
            )

        elif isinstance(msg, AwardAckMessage) and self.phase == AuctionPhase.AUCTIONING and self._awaiting_ack:
            self.get_logger().info(f"Received award ack from {msg.sender_id}")
            self._awaiting_ack = False
            if self.auction_timer is not None:
                self.auction_timer.cancel()
                self.auction_timer = None
            self._announce_waypoint()

        elif isinstance(msg, AuctionTerminatedMessage) and self.phase != AuctionPhase.EXECUTING:
            self.get_logger().info("Auction terminated by auctioneer")
            self._start_execution()

    def on_lidar_scan(self, scan: LaserScan) -> None:
        """Handle a lidar scan."""
        pass

    def on_target_detected(self, target_locations: list[tuple[float, float]]) -> None:
        """Handle a target detection."""
        pass

    def _pose_from_xy(self, x: float, y: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def _group_candidates_to_world_centers(
        self, mask: NDArray[np.bool_], suppression_size: int
    ) -> list[tuple[float, float]]:
        """Group adjacent candidates via dilation and return world-coordinate centers."""
        suppression_struct = np.ones((suppression_size, suppression_size), dtype=bool)
        dilated = binary_dilation(mask, structure=suppression_struct)
        labeled_arr, num_features = label(dilated)

        uniform = mask.astype(np.float32)
        centroids = center_of_mass(uniform, labels=labeled_arr, index=np.arange(1, num_features + 1))
        centers: list[tuple[float, float]] = []
        for i, centroid in enumerate(centroids):
            cr, cc = centroid
            comp_label = labeled_arr[round(cr), round(cc)]
            if comp_label == 0:
                comp_label = i + 1
            candidate_rows, candidate_cols = np.where(mask & (labeled_arr == comp_label))
            if len(candidate_rows) == 0:
                continue
            dists = (candidate_rows - cr) ** 2 + (candidate_cols - cc) ** 2
            best_idx = int(np.argmin(dists))
            centers.append(self._cell_to_world((candidate_rows[best_idx], candidate_cols[best_idx])))
        return centers

    def _cell_to_world(self, cell: tuple[int, int]) -> tuple[float, float]:
        if self.map_info is None:
            raise ValueError("Map info is not set")
        row, col = cell
        world_x = col * self.map_info.resolution + self.map_info.origin.position.x
        world_y = row * self.map_info.resolution + self.map_info.origin.position.y
        return world_x, world_y

    def _world_to_cell(self, world: tuple[float, float]) -> tuple[int, int]:
        if self.map_info is None:
            raise ValueError("Map info is not set")
        world_x, world_y = world
        col = int((world_x - self.map_info.origin.position.x) / self.map_info.resolution)
        row = int((world_y - self.map_info.origin.position.y) / self.map_info.resolution)
        return row, col

    def _compute_travel_distance(self, start: tuple[float, float], goal: tuple[float, float]) -> float:
        """A* travel distance on the occupancy grid; falls back to Euclidean if no path."""  # noqa: D401
        if self.map is None or self.map_info is None:
            return float(((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2) ** 0.5)

        start_cell = self._world_to_cell(start)
        goal_cell = self._world_to_cell(goal)

        rows, cols = self.map.shape
        start_cell = (max(0, min(rows - 1, start_cell[0])), max(0, min(cols - 1, start_cell[1])))
        goal_cell = (max(0, min(rows - 1, goal_cell[0])), max(0, min(cols - 1, goal_cell[1])))

        gr, gc = goal_cell

        res = self.map_info.resolution
        # Priority queue: (f_cost, g_cost, row, col)
        h0 = abs(start_cell[0] - gr) + abs(start_cell[1] - gc)
        open_heap: list[tuple[int, int, int, int]] = [(h0, 0, start_cell[0], start_cell[1])]
        g_cost = np.full((rows, cols), np.iinfo(np.int32).max, dtype=np.int32)
        g_cost[start_cell[0], start_cell[1]] = 0

        while open_heap:
            _, g, r, c = heapq.heappop(open_heap)
            if (r, c) == goal_cell:
                return float(g * res)

            if g > g_cost[r, c]:
                continue

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and self.map[nr, nc] != 100:
                    ng = g + 1
                    if ng < g_cost[nr, nc]:
                        g_cost[nr, nc] = ng
                        h = abs(nr - gr) + abs(nc - gc)
                        heapq.heappush(open_heap, (ng + h, ng, nr, nc))

        return float(((start[0] - goal[0]) ** 2 + (start[1] - goal[1]) ** 2) ** 0.5)


def main(args: list[str] | None = None) -> None:
    """Entry point for the Sequential Auction Agent."""
    rclpy.init(args=args)
    agent = SequentialAuctionAgent()
    executor = MultiThreadedExecutor()
    executor.add_node(agent)
    executor.spin()
    agent.destroy_node()
    rclpy.shutdown()
