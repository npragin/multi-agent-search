"""
Target Detector node for the multi-agent search system.

Subscribes to all robots' laser scans and detects when targets are found,
notifying the discovering agent via a service call.
"""

from __future__ import annotations

import ast
import math

import numpy as np
import numpy.typing as npt
from skimage.draw import line as skimage_line

import rclpy
import tf2_geometry_msgs  # noqa: F401 — registers PointStamped with tf2
from geometry_msgs.msg import Point, PointStamped, Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from rclpy.client import Client
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.subscription import Subscription
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

from multi_agent_search_interfaces.srv import TargetDetected


class TargetDetector(LifecycleNode):
    """
    Lifecycle node that monitors all robots' laser scans for target detection.

    Subscribes to each robot's laser scan and ground truth odometry, plus the
    ground truth map. When a robot's lidar scan reveals a target location,
    calls that agent's target_detected service and tracks global found-target state.
    """

    def __init__(self) -> None:
        """Initialize the Target Detector and declare parameters."""
        super().__init__("target_detector")
        self.declare_parameter("agent_ids", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("target_positions", "[]")
        self.declare_parameter("target_radius", 1.0)
        self.declare_parameter("scan_topic", "base_scan")
        self.declare_parameter("use_known_map", False)

        self._target_positions: list[tuple[float, float]] = []
        self._target_radius: float = 1.0
        self._found_targets: set[int] = set()
        self._agent_ids: list[str] = []
        self._scan_topic: str = "base_scan"
        self._use_known_map: bool = False
        self._poses: dict[str, Pose | None] = {}
        self._map_info: MapMetaData | None = None
        self._map_frame_id: str = "map"

        self._managed_subscriptions: list[Subscription[LaserScan | Odometry | OccupancyGrid]] = []
        self._managed_service_clients: list[Client[TargetDetected.Request, TargetDetected.Response]] = []
        self._target_detected_clients: dict[str, Client[TargetDetected.Request, TargetDetected.Response]] = {}

        self._tf_buffer: Buffer | None = None
        self._tf_listener: TransformListener | None = None

    # -------------------------------------------------------------------------
    # Lifecycle Callbacks
    # -------------------------------------------------------------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure: read params, create subscriptions and service clients."""
        self._agent_ids = self.get_parameter("agent_ids").value
        raw = self.get_parameter("target_positions").value
        parsed = ast.literal_eval(raw)
        self._target_positions = [(float(p[0]), float(p[1])) for p in parsed]
        self._target_radius = self.get_parameter("target_radius").value
        self._scan_topic = self.get_parameter("scan_topic").value
        self._use_known_map = self.get_parameter("use_known_map").value
        self._found_targets = set()

        self._poses = dict.fromkeys(self._agent_ids)
        self._map_info = None
        self._map_frame_id = "map"

        if not self._use_known_map:
            self._tf_buffer = Buffer()
            self._tf_listener = TransformListener(self._tf_buffer, self)

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        sub_map = self.create_subscription(OccupancyGrid, "/ground_truth_map", self._on_map, latched_qos)
        self._managed_subscriptions.append(sub_map)

        for agent_id in self._agent_ids:
            sub_scan = self.create_subscription(
                LaserScan,
                f"/{agent_id}/{self._scan_topic}",
                lambda msg, aid=agent_id: self._on_scan(aid, msg),
                10,
            )
            sub_odom = self.create_subscription(
                Odometry,
                f"/{agent_id}/ground_truth",
                lambda msg, aid=agent_id: self._on_odom(aid, msg),
                10,
            )
            self._managed_subscriptions.extend([sub_scan, sub_odom])

            client: Client[TargetDetected.Request, TargetDetected.Response] = self.create_client(
                TargetDetected, f"/{agent_id}/target_detected"
            )
            self._target_detected_clients[agent_id] = client
            self._managed_service_clients.append(client)

        self.get_logger().info("Configured")
        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate the target detector."""
        self.get_logger().info("Activated")
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivate the target detector."""
        self.get_logger().info("Deactivated")
        return super().on_deactivate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Cleanup: destroy all interfaces."""
        self._destroy_interfaces()
        self.get_logger().info("Cleaned up")
        return super().on_cleanup(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Shutdown: destroy all interfaces."""
        self._destroy_interfaces()
        self.get_logger().info("Shut down")
        return super().on_shutdown(state)

    def _destroy_interfaces(self) -> None:
        """Destroy all ROS2 interfaces created during configure."""
        for sub in self._managed_subscriptions:
            self.destroy_subscription(sub)
        for client in self._managed_service_clients:
            self.destroy_client(client)
        self._managed_subscriptions.clear()
        self._managed_service_clients.clear()
        self._target_detected_clients.clear()
        self._tf_listener = None
        self._tf_buffer = None

    # -------------------------------------------------------------------------
    # Subscription Callbacks
    # -------------------------------------------------------------------------

    def _on_odom(self, agent_id: str, msg: Odometry) -> None:
        """Cache the latest ground truth pose for a robot."""
        self._poses[agent_id] = msg.pose.pose

    def _on_map(self, msg: OccupancyGrid) -> None:
        """Cache the ground truth map metadata and frame."""
        self._map_info = msg.info
        self._map_frame_id = msg.header.frame_id

    def _on_scan(self, agent_id: str, scan: LaserScan) -> None:
        """Handle a lidar scan: trace rays and check for targets."""
        pose = self._poses.get(agent_id)
        if pose is None or self._map_info is None:
            return

        if not self._target_positions:
            self.get_logger().warn("Target positions are not set, skipping target detection", once=True)
            return

        if len(self._found_targets) >= len(self._target_positions):
            self.get_logger().info("All targets have been found!")
            return

        all_rr, all_cc = self._trace_scan_rays(scan, pose)
        if len(all_rr) == 0:
            return
        self._check_for_targets(agent_id, all_rr, all_cc)

    # -------------------------------------------------------------------------
    # Ray Tracing
    # -------------------------------------------------------------------------

    def _trace_scan_rays(
        self,
        scan: LaserScan,
        pose: Pose,
    ) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp]]:
        """
        Trace all lidar rays using Bresenham's algorithm.

        Return the concatenated (row, col) arrays for every cell touched by
        every finite-range beam in the scan.
        """
        if self._map_info is None:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

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

        if len(valid_ranges) == 0:
            return np.array([], dtype=np.intp), np.array([], dtype=np.intp)

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

        rows = self._map_info.height
        cols = self._map_info.width
        mask = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        return rr[mask], cc[mask]

    # -------------------------------------------------------------------------
    # Target Detection
    # -------------------------------------------------------------------------

    def _check_for_targets(
        self,
        agent_id: str,
        all_rr: npt.NDArray[np.intp],
        all_cc: npt.NDArray[np.intp],
    ) -> None:
        """
        Check if any traced ray cell is within target_radius of an unfound target.

        When new targets are found, updates internal state, logs the discovery,
        and calls the detecting agent's target_detected service.
        """
        if self._map_info is None:
            return

        unfound = [(i, pos) for i, pos in enumerate(self._target_positions) if i not in self._found_targets]
        if not unfound:
            return

        res = self._map_info.resolution
        ox = self._map_info.origin.position.x
        oy = self._map_info.origin.position.y

        world_x = all_cc.astype(np.float64) * res + ox + res / 2.0
        world_y = all_rr.astype(np.float64) * res + oy + res / 2.0

        newly_found: list[tuple[int, tuple[float, float]]] = []
        radius_sq = self._target_radius**2
        for target_idx, (tx, ty) in unfound:
            dist_sq = (world_x - tx) ** 2 + (world_y - ty) ** 2
            if np.any(dist_sq <= radius_sq):
                self._found_targets.add(target_idx)
                newly_found.append((target_idx, (tx, ty)))

        if newly_found:
            for target_idx, (tx, ty) in newly_found:
                self.get_logger().info(f"Target {target_idx} found by {agent_id} at ({tx:.2f}, {ty:.2f})")

            target_points = self._build_target_points(agent_id, newly_found)
            request = TargetDetected.Request(targets=target_points)
            client = self._target_detected_clients[agent_id]
            client.call_async(request)

            if len(self._found_targets) >= len(self._target_positions):
                self.get_logger().info("All targets have been found!")

    def _build_target_points(
        self,
        agent_id: str,
        newly_found: list[tuple[int, tuple[float, float]]],
    ) -> list[Point]:
        """
        Build target Point list, transforming to the agent's map frame when using SLAM.

        In known-map mode, target coordinates are already in the shared map frame.
        In SLAM mode (use_known_map=False), targets are transformed from the ground
        truth map frame into the agent's ``{agent_id}/map`` frame via TF2.
        """
        points: list[Point] = []
        for _, (tx, ty) in newly_found:
            if self._use_known_map or self._tf_buffer is None:
                points.append(Point(x=tx, y=ty, z=0.0))
            else:
                target_frame = f"{agent_id}/map"
                pt_stamped = PointStamped()
                pt_stamped.header.frame_id = self._map_frame_id
                pt_stamped.header.stamp = self.get_clock().now().to_msg()
                pt_stamped.point = Point(x=tx, y=ty, z=0.0)
                try:
                    transformed = self._tf_buffer.transform(pt_stamped, target_frame)
                    points.append(transformed.point)
                except Exception as e:
                    self.get_logger().warn(
                        f"Could not transform target to {target_frame}, sending in {self._map_frame_id} frame: {e}"
                    )
                    points.append(Point(x=tx, y=ty, z=0.0))
        return points


def main(args: list[str] | None = None) -> None:
    """Entry point for the Target Detector."""
    rclpy.init(args=args)
    node = TargetDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
