"""
Metrics Monitor node for the multi-agent search system.

Tracks and records search performance metrics across experiments.
Metrics are saved to a CSV file that persists across runs.
"""

import csv
import json
import math
import os

import numpy as np
from numpy.typing import NDArray
from skimage.draw import line as skimage_line

import rclpy
from geometry_msgs.msg import Pose
from nav_msgs.msg import MapMetaData, OccupancyGrid, Odometry
from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.subscription import Subscription
from rclpy.timer import Timer
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray


class MetricsMonitor(LifecycleNode):
    """
    Lifecycle node that tracks search performance metrics and writes results to CSV.

    Monitors agent positions, lidar scans for coverage tracking, and target
    discovery events. Continuously overwrites the current trial's CSV row on
    each coverage sample tick so data survives interrupted runs. Stops
    overwriting once all targets are found.
    """

    def __init__(self) -> None:
        """Initialize the Metrics Monitor and declare parameters."""
        super().__init__("metrics_monitor")
        self.declare_parameter("agent_ids", rclpy.Parameter.Type.STRING_ARRAY)
        self.declare_parameter("num_targets", 0)
        self.declare_parameter("scan_topic", "base_scan")
        self.declare_parameter("csv_path", "output/metrics.csv")
        self.declare_parameter("metrics_rate", 1.0)

        self._agent_ids: list[str] = []
        self._num_targets: int = 0
        self._scan_topic: str = "base_scan"
        self._csv_path: str = "output/metrics.csv"
        self._metrics_rate: float = 1.0

        self._agent_positions: dict[str, Pose | None] = {}
        self._cumulative_distance: dict[str, float] = {}
        self._coverage_grids: dict[str, NDArray[np.bool_]] = {}
        self._found_target_count: int = 0
        self._coverage_history: list[tuple[float, float]] = []
        self._first_target_time: float | None = None
        self._all_targets_time: float | None = None
        self._start_time: float = 0.0
        self._map_data: NDArray[np.uint8] | None = None
        self._map_info: MapMetaData | None = None
        self._csv_valid: bool = False
        self._csv_row_offset: int = -1
        self._metrics_finalized: bool = False

        self._managed_subscriptions: list[Subscription[Odometry | LaserScan | OccupancyGrid | Int32MultiArray]] = []
        self._managed_timers: list[Timer] = []

    # -------------------------------------------------------------------------
    # Lifecycle Callbacks
    # -------------------------------------------------------------------------

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure: read params, create subscriptions, validate/create CSV file."""
        self._agent_ids = self.get_parameter("agent_ids").value
        self._num_targets = self.get_parameter("num_targets").value
        self._scan_topic = self.get_parameter("scan_topic").value
        self._csv_path = self.get_parameter("csv_path").value
        self._metrics_rate = self.get_parameter("metrics_rate").value

        self._agent_positions = dict.fromkeys(self._agent_ids)
        self._cumulative_distance = dict.fromkeys(self._agent_ids, 0.0)
        self._coverage_grids = {}
        self._found_target_count = 0
        self._coverage_history = []
        self._first_target_time = None
        self._all_targets_time = None
        self._map_data = None
        self._map_info = None

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        sub_map = self.create_subscription(OccupancyGrid, "/ground_truth_map", self._on_map, latched_qos)
        self._managed_subscriptions.append(sub_map)

        sub_targets = self.create_subscription(Int32MultiArray, "/targets_found", self._on_targets_found, latched_qos)
        self._managed_subscriptions.append(sub_targets)

        for agent_id in self._agent_ids:
            sub_odom = self.create_subscription(
                Odometry,
                f"/{agent_id}/ground_truth",
                lambda msg, aid=agent_id: self._on_ground_truth_pose(msg, aid),
                10,
            )
            sub_scan = self.create_subscription(
                LaserScan,
                f"/{agent_id}/{self._scan_topic}",
                lambda msg, aid=agent_id: self._on_scan(aid, msg),
                10,
            )
            self._managed_subscriptions.extend([sub_odom, sub_scan])

        self._validate_csv()
        self.get_logger().info("Configured")
        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate: record start time and start coverage sampling timer."""
        self._start_time = self.get_clock().now().nanoseconds / 1e9
        period = 1.0 / self._metrics_rate
        timer = self.create_timer(period, self._sample_coverage)
        self._managed_timers.append(timer)
        self.get_logger().info("Activated")
        return super().on_activate(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivate: cancel timers."""
        for timer in self._managed_timers:
            timer.cancel()
        self._managed_timers.clear()
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
        for timer in self._managed_timers:
            timer.cancel()
        self._managed_timers.clear()
        for sub in self._managed_subscriptions:
            self.destroy_subscription(sub)
        self._managed_subscriptions.clear()

    # -------------------------------------------------------------------------
    # Subscription Callbacks
    # -------------------------------------------------------------------------

    def _on_ground_truth_pose(self, msg: Odometry, agent_id: str) -> None:
        """Update agent position and accumulate distance traveled."""
        prev = self._agent_positions.get(agent_id)
        if prev is not None:
            dx = msg.pose.pose.position.x - prev.position.x
            dy = msg.pose.pose.position.y - prev.position.y
            self._cumulative_distance[agent_id] += math.sqrt(dx * dx + dy * dy)

        self._agent_positions[agent_id] = msg.pose.pose

    def _on_scan(self, agent_id: str, scan: LaserScan) -> None:
        """Ray trace the scan to mark cells in the agent's coverage grid."""
        pose = self._agent_positions.get(agent_id)
        if pose is None or self._map_info is None or self._map_data is None:
            return
        if agent_id not in self._coverage_grids:
            return

        robot_x = pose.position.x
        robot_y = pose.position.y
        robot_yaw = 2.0 * math.atan2(pose.orientation.z, pose.orientation.w)

        res = self._map_info.resolution
        ox = self._map_info.origin.position.x
        oy = self._map_info.origin.position.y
        rows = self._map_info.height
        cols = self._map_info.width

        robot_row = int((robot_y - oy) / res)
        robot_col = int((robot_x - ox) / res)

        angles = np.linspace(scan.angle_min, scan.angle_max, len(scan.ranges))
        ranges = np.array(scan.ranges)

        valid_mask = np.isfinite(ranges)
        valid_angles = angles[valid_mask] + robot_yaw
        valid_ranges = ranges[valid_mask]

        if len(valid_ranges) == 0:
            return

        end_rows = ((robot_y + valid_ranges * np.sin(valid_angles) - oy) / res).astype(int)
        end_cols = ((robot_x + valid_ranges * np.cos(valid_angles) - ox) / res).astype(int)

        all_rr: list[NDArray[np.intp]] = []
        all_cc: list[NDArray[np.intp]] = []
        for i in range(len(valid_ranges)):
            rr, cc = skimage_line(robot_row, robot_col, end_rows[i], end_cols[i])  # type: ignore[no-untyped-call]
            all_rr.append(rr)
            all_cc.append(cc)

        rr = np.concatenate(all_rr)
        cc = np.concatenate(all_cc)
        mask = (rr >= 0) & (rr < rows) & (cc >= 0) & (cc < cols)
        self._coverage_grids[agent_id][rr[mask], cc[mask]] = True

    def _on_targets_found(self, msg: Int32MultiArray) -> None:
        """Update found target count and record milestone times."""
        count = len(msg.data)
        if count > self._found_target_count:
            now = self.get_clock().now().nanoseconds / 1e9
            if self._first_target_time is None:
                self._first_target_time = now
            self._found_target_count = count
            if self._num_targets > 0 and count >= self._num_targets and self._all_targets_time is None:
                self._all_targets_time = now
                self._write_metrics_row()
                self._metrics_finalized = True
                self.get_logger().info("All targets found — final metrics written")

    def _on_map(self, msg: OccupancyGrid) -> None:
        """Cache ground truth map data and initialize coverage grids."""
        self._map_info = msg.info
        self._map_data = np.array(msg.data, dtype=np.uint8).reshape(msg.info.height, msg.info.width)
        for agent_id in self._agent_ids:
            if agent_id not in self._coverage_grids:
                self._coverage_grids[agent_id] = np.zeros((msg.info.height, msg.info.width), dtype=np.bool_)

    # -------------------------------------------------------------------------
    # Coverage Sampling
    # -------------------------------------------------------------------------

    def _sample_coverage(self) -> None:
        """Timer callback: compute coverage, record snapshot, and write metrics row to CSV."""
        if not self._csv_valid:
            self.get_logger().error("CSV file is invalid, cannot record metrics", once=True)
            return

        if self._metrics_finalized:
            return

        if self._map_data is None or not self._coverage_grids:
            return

        coverage_pct, redundant_pct = self._compute_coverage_metrics()
        elapsed = self.get_clock().now().nanoseconds / 1e9 - self._start_time
        self._coverage_history.append((elapsed, coverage_pct))
        self._write_metrics_row(redundant_pct)

    def _compute_coverage_metrics(self) -> tuple[float, float]:
        """
        Compute map coverage and redundant search percentages in a single pass.

        Returns:
            (coverage_pct, redundant_pct) — both as percentages of free cells.

        """
        if self._map_data is None or not self._coverage_grids:
            return 0.0, 0.0

        free_mask = self._map_data == 0
        total_free = int(np.sum(free_mask))
        if total_free == 0:
            return 0.0, 0.0

        union_seen = np.zeros_like(free_mask)
        overlap_count = np.zeros(free_mask.shape, dtype=np.int32)
        for grid in self._coverage_grids.values():
            masked = grid & free_mask
            union_seen |= masked
            overlap_count += masked.astype(np.int32)

        seen_free = int(np.sum(union_seen))
        if seen_free == 0:
            return 0.0, 0.0

        coverage_pct = seen_free / total_free * 100.0
        redundant_pct = int(np.sum(overlap_count[union_seen] > 1)) / seen_free * 100.0
        return coverage_pct, redundant_pct

    # -------------------------------------------------------------------------
    # CSV Handling
    # -------------------------------------------------------------------------

    def _get_csv_header(self) -> list[str]:
        """Build the CSV header row based on current agent IDs."""
        distance_cols = [f"{aid}_distance" for aid in self._agent_ids]
        return [
            "time_to_first_target",
            "time_to_all_targets",
            *distance_cols,
            "cumulative_distance",
            "map_coverage_pct",
            "redundant_search_pct",
            "coverage_over_time",
        ]

    def _validate_csv(self) -> None:
        """Validate or create CSV file with correct header."""
        os.makedirs(os.path.dirname(self._csv_path) or ".", exist_ok=True)

        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self._get_csv_header())
            self._csv_valid = True
            self.get_logger().info(f"Created CSV file: {self._csv_path}")
        else:
            with open(self._csv_path, newline="") as f:
                reader = csv.reader(f)
                existing_header = next(reader, None)
            if existing_header == self._get_csv_header():
                self._csv_valid = True
                self.get_logger().info(f"CSV file validated: {self._csv_path}")
            else:
                self._csv_valid = False
                self.get_logger().error(
                    f"CSV header mismatch in {self._csv_path}. "
                    f"Expected: {self._get_csv_header()}, got: {existing_header}"
                )

    def _write_metrics_row(self, redundant_pct: float | None = None) -> None:
        """
        Write or overwrite the current trial's metrics row in the CSV file.

        Tracks the byte offset where this trial's row starts. On each call,
        seeks back to that offset and truncates before writing, so only one
        row per trial exists in the file at any time.
        """
        if not self._csv_valid:
            self.get_logger().error("CSV file is invalid, skipping metrics write")
            return

        time_to_first = ""
        if self._first_target_time is not None:
            time_to_first = f"{self._first_target_time - self._start_time:.2f}"

        time_to_all = ""
        if self._all_targets_time is not None:
            time_to_all = f"{self._all_targets_time - self._start_time:.2f}"

        distance_values = [f"{self._cumulative_distance.get(aid, 0.0):.2f}" for aid in self._agent_ids]
        cumulative_dist = f"{sum(self._cumulative_distance.values()):.2f}"

        coverage_pct = ""
        if self._coverage_history:
            coverage_pct = f"{self._coverage_history[-1][1]:.2f}"

        redundant_str = ""
        if redundant_pct is None:
            _, computed = self._compute_coverage_metrics()
            redundant_str = f"{computed:.2f}" if self._coverage_grids else ""
        else:
            redundant_str = f"{redundant_pct:.2f}"

        coverage_json = json.dumps([[round(t, 2), round(c, 2)] for t, c in self._coverage_history])

        row = [
            time_to_first,
            time_to_all,
            *distance_values,
            cumulative_dist,
            coverage_pct,
            redundant_str,
            coverage_json,
        ]

        with open(self._csv_path, "r+", newline="") as f:
            if self._csv_row_offset < 0:
                f.seek(0, 2)
                self._csv_row_offset = f.tell()
            else:
                f.seek(self._csv_row_offset)
                f.truncate()

            writer = csv.writer(f)
            writer.writerow(row)


def main(args: list[str] | None = None) -> None:
    """Entry point for the Metrics Monitor."""
    rclpy.init(args=args)
    node = MetricsMonitor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
