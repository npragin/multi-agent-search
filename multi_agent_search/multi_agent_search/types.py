"""Shared types, enums, and configuration dataclasses for the multi-agent search system."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

from geometry_msgs.msg import Pose


class CommsZone(IntEnum):
    """Communication zone between a pair of agents."""

    CLOSE_RANGE = 0
    LONG_RANGE = 1
    BLACKOUT = 2


class NavStatus(IntEnum):
    """Navigation status for an agent."""

    IDLE = 0
    NAVIGATING = 1
    SUCCEEDED = 2
    FAILED = 3


@dataclass
class CommsConfig:
    """Configuration parameters for the communications manager."""

    close_range_threshold: float = 10.0  # meters
    long_range_threshold: float = 50.0  # meters
    fusion_range_threshold: float = 5.0  # meters (unknown map mode only)
    close_range_rate: float = 10.0  # Hz
    long_range_rate: float = 1.0  # Hz
    fusion_cooldown: float = 5.0  # seconds (unknown map mode only)
    fusion_timeout: float = 5.0  # seconds; max time to wait for get/set service calls during fusion


@dataclass
class BaseCoordinationMessage:
    """
    Base dataclass for algorithm-specific coordination message payloads.

    Provides common fields populated automatically by AgentBase before
    sending. Subclasses add protocol-specific fields and are serialized
    with pickle — no manual serialization needed.

    Fields:
        sender_id:  Agent ID of the sender, set by AgentBase before publishing.
        timestamp:  Send time in seconds (ROS time), set by AgentBase before publishing.

    Usage::

        @dataclass
        class MyCoordMsg(BaseCoordinationMessage):
            task_id: int = 0

        # Sending:
        self.publish_coordination_message(MyCoordMsg(task_id=42))

        # Receiving (in on_coordination):
        def on_coordination(self, sender: str, msg: BaseCoordinationMessage) -> None:
            assert isinstance(msg, MyCoordMsg)
            print(msg.sender_id, msg.task_id)
    """

    sender_id: str = ""
    timestamp: float = 0.0


@dataclass
class HeartbeatMessage:
    """Heartbeat message carrying this agent's current pose."""

    sender_id: str = ""
    timestamp: float = 0.0
    pose: Pose = field(default_factory=Pose)
