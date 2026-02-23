"""Shared types, enums, and configuration dataclasses for the multi-agent search system."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


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


class MsgType(IntEnum):
    """Message type enum matching AgentMessage.msg constants."""

    HEARTBEAT = 0
    COORDINATION = 1


@dataclass
class CommsConfig:
    """Configuration parameters for the communications manager."""

    close_range_threshold: float = 10.0  # meters
    long_range_threshold: float = 50.0  # meters
    fusion_range_threshold: float = 5.0  # meters (unknown map mode only)
    close_range_rate: float = 10.0  # Hz
    long_range_rate: float = 1.0  # Hz
    fusion_cooldown: float = 5.0  # seconds (unknown map mode only)
