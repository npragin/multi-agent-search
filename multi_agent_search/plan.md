# Multi-Agent Communication System Architecture

## Overview

This document describes the software architecture for a ROS2-based multi-agent search system with communication constraints. The system consists of a central **Communications Manager Node** that mediates all inter-agent communication, and an **Agent Base Class** that provides common communication infrastructure for agent implementations.

The system uses Nav2 for motion execution and SLAM. AgentBase integrates with Nav2 as an action client for navigation goals and subscribes to slam_toolbox's map output rather than maintaining its own scan-based map updates. Nav2 components are namespaced under `agent_id`.

### Communication Model

- **Close Range:** Agents within close-range threshold receive all message types (configurable rate, default 10 Hz)
- **Long Range:** Agents beyond close range but within long-range threshold receive only heartbeat and target-found messages (configurable rate, default 1 Hz). Coordination messages are not delivered.
- **Blackout:** No communication possible (beyond long-range threshold)
- **Fusion Range:** Agents within fusion threshold *and* with line-of-sight trigger map/belief fusion (both are occupancy grids). Fusion is only active in unknown map mode.

Line-of-sight is only required for map fusion, not for communication. The comms manager enforces message rates and message type filtering—agents publish to a single channel, but messages are only propagated to other agents at the appropriate rate based on their pairwise distance, and long-range recipients only receive heartbeat and target-found messages.

**Important:** When an agent publishes a non-heartbeat message, long-range agents do not continue to receive the previous heartbeat. The comms manager delivers the most recent message from each sender, subject to rate and type filtering.

---

## Message Types

All inter-agent communication uses a single unified message type:

```
# AgentMessage.msg
uint8 msg_type
string sender_id
string recipient_id           # Empty string = broadcast
uint64 timestamp
bool overwrite_targeted       # If broadcast, whether to overwrite recipient-specific messages
bytes payload                 # Serialized inner message
```

### Message Type Enum

| Value | Type | Description | Delivered at Long Range |
|-------|------|-------------|-------------------------|
| 0 | HEARTBEAT | Pose update and liveness signal | Yes |
| 1 | TARGET_FOUND | Alert that a target has been discovered | Yes |
| 2 | COORDINATION | Algorithm-specific (task assignment, ACK/NACK, exploration intent, etc.) | **No** |

The coordination message type is intentionally generic. Subclasses define their own coordination protocol, which may include:
- Task assignments / rendezvous commands
- ACK / NACK for reliable delivery
- Exploration intentions
- Any other algorithm-specific communication

This keeps the base infrastructure simple while allowing full flexibility for different search algorithms.

**Note:** Coordination messages are only delivered to agents within close range. Long-range agents will not receive coordination messages regardless of when they were sent.

---

## Communications Manager Node

### Responsibilities

1. Receive messages from all agents on a single input topic per agent
2. Determine pairwise communication zones based on distance and line-of-sight
3. Propagate messages at appropriate rates (1 Hz or 10 Hz) to recipient agents
4. Filter message types based on communication zone (long-range excludes coordination)
5. Manage map fusion when agents are in fusion range with line-of-sight (unknown map mode only)
6. Track agent liveness and positions

### State

```python
class CommsManager(Node):
    
    # Agent tracking
    agent_ids: set[str]                           # Registered agents
    agent_poses: dict[str, Pose]                  # Current pose per agent (from ground truth)
    
    # Message buffer
    # Outer dict: sender_id -> Inner dict: recipient_id -> message
    # recipient_id of "" (empty string) represents broadcast
    message_buffer: dict[str, dict[str, AgentMessage]]
    
    # Communication zone cache
    # Key: (agent_a, agent_b) sorted tuple
    pairwise_zones: dict[tuple[str, str], CommsZone]
    
    # Map fusion tracking (unknown map mode only)
    # Key: (agent_a, agent_b) sorted tuple
    last_fusion_time: dict[tuple[str, str], Time]
    fusion_cooldown: float                        # Seconds between fusions for a pair
    
    # Last delivery tracking (for rate limiting)
    # Key: (sender_id, recipient_id) -> last delivery time
    last_delivery_time: dict[tuple[str, str], Time]
    
    # Configuration
    use_known_map: bool                           # Disables fusion when True
    close_range_threshold: float                  # Distance for close-range zone
    long_range_threshold: float                   # Distance for long-range zone (beyond = blackout)
    fusion_range_threshold: float                 # Distance for map fusion eligibility
    close_range_rate: float                       # Hz for close-range message propagation
    long_range_rate: float                        # Hz for long-range message propagation
```

### Topics

#### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/comms/input/{agent_id}` | AgentMessage | Messages from agent |
| `/{agent_id}/base_pose_ground_truth` | Odometry | Ground truth pose from Stage simulator |

#### Publications

| Topic | Type | Description |
|-------|------|-------------|
| `/comms/output/{agent_id}` | AgentMessage | Messages delivered to specific agent |

### Services (Clients)

| Service | Type | Description |
|---------|------|-------------|
| `/{agent_id}/get_map` | GetMap | Retrieve agent's occupancy grid (environment map) |
| `/{agent_id}/set_map` | SetMap | Update agent's occupancy grid (environment map) |
| `/{agent_id}/get_belief` | GetMap | Retrieve agent's belief grid (search coverage) |
| `/{agent_id}/set_belief` | SetMap | Update agent's belief grid (search coverage) |
| `/{agent_id}/fusion_complete` | FusionComplete | Notify agent that fusion is complete |
| `/{agent_id}/map_server/load_map` | LoadMap | Push fused map to Nav2's map server (unknown map mode only) |

### Functions

#### Initialization

```python
def __init__(self, agent_ids: list[str], config: CommsConfig):
    """
    Initialize comms manager with known agent IDs and configuration.
    Sets up all subscriptions (including ground truth pose), publishers, 
    timers, and service clients.
    """

def _setup_agent(self, agent_id: str) -> None:
    """
    Create subscription and publisher for a single agent.
    Includes subscription to agent's ground truth pose topic.
    Called during init for each agent.
    """
```

#### Ground Truth Pose Tracking

```python
def _on_ground_truth_pose(self, msg: Odometry, agent_id: str) -> None:
    """
    Callback for ground truth pose from Stage simulator.
    Updates agent_poses dict for use in distance calculations and LOS checks.
    The comms manager uses ground truth rather than relying on heartbeat 
    messages to ensure accurate zone computation independent of communication.
    """
```

#### Message Handling

```python
def _on_message(self, msg: AgentMessage, sender_id: str) -> None:
    """
    Callback for input topic.
    Updates message_buffer respecting overwrite_targeted flag.
    """

def _update_buffer(
    self, 
    sender_id: str, 
    msg: AgentMessage
) -> None:
    """
    Update message buffer with new message, respecting overwrite_targeted flag.
    
    If message is targeted (recipient_id != ""):
        - Simply store/overwrite at message_buffer[sender][recipient]
    
    If message is broadcast (recipient_id == ""):
        - Store at message_buffer[sender][""]
        - If overwrite_targeted is True: clear all targeted messages for this sender
        - If overwrite_targeted is False: preserve existing targeted messages
    """

def _propagate_close_range(self) -> None:
    """
    Timer callback (close-range rate).
    For each sender, determine what to send to each recipient in CLOSE_RANGE zone:
    - All message types are delivered (heartbeat, target_found, coordination)
    - If targeted message exists for recipient: send targeted
    - Else if broadcast exists: send broadcast
    """

def _propagate_long_range(self) -> None:
    """
    Timer callback (long-range rate).
    For each sender, determine what to send to each recipient in LONG_RANGE zone.
    
    Filtering logic:
    - Skip recipients in CLOSE_RANGE (close-range handler covers them)
    - Skip recipients in BLACKOUT
    - For LONG_RANGE recipients: only deliver if msg_type is HEARTBEAT or TARGET_FOUND
    - Coordination messages are NOT delivered to long-range recipients
    
    Note: If the sender's current message is a coordination message, long-range
    recipients receive nothing (the previous heartbeat is not preserved).
    """

def _publish_to_agent(self, recipient_id: str, msg: AgentMessage) -> None:
    """Publish message to specific agent's output topic."""

def _should_deliver(self, sender: str, recipient: str, msg: AgentMessage) -> bool:
    """
    Check if message from sender should be delivered to recipient.
    
    Returns True if:
    - Pairwise zone is CLOSE_RANGE, OR
    - Pairwise zone is LONG_RANGE AND msg_type is HEARTBEAT or TARGET_FOUND
    
    Returns False if:
    - Pairwise zone is BLACKOUT, OR
    - Pairwise zone is LONG_RANGE AND msg_type is COORDINATION
    """

def _is_long_range_eligible(self, msg_type: int) -> bool:
    """
    Check if message type can be delivered at long range.
    Returns True for HEARTBEAT and TARGET_FOUND, False for COORDINATION.
    """

def _get_pairwise_zone(self, agent_a: str, agent_b: str) -> CommsZone:
    """Look up cached pairwise zone."""
```

#### Communication Zone Management

```python
def _update_pairwise_zones(self) -> None:
    """
    Recompute communication zones for all agent pairs.
    Called periodically or when poses update.
    Uses distance and line-of-sight checks.
    """

def _compute_zone(self, agent_a: str, agent_b: str) -> CommsZone:
    """
    Determine comms zone between two agents.
    1. Compute distance from poses
    2. If distance > long_range_threshold: BLACKOUT
    3. If distance > close_range_threshold: LONG_RANGE
    4. Otherwise: CLOSE_RANGE
    
    Note: Line-of-sight is only checked for fusion eligibility, not communication.
    """

def _has_line_of_sight(self, pose_a: Pose, pose_b: Pose) -> bool:
    """
    Ray-cast between two poses to check for obstacles.
    Implementation depends on map representation.
    """
```

#### Map Fusion (Unknown Map Mode Only)

```python
def _check_fusion_eligibility(self) -> None:
    """
    Timer callback (1 Hz or slower). No-op if use_known_map is True.
    For each agent pair within fusion range with line-of-sight, check if fusion is due.
    Initiates fusion if cooldown has elapsed.
    """

def _should_fuse(self, agent_a: str, agent_b: str) -> bool:
    """
    Check if pair is eligible for fusion.
    Always returns False if use_known_map is True.
    Otherwise returns True if:
    - Distance is within fusion_range_threshold
    - Agents have line-of-sight
    - Cooldown period has elapsed since last fusion
    """

def _perform_fusion(self, agent_a: str, agent_b: str) -> None:
    """
    Execute map and belief fusion for an agent pair (unknown map mode only).
    1. Call get_map and get_belief on both agents
    2. Fuse environment maps (cell-wise max or Bayesian update)
    3. Fuse belief/coverage grids (cell-wise max or Bayesian update)
    4. Call set_map and set_belief on both agents
    5. Push fused map to /{agent_id}/map_server/load_map on both agents
       so Nav2's global costmap static layer reflects the updated map
    6. Call fusion_complete on both agents to trigger hooks
    7. Update last_fusion_time for pair
    
    Note: slam_toolbox's pose graph is not modified. The fused map improves
    Nav2's global planning without affecting localization.
    """

def _fuse_maps(self, map_a: OccupancyGrid, map_b: OccupancyGrid) -> OccupancyGrid:
    """
    Combine two environment occupancy grids.
    Strategy: cell-wise maximum probability (conservative)
    or Bayesian update if log-odds representation.
    """

def _fuse_beliefs(self, belief_a: OccupancyGrid, belief_b: OccupancyGrid) -> OccupancyGrid:
    """
    Combine belief grids (search coverage).
    Both beliefs are occupancy grids representing search coverage probability.
    Strategy: cell-wise maximum (if either agent has searched an area, mark it searched)
    or Bayesian update if log-odds representation.
    """
```

#### Utility

```python
def _get_pair_key(self, agent_a: str, agent_b: str) -> tuple[str, str]:
    """Return sorted tuple for consistent dictionary keys."""
    return tuple(sorted([agent_a, agent_b]))

def get_active_agents(self, timeout: float = 5.0) -> list[str]:
    """Return list of agents with recent heartbeats."""
```

---

## Agent Base Class

### Responsibilities

1. Provide publishing interface for outgoing messages (single channel)
2. Subscribe to incoming messages from comms manager
3. Deserialize messages and dispatch to appropriate callbacks
4. Expose services for map/belief get and set
5. Manage belief grid updates from lidar
6. Source environment map from slam_toolbox or known-map node via map topic subscription
7. Detect targets based on lidar scan endpoints and configured target locations
8. Publish found targets for external success criteria evaluation
9. Send navigation goals to Nav2 and dispatch navigation result callbacks

### State

```python
class AgentBase(Node, ABC):
    
    # Identity
    agent_id: str
    
    # Publisher (single channel - comms manager handles rate/type filtering)
    pub_outgoing: Publisher          # To /comms/input/{agent_id}
    pub_found_targets: Publisher     # To /{agent_id}/found_targets
    
    # Subscribers
    sub_incoming: Subscription       # From /comms/output/{agent_id}
    sub_lidar: Subscription          # From /{agent_id}/scan
    sub_map: Subscription            # From /{agent_id}/map (slam_toolbox or known-map node)
    
    # Nav2 interface
    _nav_client: ActionClient        # NavigateToPose at /{agent_id}/navigate_to_pose
    _current_nav_goal: GoalHandle    # Handle to active navigation goal, or None
    _nav_status: NavStatus           # IDLE, NAVIGATING, SUCCEEDED, FAILED
    
    # Local state (managed by base class, exposed via services)
    _map: OccupancyGrid              # Environment map (sourced from map topic subscription)
    _belief: OccupancyGrid           # Search coverage grid (agent-managed)
    
    # Target detection (configured via parameters)
    _target_positions: list[tuple[float, float]]  # (x, y) coordinates of targets
    _target_radius: float                          # Detection radius for all targets
    _found_targets: set[int]                       # Indices of targets that have been found
    
    # Services
    srv_get_map: Service
    srv_set_map: Service
    srv_get_belief: Service
    srv_set_belief: Service
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `use_known_map` | bool | If true, subscribe to known map from external node. If false, subscribe to slam_toolbox output. Both publish to `/{agent_id}/map`. |
| `target_positions` | list[tuple[float, float]] | List of (x, y) coordinates for target locations |
| `target_radius` | float | Detection radius - target is found if lidar endpoint within this distance |

### Topics

#### Publications

| Topic | Type | Description |
|-------|------|-------------|
| `/comms/input/{agent_id}` | AgentMessage | Outgoing messages |
| `/{agent_id}/found_targets` | Int32MultiArray | Indices of found targets (for external success evaluation, not for agent subscription) |

#### Subscriptions

| Topic | Type | Description |
|-------|------|-------------|
| `/comms/output/{agent_id}` | AgentMessage | Incoming messages from comms manager |
| `/{agent_id}/scan` | LaserScan | Lidar scan data |
| `/{agent_id}/map` | OccupancyGrid | Map from slam_toolbox (unknown map mode) or known-map node (known map mode) |

### Services (Servers)

| Service | Type | Description |
|---------|------|-------------|
| `/{agent_id}/get_map` | GetMap | Return current environment occupancy grid |
| `/{agent_id}/set_map` | SetMap | Update environment occupancy grid (from fusion) |
| `/{agent_id}/get_belief` | GetMap | Return current belief/coverage occupancy grid |
| `/{agent_id}/set_belief` | SetMap | Update belief/coverage occupancy grid (from fusion) |
| `/{agent_id}/fusion_complete` | FusionComplete | Signal that fusion is complete, triggers hook |

### Functions

#### Initialization

```python
def __init__(self, agent_id: str):
    """
    Initialize base agent with ID.
    Sets up publisher, subscriber, services, nav client, and timers.
    """

def _setup_comms(self) -> None:
    """Create publisher and subscriber for comms manager interface."""

def _setup_services(self) -> None:
    """Create map/belief get/set services."""

def _setup_nav(self) -> None:
    """Create NavigateToPose action client at /{agent_id}/navigate_to_pose."""

def _setup_timers(self) -> None:
    """Create timers for periodic tasks if needed."""
```

#### Publishing Interface

The subclass publishes directly to `pub_outgoing` whenever it wants to update what the comms manager is sending. The comms manager handles rate limiting and message type filtering based on recipient distance.

```python
def publish_message(
    self,
    msg_type: int,
    payload: bytes,
    recipient: str = "",
    overwrite_targeted: bool = True
) -> None:
    """
    Publish message to the comms manager.
    
    Args:
        msg_type: Message type enum value (HEARTBEAT, TARGET_FOUND, COORDINATION)
        payload: Serialized message data
        recipient: Specific recipient ID, or "" for broadcast
        overwrite_targeted: For broadcasts, whether to clear recipient-specific 
                           messages in comms manager buffer
    
    Note: The comms manager will deliver this message to close-range agents at
    close-range rate. For long-range agents, only HEARTBEAT and TARGET_FOUND
    messages will be delivered (at long-range rate). COORDINATION messages are
    never delivered to long-range agents.
    
    Important: When you publish a COORDINATION message, long-range agents will
    not receive your previous heartbeat - they simply receive nothing until you
    publish another HEARTBEAT or TARGET_FOUND message.
    """
```

#### Message Packing Helpers

```python
def pack_heartbeat(self, pose: Pose) -> tuple[int, bytes]:
    """Serialize heartbeat message. Returns (msg_type, payload)."""

def pack_target_found(self, target_id: str, location: Point) -> tuple[int, bytes]:
    """Serialize target found alert."""

# Coordination packing is handled entirely by subclass
```

#### Message Receiving

```python
def _on_incoming_message(self, msg: AgentMessage) -> None:
    """
    Callback for messages from comms manager.
    Deserializes payload and dispatches to appropriate handler.
    """

def _dispatch_message(self, msg: AgentMessage) -> None:
    """
    Route message to appropriate callback based on msg_type.
    """
```

#### Navigation Interface

```python
def navigate_to(self, goal: PoseStamped) -> None:
    """
    Send a NavigateToPose goal to Nav2.
    Cancels any active goal before sending the new one.
    Sets _nav_status to NAVIGATING.
    """

def cancel_navigation(self) -> None:
    """
    Cancel the active navigation goal if one exists.
    Sets _nav_status to IDLE.
    """
```

#### Map Subscription Handler

```python
def _on_map_updated(self, msg: OccupancyGrid) -> None:
    """
    Callback for /{agent_id}/map topic.
    Updates _map with the latest occupancy grid from slam_toolbox or the
    known-map node. Calls on_map_updated() hook after updating.
    """
```

#### Service Handlers

```python
def _handle_get_map(self, request, response) -> GetMapResponse:
    """Service handler: return current environment map."""
    
def _handle_set_map(self, request, response) -> SetMapResponse:
    """
    Service handler: update environment map from fusion.
    Note: Does NOT call on_map_updated - fusion completion is handled separately
    via on_fusion_completed.
    """

def _handle_get_belief(self, request, response) -> GetMapResponse:
    """Service handler: return current belief/coverage grid."""

def _handle_set_belief(self, request, response) -> SetMapResponse:
    """
    Service handler: update belief/coverage grid from fusion.
    Note: Does NOT call on_belief_updated - fusion completion is handled separately
    via on_fusion_completed.
    """

def _handle_fusion_complete(self, request, response) -> FusionCompleteResponse:
    """
    Service handler: called by comms manager after both map and belief are set.
    Calls on_fusion_completed() hook with new map and belief.
    """
```

#### Belief Update and Target Detection (Base Class Implementation)

```python
def _on_lidar_callback(self, scan: LaserScan) -> None:
    """
    Internal callback for lidar data.
    1. Updates belief/coverage grid via _update_belief_from_scan
    2. Checks for target detection via _check_for_targets
    3. Forwards to subclass via on_lidar_scan for algorithm-specific processing
    """

def _update_belief_from_scan(self, scan: LaserScan) -> None:
    """
    Update belief/coverage grid based on sensor field of view.
    Marks cells within sensor range as searched.
    Called automatically by _on_lidar_callback.
    """

def _check_for_targets(self, scan: LaserScan) -> None:
    """
    Check if any lidar beam passes within target_radius of any target position.
    
    Sampling along the beam (rather than checking only the endpoint) is necessary
    because a target may lie along the beam's path without being the closest
    obstacle - the beam would pass through the target and report a farther
    endpoint, causing endpoint-only detection to miss it entirely.
    
    For each beam in the scan:
    1. Sample points at regular intervals along the beam from origin to endpoint
    2. For each target not yet found:
       - Compute the minimum distance from any sample point to the target position
       - If minimum distance <= target_radius, mark target as found
    
    Sample interval should be small enough relative to target_radius that no
    target can be skipped between samples (e.g., target_radius / 2).
    
    When a new target is found:
    1. Add target index to _found_targets set
    2. Publish updated found_targets list to /{agent_id}/found_targets
    3. Call on_target_detected() hook for subclass to handle (e.g., publish TARGET_FOUND message)
    
    Called automatically by _on_lidar_callback.
    """

def _sample_beam(
    self,
    origin: tuple[float, float],
    angle: float,
    range_m: float,
    step: float
) -> list[tuple[float, float]]:
    """
    Generate world-frame sample points along a single lidar beam.
    
    Args:
        origin: (x, y) world-frame position of the sensor
        angle:  World-frame angle of the beam in radians
        range_m: Range of the beam in meters (up to max range)
        step:   Distance between consecutive sample points in meters
    
    Returns list of (x, y) world-frame coordinates sampled from the
    sensor origin to the beam endpoint at intervals of `step`.
    The endpoint itself is always included as the final sample.
    """

def _publish_found_targets(self) -> None:
    """
    Publish the current list of found target indices to /{agent_id}/found_targets.
    Message contains sorted list of indices corresponding to _target_positions.
    This topic is for external success criteria evaluation only.
    """
```

#### Abstract Callbacks (Must Implement)

```python
@abstractmethod
def on_heartbeat(self, sender: str, pose: Pose, timestamp: Time) -> None:
    """Called when heartbeat received from another agent."""

@abstractmethod
def on_target_found(self, sender: str, target_id: str, location: Point) -> None:
    """Called when another agent reports finding a target."""

@abstractmethod
def on_coordination(self, sender: str, payload: bytes) -> None:
    """
    Called when coordination message received.
    Subclass must deserialize payload according to its coordination protocol.
    
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
    Called when lidar scan received, after base class has updated belief
    and checked for targets. Subclass can use this for algorithm-specific 
    processing (e.g., frontier identification) but does NOT need to handle 
    belief updates or target detection.
    """

@abstractmethod
def on_target_detected(self, target_index: int, target_position: tuple[float, float]) -> None:
    """
    Called when this agent detects a target via lidar scan.
    
    Args:
        target_index: Index into the target_positions parameter list
        target_position: (x, y) coordinates of the detected target
    
    Subclass should implement this to handle target discovery, typically by:
    - Publishing a TARGET_FOUND message to notify other agents
    - Updating internal search state
    - Potentially triggering a rendezvous or task reassignment
    
    Note: The base class has already updated _found_targets and published
    to the found_targets topic before calling this hook.
    """
```

#### Virtual Hooks (Optional Override)

```python
def on_map_updated(self, new_map: OccupancyGrid) -> None:
    """
    Called after _map is updated from the map topic subscription.
    NOT called during fusion - use on_fusion_completed for that.
    Override to trigger replanning or other responses.
    Default: no-op.
    """

def on_belief_updated(self, new_belief: OccupancyGrid) -> None:
    """
    Called after belief/coverage grid is updated from a lidar scan.
    NOT called during fusion - use on_fusion_completed for that.
    Override to trigger replanning or other responses.
    Default: no-op.
    """

def on_fusion_completed(self, new_map: OccupancyGrid, new_belief: OccupancyGrid) -> None:
    """
    Called after both map and belief are updated via fusion with another agent.
    on_map_updated and on_belief_updated are NOT called when fusion occurs;
    this hook is the sole notification for fusion-driven grid updates.
    Override to trigger replanning or respond to newly discovered information.
    Default: no-op.
    """

def on_navigation_succeeded(self) -> None:
    """
    Called when Nav2 reports goal reached.
    Default: no-op.
    """

def on_navigation_failed(self, reason: str) -> None:
    """
    Called when Nav2 reports failure or the goal is aborted.
    Subclass should typically select a new goal or trigger recovery.
    Default: no-op.
    """
```

---

## Agent Subclass Requirements

A concrete agent implementation must:

### 1. Inherit from AgentBase

```python
class MySearchAgent(AgentBase):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)
        # Algorithm-specific initialization
```

### 2. Implement All Abstract Callbacks

```python
def on_heartbeat(self, sender: str, pose: Pose, timestamp: Time) -> None:
    # Update internal model of peer locations
    pass

def on_target_found(self, sender: str, target_id: str, location: Point) -> None:
    # Mark target as found, potentially abort if searching for same target
    pass

def on_coordination(self, sender: str, payload: bytes) -> None:
    # Deserialize and handle algorithm-specific coordination
    # Note: Only received from close-range agents
    cmd = self._deserialize_coordination(payload)
    
    # Example: handle different coordination subtypes
    if cmd.type == CoordType.TASK_ASSIGN:
        self._handle_task_assignment(sender, cmd)
    elif cmd.type == CoordType.RENDEZVOUS:
        self._handle_rendezvous(sender, cmd)
    elif cmd.type == CoordType.ACK:
        self._handle_ack(sender, cmd)
    elif cmd.type == CoordType.NACK:
        self._handle_nack(sender, cmd)
    elif cmd.type == CoordType.EXPLORATION_INTENT:
        self._handle_exploration_intent(sender, cmd)

def on_lidar_scan(self, scan: LaserScan) -> None:
    # Base class has already updated belief and checked for targets
    # Use this for algorithm-specific processing
    self._identify_frontiers()

def on_target_detected(self, target_index: int, target_position: tuple[float, float]) -> None:
    # Notify other agents about the target discovery
    msg_type, payload = self.pack_target_found(
        target_id=str(target_index),
        location=Point(x=target_position[0], y=target_position[1], z=0.0)
    )
    self.publish_message(msg_type, payload)
    
    # Update internal state
    self._update_search_priority(target_index)
```

### 3. Define Coordination Message Structure

The subclass owns the entire coordination protocol:

```python
class CoordType(Enum):
    TASK_ASSIGN = 0
    RENDEZVOUS = 1
    ACK = 2
    NACK = 3
    EXPLORATION_INTENT = 4
    # Add algorithm-specific types as needed

@dataclass
class MyCoordinationCommand:
    type: CoordType
    msg_id: Optional[str]            # For ACK/NACK tracking
    task_id: Optional[str]
    target_location: Optional[Point]
    rendezvous_time: Optional[Time]
    exploration_goal: Optional[Point]
    nack_reason: Optional[str]
    # ... algorithm-specific fields

def _serialize_coordination(self, cmd: MyCoordinationCommand) -> bytes:
    # JSON, protobuf, pickle, etc.
    pass

def _deserialize_coordination(self, payload: bytes) -> MyCoordinationCommand:
    pass

def send_coordination(
    self, 
    cmd: MyCoordinationCommand, 
    recipient: str = "",
    overwrite_targeted: bool = True
) -> None:
    """
    Helper to send coordination messages.
    Note: These will only be delivered to close-range agents.
    """
    payload = self._serialize_coordination(cmd)
    self.publish_message(MsgType.COORDINATION, payload, recipient, overwrite_targeted)
```

### 4. Respond to Map, Belief, and Target Updates

The base class manages `_map`, `_belief` (both occupancy grids), and target detection:

- **Environment map (`_map`):** Updated automatically from the `/{agent_id}/map` topic subscription (slam_toolbox in unknown map mode, known-map node in known map mode)
- **Belief/coverage grid (`_belief`):** Updated automatically from sensor field of view by the base class
- **Target detection:** Base class checks lidar beams against configured target positions
- **Fusion updates (unknown map mode only):** Both grids may be overwritten when fusion occurs with nearby agents; fused map is also pushed to Nav2's map server

The subclass can:
- Read `_map` and `_belief` for planning and decision-making
- Override `on_map_updated()` and `on_belief_updated()` hooks to respond to scan-driven updates (these are **not** called during fusion)
- Override `on_fusion_completed()` hook to respond to fusion events (the sole notification for fusion-driven grid updates)
- Implement `on_target_detected()` to handle target discoveries (e.g., publish TARGET_FOUND message)
- Use `on_lidar_scan()` for algorithm-specific processing (e.g., frontier identification)

### 5. Drive Agent Behavior

The base class handles communication and navigation dispatch; the subclass implements the search/exploration logic:

```python
def main_loop(self) -> None:
    """
    Called by timer or spin.
    1. Process latest lidar data (handled by on_lidar_scan callback)
    2. Decide next action based on belief, peer states, coordination
    3. Publish heartbeat (consider that publishing coordination will
       prevent long-range agents from receiving your heartbeat)
    4. Send coordination messages as needed (close-range only)
    5. Call navigate_to() with selected goal
    """
```

### 6. Consider Communication Range Implications

When designing the search algorithm, keep in mind:

- **Heartbeats maintain long-range awareness:** Publishing heartbeats allows distant agents to track your position.
- **Coordination is close-range only:** Task assignments, acknowledgments, and exploration intentions only reach nearby agents.
- **Publishing coordination blocks long-range heartbeats:** When you publish a coordination message, long-range agents receive nothing until your next heartbeat or target-found message.
- **Strategy consideration:** You may want to alternate between heartbeats and coordination messages, or ensure heartbeats are sent at a regular interval even when coordinating.

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Agent Subclass                              │
│  - Search/exploration logic                                         │
│  - Coordination protocol                                            │
│  - Frontier selection, navigate_to() calls                         │
│  - Peer tracking                                                    │
└─────────────────────────────────────────────────────────────────────┘
                                   │
                                   │ inherits
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          AgentBase                                  │
│  - Publish interface (publish_message)                              │
│  - Message dispatch (on_heartbeat, on_target_found, on_coordination)│
│  - Map subscription (_map sourced from /{agent_id}/map)            │
│  - Belief updates and target detection from lidar                  │
│  - Nav2 action client (navigate_to / cancel_navigation)            │
│  - Map/belief services                                              │
└─────────────────────────────────────────────────────────────────────┘
           │                                         ▲
           │ /comms/input/{id}                       │ /comms/output/{id}
           ▼                                         │
┌─────────────────────────────────────────────────────────────────────┐
│                       CommsManager                                  │
│  - Rate-limited propagation                                         │
│  - Message type filtering (coordination excluded at long range)     │
│  - Pairwise zone computation                                        │
│  - Map fusion orchestration (unknown map mode only)                 │
│  - Pushes fused map to /{agent_id}/map_server/load_map             │
└─────────────────────────────────────────────────────────────────────┘
           │                                         ▲
           │ /{id}/get_map, /{id}/set_map            │
           │ /{id}/get_belief, /{id}/set_belief      │
           ▼                                         │
┌─────────────────────────────────────────────────────────────────────┐
│                          AgentBase                                  │
│                    (service handlers)                               │
└─────────────────────────────────────────────────────────────────────┘
           │                                         
           │ /{id}/navigate_to_pose (action)         
           ▼                                         
┌─────────────────────────────────────────────────────────────────────┐
│                            Nav2                                     │
│  - Motion execution and obstacle avoidance                          │
│  - slam_toolbox: SLAM and localization (unknown map mode)          │
│  - Known-map node: static map publisher (known map mode)           │
│  - map_server: serves current map to global costmap                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Communication Behavior Summary

| Sender's Current Message | Close-Range Recipient | Long-Range Recipient |
|--------------------------|----------------------|----------------------|
| HEARTBEAT | Receives at 10 Hz | Receives at 1 Hz |
| TARGET_FOUND | Receives at 10 Hz | Receives at 1 Hz |
| COORDINATION | Receives at 10 Hz | **Receives nothing** |

---

## Configuration Parameters

### Communications Manager

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_known_map` | false | Disables map fusion when true |
| `close_range_threshold` | 10.0 m | Distance for close-range communication |
| `long_range_threshold` | 50.0 m | Distance for long-range communication (beyond = blackout) |
| `close_range_rate` | 10.0 Hz | Propagation rate for close-range messages |
| `long_range_rate` | 1.0 Hz | Propagation rate for long-range messages |
| `fusion_range_threshold` | 5.0 m | Distance for map fusion eligibility (unknown map mode only) |
| `fusion_cooldown` | 5.0 s | Minimum time between fusions for a pair (unknown map mode only) |

### Agent Base

| Parameter | Default | Description |
|-----------|---------|-------------|
| `use_known_map` | false | If true, `/{agent_id}/map` is expected from a known-map node; slam_toolbox runs in localization-only mode. If false, slam_toolbox runs in mapping mode. |
| `target_positions` | [] | List of (x, y) coordinates for target locations |
| `target_radius` | — | Detection radius for target discovery via lidar beam sampling |

---

## Open Questions / Future Considerations

1. **Message size limits:** Should we enforce hard limits on payload size in `AgentMessage`? Large payloads (e.g., embedding full map data in coordination messages) could impact performance and aren't realistic for real-world comms. If limits are added, the comms manager could drop oversized messages or the base class could reject them at publish time.

2. **Fused map and Nav2 global planner mismatch:** After pushing a fused map to Nav2's map server, the global costmap may briefly plan through areas slam_toolbox hasn't locally verified yet. Nav2's local planner recovery behaviors handle this in practice, but recovery parameters should be tuned once agents are running.