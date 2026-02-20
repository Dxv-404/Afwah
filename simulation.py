"""
Afwah: Multi-Platform Misinformation Cascade Simulation
Core simulation engine — Phase 2: Full Feature Set

Event-driven simulation of rumor spread through social media networks
modeled as queuing networks with heterogeneous agents.
Supports multi-platform (WhatsApp, Twitter, Instagram, Reddit) with
per-platform topologies, service times, and cross-platform hopping.
"""

from __future__ import annotations

import heapq
import math
import random
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import networkx as nx


# =============================================================================
# Enums & Constants
# =============================================================================

class AgentType(Enum):
    REGULAR = "regular"
    INFLUENCER = "influencer"
    FACT_CHECKER = "fact_checker"
    BOT = "bot"
    LURKER = "lurker"


class NodeStatus(Enum):
    UNAWARE = "unaware"
    BELIEVING = "believing"
    SILENT_BELIEVER = "silent_believer"
    CORRECTED = "corrected"
    IMMUNE = "immune"
    REMOVED = "removed"


class Platform(Enum):
    WHATSAPP = "whatsapp"
    TWITTER = "twitter"
    INSTAGRAM = "instagram"
    REDDIT = "reddit"


class EventType(Enum):
    MESSAGE_ARRIVAL = "message_arrival"
    PROCESS_MESSAGE = "process_message"
    SHARE_RUMOR = "share_rumor"
    GENERATE_CORRECTION = "generate_correction"
    SHARE_CORRECTION = "share_correction"
    ATTENTION_RECOVERY = "attention_recovery"
    CHECK_TERMINATION = "check_termination"
    CHECKPOINT = "checkpoint"
    # Phase 2 events
    WHATSAPP_SELF_CORRECTION = "whatsapp_self_correction"
    REDDIT_MOD_ACTION = "reddit_mod_action"
    STORY_EXPIRY_CHECK = "story_expiry_check"
    PLATFORM_HOP = "platform_hop"
    TWITTER_COMMUNITY_NOTE = "twitter_community_note"
    EMERGENCY_CORRECTION = "emergency_correction"
    # Group E events
    SUPER_SPREADER_CHECK = "super_spreader_check"
    CRISIS_START = "crisis_start"
    CRISIS_END = "crisis_end"
    BOT_WAVE = "bot_wave"
    BOT_CREDIBILITY_TICK = "bot_credibility_tick"


class DeathType(Enum):
    STARVED = "starved"
    CORRECTED = "corrected"
    SATURATED = "saturated"
    MUTATED_AWAY = "mutated_away"
    TIME_DECAYED = "time_decayed"
    STILL_ALIVE = "still_alive"


class SeedPersona(Enum):
    RANDOM_PERSON = "random_person"
    NEWS_CHANNEL = "news_channel"
    BLOGGER = "blogger"
    CELEBRITY = "celebrity"
    ANONYMOUS_TIP = "anonymous_tip"


class RumorShape(Enum):
    """Message framing archetypes for rumor shares (Item 28)."""
    BARE_FORWARD = "bare_forward"
    REACTION = "reaction"
    PERSONAL_FRAME = "personal_frame"
    ELABORATOR = "elaborator"
    SKEPTIC_SHARER = "skeptic_sharer"


class CorrectionShape(Enum):
    """Message framing archetypes for correction shares (Item 28)."""
    DEBUNK = "debunk"
    RELAY = "relay"
    TOLD_YOU_SO = "told_you_so"
    RELUCTANT_WALKBACK = "reluctant_walkback"


# =============================================================================
# Configuration Dicts
# =============================================================================

# Item 29: Framing modifier values per shape
RUMOR_FRAMING_MODIFIERS = {
    RumorShape.BARE_FORWARD:   0.00,  # no persuasive framing
    RumorShape.REACTION:       0.05,  # emotional contagion
    RumorShape.PERSONAL_FRAME: 0.15,  # personal credibility boost
    RumorShape.ELABORATOR:     0.10,  # detail = believability
    RumorShape.SKEPTIC_SHARER: 0.20,  # skeptic endorsement = most persuasive
}

CORRECTION_FRAMING_MODIFIERS = {
    CorrectionShape.DEBUNK:             0.15,  # authoritative
    CorrectionShape.RELAY:              0.05,  # casual
    CorrectionShape.TOLD_YOU_SO:        0.00,  # off-putting
    CorrectionShape.RELUCTANT_WALKBACK: 0.10,  # peer credibility
}

AGENT_TYPE_CONFIG = {
    AgentType.REGULAR: {
        "population_pct": 0.60,
        "threshold_dist": ("uniform", 0.4, 0.8),
        "connections": (5, 15),
        "service_time_modifier": 1.0,
    },
    AgentType.INFLUENCER: {
        "population_pct": 0.05,
        "threshold_dist": ("uniform", 0.2, 0.5),
        "connections": (100, 500),
        "service_time_modifier": 0.4,
    },
    AgentType.FACT_CHECKER: {
        "population_pct": 0.03,
        "threshold_dist": ("fixed", 0.95),
        "connections": (50, 200),
        "service_time_modifier": 3.0,
    },
    AgentType.BOT: {
        "population_pct": 0.07,
        "threshold_dist": ("fixed", 0.01),
        "connections": (30, 100),
        "service_time_modifier": 0.02,
    },
    AgentType.LURKER: {
        "population_pct": 0.25,
        "threshold_dist": ("uniform", 0.8, 0.99),
        "connections": (3, 8),
        "service_time_modifier": 6.0,
    },
}

PLATFORM_CONFIG = {
    Platform.TWITTER: {
        "topology": "barabasi_albert",
        "arrival_rate": 15.0,        # λ per minute (Item 2)
        "base_service_time": 30.0,   # μ in seconds (Item 1)
        "threshold_modifier": 1.15,  # Item 3: public skepticism → higher threshold
        "base_credibility": 0.4,
        "correction_speed_modifier": 1.5,
        "algorithmic_amplification": True,
        "engagement_threshold_pct": 0.10,
        "engagement_window_min": 5,
        "forward_limit": None,
        "story_decay": False,
        "story_ttl": None,
        "bot_detection_multiplier": 1.5,
        "edge_bidirectional": False,  # Item 8: Twitter follows are directed
    },
    Platform.WHATSAPP: {
        "topology": "watts_strogatz",
        "arrival_rate": 2.0,         # λ per minute
        "base_service_time": 600.0,  # 10 minutes in seconds
        "threshold_modifier": 0.85,  # Item 3: personal trust → lower threshold
        "base_credibility": 0.7,
        "correction_speed_modifier": 0.5,
        "algorithmic_amplification": False,
        "engagement_threshold_pct": None,
        "engagement_window_min": None,
        "forward_limit": 5,
        "story_decay": False,
        "story_ttl": None,
        "bot_detection_multiplier": 0.3,
        "edge_bidirectional": True,  # Item 8: WhatsApp is mutual
    },
    Platform.INSTAGRAM: {
        "topology": "ring_lattice",
        "arrival_rate": 5.0,         # λ per minute
        "base_service_time": 300.0,  # 5 minutes
        "threshold_modifier": 1.0,   # Item 3: medium credibility
        "base_credibility": 0.5,
        "correction_speed_modifier": 0.8,
        "algorithmic_amplification": True,
        # Spec: Twitter > Instagram in virality hierarchy. At N=500, Instagram's
        # ring-lattice + story decay creates high local clustering that amplifies spread.
        # Raised threshold to 0.23 so algo amp triggers less frequently than Twitter's 0.10.
        # (Increased from 0.20 after demographic sharing modifier §2.6 was implemented.)
        "engagement_threshold_pct": 0.23,
        "engagement_window_min": 10,
        "forward_limit": None,
        "story_decay": True,
        "story_ttl": 86400.0,        # 24-hour story TTL
        "bot_detection_multiplier": 1.0,
        "edge_bidirectional": True,  # Item 8: Instagram follow clusters are mutual-ish
    },
    Platform.REDDIT: {
        "topology": "stochastic_block",
        "arrival_rate": 8.0,         # λ per minute
        "base_service_time": 120.0,  # 2 minutes
        "threshold_modifier": 1.0,   # Item 3: varies by karma (applied dynamically)
        "base_credibility": 0.5,
        "correction_speed_modifier": 1.3,
        "algorithmic_amplification": False,
        "engagement_threshold_pct": None,
        "engagement_window_min": None,
        "forward_limit": None,
        "story_decay": False,
        "story_ttl": None,
        "bot_detection_multiplier": 1.3,
        "edge_bidirectional": True,  # Item 8: Reddit communities are undirected
    },
}

# Platform preference weights by agent type (for multi-platform assignment, Item 7)
PLATFORM_PREFERENCE = {
    AgentType.REGULAR:      {Platform.WHATSAPP: 0.25, Platform.TWITTER: 0.25, Platform.INSTAGRAM: 0.25, Platform.REDDIT: 0.25},
    AgentType.INFLUENCER:   {Platform.WHATSAPP: 0.1,  Platform.TWITTER: 0.4,  Platform.INSTAGRAM: 0.3,  Platform.REDDIT: 0.2},
    AgentType.FACT_CHECKER:  {Platform.WHATSAPP: 0.15, Platform.TWITTER: 0.4,  Platform.INSTAGRAM: 0.15, Platform.REDDIT: 0.3},
    AgentType.BOT:          {Platform.WHATSAPP: 0.1,  Platform.TWITTER: 0.5,  Platform.INSTAGRAM: 0.1,  Platform.REDDIT: 0.3},
    AgentType.LURKER:       {Platform.WHATSAPP: 0.2,  Platform.TWITTER: 0.2,  Platform.INSTAGRAM: 0.3,  Platform.REDDIT: 0.3},
}

# Multi-platform count distribution (Item 7): how many platforms each node is on
MULTI_PLATFORM_DISTRIBUTION = {
    1: 0.45,  # 45% on 1 platform
    2: 0.40,  # 40% on 2 platforms
    3: 0.13,  # 13% on 3 platforms
    4: 0.02,  # 2% on all 4 platforms
}

PLATFORM_FIT_MATRIX = {
    #                    text  image  video  reel
    Platform.WHATSAPP:  [0.9,  0.7,  0.5,  0.3],
    Platform.TWITTER:   [1.0,  0.8,  0.6,  0.4],
    Platform.INSTAGRAM: [0.3,  0.9,  0.7,  1.0],
    Platform.REDDIT:    [0.9,  0.6,  0.4,  0.2],
}

MEDIA_TYPES = ["text", "image", "video", "reel"]

# ISSUE 10 FIX: Platform-specific default media types.
# Instagram is an image/video platform — text-only posts are uncommon IRL.
# Using "reel" for Instagram also triggers story TTL (24h expiry_time),
# enabling the story_decay mechanic that was otherwise dead for text.
PLATFORM_DEFAULT_MEDIA = {
    Platform.TWITTER: "text",
    Platform.WHATSAPP: "text",
    Platform.INSTAGRAM: "reel",
    Platform.REDDIT: "text",
}

RUMOR_SCENARIOS = {
    "celebrity": {
        "topic_tag": "celebrity",
        "emotions": {"fear": 0.4, "outrage": 0.2, "humor": 0.1, "curiosity": 0.8, "urgency": 0.6},
        "platform_affinity": {
            Platform.WHATSAPP: 0.5, Platform.TWITTER: 0.9,
            Platform.INSTAGRAM: 0.8, Platform.REDDIT: 0.4,
        },
    },
    "financial": {
        "topic_tag": "financial",
        "emotions": {"fear": 0.9, "outrage": 0.5, "humor": 0.0, "curiosity": 0.3, "urgency": 1.0},
        "platform_affinity": {
            Platform.WHATSAPP: 0.9, Platform.TWITTER: 0.7,
            Platform.INSTAGRAM: 0.2, Platform.REDDIT: 0.6,
        },
    },
    "health": {
        "topic_tag": "health",
        "emotions": {"fear": 0.6, "outrage": 0.2, "humor": 0.1, "curiosity": 0.5, "urgency": 0.4},
        "platform_affinity": {
            Platform.WHATSAPP: 0.8, Platform.TWITTER: 0.5,
            Platform.INSTAGRAM: 0.4, Platform.REDDIT: 0.7,
        },
    },
    "campus": {
        "topic_tag": "campus",
        "emotions": {"fear": 0.0, "outrage": 0.0, "humor": 0.3, "curiosity": 0.7, "urgency": 0.5},
        "platform_affinity": {
            Platform.WHATSAPP: 0.7, Platform.TWITTER: 0.4,
            Platform.INSTAGRAM: 0.6, Platform.REDDIT: 0.5,
        },
    },
}

SEED_PERSONA_CONFIG = {
    # ISSUE 2 FIX: All 5 personas must have DISTINCT credibility modifiers
    SeedPersona.RANDOM_PERSON: {
        "maps_to": AgentType.REGULAR,
        "credibility_modifier": 1.0,
    },
    SeedPersona.NEWS_CHANNEL: {
        "maps_to": AgentType.INFLUENCER,
        "credibility_modifier": 2.5,
    },
    SeedPersona.BLOGGER: {
        "maps_to": AgentType.REGULAR,
        "credibility_modifier": 1.3,  # spec §1.6: moderate following
    },
    SeedPersona.CELEBRITY: {
        "maps_to": AgentType.INFLUENCER,
        "credibility_modifier": 1.8,  # spec §1.6: fame but not news authority
    },
    SeedPersona.ANONYMOUS_TIP: {
        "maps_to": AgentType.BOT,
        "credibility_modifier": 0.7,  # spec §1.6: no identity = less trust
    },
}

# Base activity curve (probability of being online per hour)
BASE_ACTIVITY_CURVE = [
    0.05, 0.05, 0.06, 0.07, 0.08, 0.10,  # 0-5
    0.30, 0.45, 0.60,                       # 6-8
    0.40, 0.45, 0.50,                       # 9-11
    0.65, 0.75,                              # 12-13
    0.35, 0.40, 0.45,                        # 14-16
    0.70, 0.80, 0.85, 0.90, 0.80,           # 17-21
    0.50, 0.30,                              # 22-23
]

SENDER_TRUST_MODIFIERS = {
    "influencer": 2.0,
    "strong_tie": 1.5,
    "regular": 1.0,
    "stranger": 0.6,
    "bot_detected": 0.3,
    "fact_checker_shares_rumor": 3.0,
}

# Checkpoint times in seconds (for Monte Carlo consistency)
CHECKPOINT_TIMES = [300, 600, 900, 1800, 2700, 3600, 7200, 14400, 28800, 43200, 86400, 172800]  # 1h, 2h, 4h, 8h, 12h, 24h, 48h

# Item 30: Age-group demographics
AGE_GROUP_DISTRIBUTION = {"young": 0.40, "middle": 0.35, "older": 0.25}
AGE_GROUP_DIGITAL_NATIVITY = {
    "young":  (7, 3),   # Beta(7,3) -> mean 0.7
    "middle": (5, 5),   # Beta(5,5) -> mean 0.5
    "older":  (3, 7),   # Beta(3,7) -> mean 0.3
}
AGE_GROUP_PLATFORM_BIAS = {
    "young":  {Platform.TWITTER: 0.30, Platform.INSTAGRAM: 0.40, Platform.REDDIT: 0.20, Platform.WHATSAPP: 0.10},
    "middle": {Platform.TWITTER: 0.25, Platform.INSTAGRAM: 0.20, Platform.REDDIT: 0.15, Platform.WHATSAPP: 0.40},
    "older":  {Platform.TWITTER: 0.10, Platform.INSTAGRAM: 0.10, Platform.REDDIT: 0.05, Platform.WHATSAPP: 0.75},
}

# Item 32: Bot detection multipliers per platform
BOT_DETECTION_PLATFORM_MULT = {
    Platform.TWITTER: 1.5,
    Platform.REDDIT: 1.3,
    Platform.INSTAGRAM: 1.0,
    Platform.WHATSAPP: 0.3,
}


# =============================================================================
# Dataclasses
# =============================================================================

@dataclass
class EmotionVector:
    """5-dimensional emotional profile."""
    fear: float = 0.0
    outrage: float = 0.0
    humor: float = 0.0
    curiosity: float = 0.0
    urgency: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.fear, self.outrage, self.humor, self.curiosity, self.urgency])

    def dot(self, other: EmotionVector) -> float:
        return float(np.dot(self.as_array(), other.as_array()))

    @classmethod
    def from_dict(cls, d: dict) -> EmotionVector:
        return cls(**d)

    def to_dict(self) -> dict:
        return {"fear": self.fear, "outrage": self.outrage, "humor": self.humor,
                "curiosity": self.curiosity, "urgency": self.urgency}


@dataclass
class LiteracyVector:
    """Topic-specific media literacy."""
    health: float = 0.5
    financial: float = 0.5
    political: float = 0.5
    celebrity: float = 0.5
    campus: float = 0.5
    tech: float = 0.5

    def get(self, topic: str) -> float:
        return getattr(self, topic, 0.5)


@dataclass
class Edge:
    """Connection between two nodes on a platform."""
    source: int
    target: int
    relationship_strength: float  # 0.0-1.0 (Granovetter strong/weak tie)
    trust_weight: float           # how much source trusts target's judgment
    platform: Platform
    bidirectional: bool = False


@dataclass
class Rumor:
    """A rumor instance (including mutations)."""
    id: int
    version: int = 0
    parent_version: Optional[int] = None

    # Content
    scenario: str = "celebrity"
    media_type: str = "text"
    content_quality: float = 0.5

    # Emotional profile
    emotions: EmotionVector = field(default_factory=EmotionVector)

    # 4D worldview alignment per spec §4.7: [political, health_trust, tech_trust, authority_trust]
    alignment_vector: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Spread tracking
    origin_platform: Platform = Platform.TWITTER
    origin_node: int = 0
    origin_time: float = 0.0
    forward_count: int = 0
    total_infections: int = 0

    # Platform-specific (Item 10, 11, 12)
    forwarded_tag: bool = False         # WhatsApp: True if forward_count >= 5 (spec §3.2)
    expiry_time: Optional[float] = None # Instagram stories: 24h TTL
    karma_score: int = 1               # Reddit: +1 on believe, -1 on reject
    quarantined: bool = False          # Reddit: moderator quarantined
    removed_by_mod: bool = False       # Reddit: moderator removed

    # Mutation tracking
    mutation_chain: list = field(default_factory=list)
    mutation_distance: float = 0.0

    # Virality (computed)
    virality_score: float = 0.0

    # Display
    display_text: Optional[str] = None


@dataclass
class Node:
    """A person/agent in the simulation network."""
    id: int  # Global unique ID (shared across all platforms)
    agent_type: AgentType = AgentType.REGULAR

    # Core
    credibility_threshold: float = 0.5

    # Per-platform connections (Item 7/8):
    # platform_connections: {Platform: [global_node_ids]}
    # platform_edge_weights: {Platform: {global_node_id: strength}}
    platform_connections: dict = field(default_factory=dict)
    platform_edge_weights: dict = field(default_factory=dict)

    # Legacy single-platform (populated by engine for active platform context)
    connections: list = field(default_factory=list)
    edge_weights: dict = field(default_factory=dict)

    # 2D Worldview per spec §2.2: [political_leaning, topic_interest]
    worldview_vector: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # Emotional susceptibility (dynamic)
    susceptibility: EmotionVector = field(default_factory=EmotionVector)
    original_susceptibility: EmotionVector = field(default_factory=EmotionVector)
    emotional_priming: EmotionVector = field(default_factory=EmotionVector)
    emotional_fatigue: EmotionVector = field(default_factory=EmotionVector)
    messages_processed_per_emotion: dict = field(
        default_factory=lambda: {"fear": 0, "outrage": 0, "humor": 0, "curiosity": 0, "urgency": 0}
    )

    # Literacy (shared across platforms)
    literacy_vector: LiteracyVector = field(default_factory=LiteracyVector)

    # Multi-platform identity (Item 7)
    platforms: list = field(default_factory=list)          # [Platform] — which platforms this node is on
    platform_node_ids: dict = field(default_factory=dict)  # {Platform: local_id}  (secondary key)
    echo_chamber_idx: int = 0  # which echo chamber this node belongs to

    # Cross-platform
    hop_tendency: float = 0.2

    # Attention budget
    attention_budget: float = 1.0

    # Time behavior
    active_hours_profile: list = field(default_factory=lambda: list(BASE_ACTIVITY_CURVE))
    base_lambda: float = 1.0

    # Item 30: Demographics
    age_group: str = "middle"           # "young", "middle", "older"
    digital_nativity: float = 0.5       # 0-1, from Beta distribution
    bot_detection_intuition: float = 0.15  # digital_nativity * 0.3

    # Item 31: Reactive rewiring
    rewiring_events: int = 0            # max 3
    rumor_sources: dict = field(default_factory=dict)  # {sender_id: count}

    # Item 32: Coordinated bot networks
    bot_cluster_id: Optional[int] = None
    apparent_credibility: float = 0.3   # bots build cred over time
    shares_this_hour: int = 0
    detected: bool = False

    # State (changes during simulation)
    status: NodeStatus = NodeStatus.UNAWARE
    times_exposed: int = 0
    times_correction_seen: int = 0
    effective_threshold: float = 0.5
    queue: deque = field(default_factory=lambda: deque(maxlen=1000))
    infected_by: Optional[int] = None
    infected_at: Optional[float] = None
    infected_on_platform: Optional[Platform] = None  # which platform they got infected on
    downstream_infections: int = 0
    rumor_version: int = 0


@dataclass
class SimEvent:
    """An event in the priority queue."""
    time: float
    event_type: EventType
    node_id: int
    data: dict = field(default_factory=dict)
    platform: Optional[Platform] = None  # which platform this event belongs to

    def __lt__(self, other):
        return self.time < other.time


@dataclass
class EventLogEntry:
    """A logged event for analysis and replay."""
    time: float
    event_type: str
    node_id: int
    target_id: Optional[int] = None
    platform: Optional[Platform] = None
    details: dict = field(default_factory=dict)


@dataclass
class CheckpointSnapshot:
    """Simulation state at a checkpoint time."""
    time: float
    believing_count: int
    silent_believer_count: int
    corrected_count: int
    unaware_count: int
    immune_count: int
    removed_count: int
    infection_rate: float
    r0_estimate: float
    total_mutations: int
    platform_hops: int


@dataclass
class SimulationConfig:
    """All configurable parameters for a simulation run."""
    # Network
    network_size: int = 500
    num_echo_chambers: int = 4

    # Multi-platform (Item 7, Item 9)
    # active_platforms: which platforms to simulate. Default=all 4.
    # seed_platform: where the rumor starts.
    # For single-platform comparison runs, set active_platforms=[Platform.TWITTER].
    active_platforms: list = field(default_factory=lambda: [
        Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT
    ])
    seed_platform: Platform = Platform.TWITTER

    # Legacy single-platform alias (used when active_platforms has exactly 1 entry)
    @property
    def platform(self) -> Platform:
        return self.seed_platform

    # Rumor
    scenario: str = "celebrity"
    media_type: str = "text"
    seed_persona: SeedPersona = SeedPersona.RANDOM_PERSON

    # Timing
    max_time: float = 172800.0  # 48 hours in seconds
    start_hour: int = 10        # simulation starts at 10 AM (peak morning activity)
    termination_check_interval: float = 60.0  # check every 60 seconds

    # Mechanics
    decay_rate: float = 0.90
    mutation_probability: float = 0.05
    silent_believer_probability: float = 0.60
    correction_fatigue_rate: float = 0.70
    bias_strength: float = 0.4
    emotion_weight: float = 0.15
    attention_cost: float = 0.08  # Phase 3.6: increased from 0.06 for faster depletion
    attention_recovery_rate: float = 0.04  # per simulated hour (Phase 3.6: slowed from 0.15)

    # Echo chamber wiring
    base_connection_prob: float = 0.15  # Item 6
    within_chamber_boost: float = 3.0   # Item 6

    # Reproducibility (Item 9)
    master_seed: Optional[int] = None

    # Feature toggles
    attention_budget_enabled: bool = True
    correction_enabled: bool = True

    # Item 19: Emergency correction injection (None = disabled, float = sim time in seconds)
    correction_injection_time: Optional[float] = None

    # Item 35: Crisis system
    crisis_enabled: bool = False
    crisis_time: Optional[float] = None      # BUG 12: force crisis at specific time (None = random)
    crisis_duration: Optional[float] = None  # None = random Exp(mu=4h)
    crisis_intensity: Optional[float] = None  # None = random Uniform(0.3, 0.8)

    # Phase 3: Sensitivity analysis toggles (spec §6.9)
    bot_detection_enabled: bool = True       # disable bot detection mechanic
    framing_bonus_enabled: bool = True       # disable rumor framing modifiers
    rewiring_enabled: bool = True            # disable seek/unfollow rewiring
    attention_budget_toggle: bool = True     # disable attention cost/recovery (infinite attention)
    emotional_dynamics_mode: str = 'both'    # 'static'|'priming_only'|'fatigue_only'|'both'
    worldview_dimensions: int = 4            # 2 or 4 (2 neutralizes tech_trust, authority_trust)
    correction_delay_override: Optional[float] = None  # seconds; None = default FC timing
    sharing_probability_modifier: float = 1.0  # multiply believe_probability
    network_density_modifier: float = 1.0      # multiply target connection counts in generators
    algorithmic_amplification_multiplier: float = 1.0  # multiply boost_count; 0 = disable

    # Phase 3: Detailed tracking (for single-run graphs)
    detailed_tracking: bool = False          # finer time-series every 60s of sim time

    # Phase 3: Herd immunity / literacy placement
    literacy_placement_strategy: Optional[str] = None  # 'random'|'bridge'|'influencer'|'echo_seed'
    literacy_placement_pct: float = 0.0       # fraction of nodes to boost
    literacy_placement_topic: str = "celebrity"

    # Phase 3: Surgical counterfactual overrides
    remove_top_n_bridges: int = 0  # number of bridge nodes to immunize per-run (betweenness centrality)
    block_first_influencer: bool = False             # immunize first infected influencer
    bot_detection_rate_multiplier: float = 1.0       # 2.0 = "bots detected 1h earlier"


@dataclass
class SimulationResult:
    """Output from a single simulation run."""
    # Terminal state
    termination_reason: str = "max_time"
    termination_time: float = 0.0

    # Infection stats
    final_infection_rate: float = 0.0
    peak_infection_rate: float = 0.0
    peak_time: float = 0.0

    # Counts
    total_believing: int = 0
    total_silent_believers: int = 0
    total_corrected: int = 0
    total_unaware: int = 0
    total_mutations: int = 0
    total_platform_hops: int = 0

    # Group E counts
    total_bots_detected: int = 0
    total_rewiring_events: int = 0
    total_super_spreader_events: int = 0
    crisis_occurred: bool = False

    # R0
    r0_final: float = 0.0

    # Death type
    death_type: DeathType = DeathType.STILL_ALIVE

    # Time series
    infection_timeline: list = field(default_factory=list)  # [(time, infection_rate)]
    r0_timeline: list = field(default_factory=list)          # [(time, r0)]

    # Checkpoints
    checkpoints: list = field(default_factory=list)  # [CheckpointSnapshot]

    # Event log
    event_log: list = field(default_factory=list)  # [EventLogEntry]

    # Per-platform stats (Phase 2)
    platform_infection_rates: dict = field(default_factory=dict)  # {Platform: rate}
    platform_node_counts: dict = field(default_factory=dict)      # {Platform: count}

    # Phase 3: Detailed tracking timelines (only populated when detailed_tracking=True)
    detailed_timelines: Optional[dict] = None   # {metric_name: [(time, value), ...]}
    node_data_snapshot: Optional[dict] = None    # {node_id: {status, agent_type, ...}}


@dataclass
class MonteCarloResult:
    """Aggregated results from a Monte Carlo batch of simulation runs (Phase 3)."""
    n_runs: int = 0
    base_seed: int = 42
    scenario: str = "celebrity"
    platform: str = "twitter"
    network_size: int = 500

    # Per-run arrays (numpy)
    infection_rates: np.ndarray = field(default_factory=lambda: np.array([]))
    termination_times: np.ndarray = field(default_factory=lambda: np.array([]))
    r0_values: np.ndarray = field(default_factory=lambda: np.array([]))
    tipping_points: np.ndarray = field(default_factory=lambda: np.array([]))

    # Death type distribution
    death_type_counts: dict = field(default_factory=dict)

    # Convergence data: [(n_runs_so_far, running_mean, ci_low, ci_high)]
    running_means: list = field(default_factory=list)

    # Lightweight per-run results (event_logs stripped)
    results: list = field(default_factory=list)

    # Aggregate stats
    mean_infection: float = 0.0
    std_infection: float = 0.0
    ci_95_lower: float = 0.0
    ci_95_upper: float = 0.0
    median_infection: float = 0.0
    mean_r0: float = 0.0
    mean_termination_time: float = 0.0

    # Timing
    total_time_seconds: float = 0.0
    avg_time_per_run: float = 0.0

    # Config used (for reproducibility)
    config_overrides: dict = field(default_factory=dict)


# =============================================================================
# Network Generation (Phase 2 — all 4 platform topologies)
# =============================================================================

def generate_worldview_clusters(num_chambers: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Generate echo chamber centroids in 4D worldview space per spec §5.1.

    Dimensions: [political (-1..1), health_trust (0..1), tech_trust (0..1), authority_trust (0..1)].
    """
    predefined = [
        np.array([0.7, 0.3, 0.6, 0.4]),    # right-leaning, health-skeptic, tech-friendly, moderate trust
        np.array([-0.5, 0.8, 0.4, 0.7]),    # left-leaning, health-trusting, moderate tech, high authority
        np.array([0.1, 0.2, 0.2, 0.1]),     # centrist, skeptic of everything
        np.array([-0.3, 0.6, 0.8, 0.5]),    # center-left, health-moderate, tech-optimist, moderate authority
        np.array([0.5, 0.1, 0.1, 0.2]),     # right-leaning, skeptic of health+tech+authority
        np.array([0.0, 0.5, 0.5, 0.9]),     # centrist, moderate health/tech, high authority trust
    ]
    if num_chambers <= len(predefined):
        return predefined[:num_chambers]
    # Extra: political in [-1,1], other dims in [0,1]
    extra = []
    for _ in range(num_chambers - len(predefined)):
        v = np.zeros(4)
        v[0] = rng.uniform(-1.0, 1.0)
        v[1:] = rng.uniform(0.0, 1.0, size=3)
        extra.append(v)
    return predefined + extra


def assign_worldview(chamber_center: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate a node's 4D worldview near its chamber center (spec §5.1)."""
    wv = rng.normal(loc=chamber_center, scale=0.3, size=4)
    wv[0] = np.clip(wv[0], -1.0, 1.0)   # political: [-1, 1]
    wv[1] = np.clip(wv[1], 0.0, 1.0)    # health_trust: [0, 1]
    wv[2] = np.clip(wv[2], 0.0, 1.0)    # tech_trust: [0, 1]
    wv[3] = np.clip(wv[3], 0.0, 1.0)    # authority_trust: [0, 1]
    return wv


def generate_credibility_threshold(agent_type: AgentType, rng: np.random.Generator) -> float:
    """Generate credibility threshold based on agent type distribution."""
    config = AGENT_TYPE_CONFIG[agent_type]
    dist = config["threshold_dist"]
    if dist[0] == "fixed":
        return dist[1]
    elif dist[0] == "uniform":
        return float(rng.uniform(dist[1], dist[2]))
    return 0.5


def generate_emotional_susceptibility(rng: np.random.Generator) -> EmotionVector:
    """Generate emotional susceptibility from Beta(2, 5) for each dimension."""
    return EmotionVector(
        fear=float(rng.beta(2, 5)),
        outrage=float(rng.beta(2, 5)),
        humor=float(rng.beta(2, 5)),
        curiosity=float(rng.beta(2, 5)),
        urgency=float(rng.beta(2, 5)),
    )


def generate_literacy_vector(rng: np.random.Generator) -> LiteracyVector:
    """Generate topic-specific literacy values."""
    return LiteracyVector(
        health=float(rng.uniform(0.2, 0.9)),
        financial=float(rng.uniform(0.2, 0.9)),
        political=float(rng.uniform(0.2, 0.9)),
        celebrity=float(rng.uniform(0.3, 0.95)),
        campus=float(rng.uniform(0.3, 0.95)),
        tech=float(rng.uniform(0.2, 0.9)),
    )


def generate_activity_profile(rng: np.random.Generator) -> list[float]:
    """Generate per-node activity profile with random time shift."""
    shift = int(rng.integers(-3, 4))
    profile = BASE_ACTIVITY_CURVE.copy()
    profile = profile[shift:] + profile[:shift]
    profile = [max(0.01, min(1.0, p + float(rng.normal(0, 0.05)))) for p in profile]
    return profile


def assign_agent_types(n: int, rng: np.random.Generator) -> list[AgentType]:
    """Assign agent types to n nodes based on population percentages."""
    types = []
    for atype, config in AGENT_TYPE_CONFIG.items():
        count = int(n * config["population_pct"])
        types.extend([atype] * count)
    while len(types) < n:
        types.append(AgentType.REGULAR)
    rng.shuffle(types)
    return list(types)


def assign_multi_platform_memberships(
    nodes: list[Node],
    active_platforms: list[Platform],
    rng: np.random.Generator,
) -> None:
    """
    Assign each node to 1-4 platforms using the 45/40/13/2 distribution (Item 7).

    Platform selection is weighted by agent type preferences. For example,
    influencers are biased toward Twitter and Instagram.

    If only 1 platform is active, all nodes get that platform.
    """
    if len(active_platforms) == 1:
        for node in nodes:
            node.platforms = list(active_platforms)
            node.platform_node_ids = {active_platforms[0]: node.id}
        return

    # Build cumulative distribution for platform count
    max_platforms = len(active_platforms)
    count_probs = []
    for k in range(1, max_platforms + 1):
        count_probs.append(MULTI_PLATFORM_DISTRIBUTION.get(k, 0.0))
    # Normalize in case max_platforms < 4 (redistribute higher-count mass)
    total = sum(count_probs)
    count_probs = [p / total for p in count_probs]

    for node in nodes:
        # How many platforms this node is on
        num_platforms = rng.choice(range(1, max_platforms + 1), p=count_probs)

        # Weighted platform selection: blend agent type prefs with age-group bias
        prefs = PLATFORM_PREFERENCE[node.agent_type]
        age_bias = AGE_GROUP_PLATFORM_BIAS.get(node.age_group, AGE_GROUP_PLATFORM_BIAS["middle"])
        # 50/50 blend of agent-type preference and age-group bias
        weights = np.array([
            0.5 * prefs.get(p, 0.25) + 0.5 * age_bias.get(p, 0.25)
            for p in active_platforms
        ])
        weights /= weights.sum()

        # Sample without replacement
        chosen_indices = rng.choice(
            len(active_platforms), size=num_platforms, replace=False, p=weights
        )
        chosen_platforms = [active_platforms[i] for i in chosen_indices]

        node.platforms = chosen_platforms
        # Local IDs: global_id is used as the platform-local ID
        node.platform_node_ids = {p: node.id for p in chosen_platforms}


def _compute_edge_weight(
    worldview_a: np.ndarray,
    worldview_b: np.ndarray,
    same_chamber: bool,
    rng: np.random.Generator,
) -> float:
    """Compute relationship strength between two nodes based on worldview similarity."""
    wv_distance = float(np.linalg.norm(worldview_a - worldview_b))
    # 4D worldview space [-1,1]x[0,1]^3, max distance = 4.0 (spec §4.7)
    base_strength = max(0.1, 1.0 - wv_distance / 4.0)
    if same_chamber:
        base_strength = min(0.99, base_strength * 1.3)  # slightly stronger intra-chamber ties
    strength = float(np.clip(base_strength + rng.normal(0, 0.15), 0.05, 0.99))
    return strength


def _enforce_connection_counts(
    G: nx.Graph,
    node_ids: list[int],
    nodes: list,  # list[Node]
    rng: np.random.Generator,
    density_modifier: float = 1.0,
) -> None:
    """
    Enforce per-agent-type connection count ranges for all nodes (Item 5).

    Two passes:
      Pass 1: Add edges for under-connected nodes (influencers, FCs, bots first)
      Pass 2: Prune edges for over-connected nodes (lurkers, regulars)

    This ordering prevents: adding edges to lurkers to satisfy influencer min,
    then having to prune those same lurker edges in pass 2.

    ISSUE E NOTE: Influencer connection cap vs network size.
    AGENT_TYPE_CONFIG specifies influencers target (100, 500) connections.
    At N=500, each platform gets ~45% of nodes (~225). The hard cap is
    min(spec_max, platform_nodes - 1), so influencers get at most ~224
    connections — well below the spec max of 500. This is intentional:
    at small N, influencers are hub nodes relative to their platform,
    not absolute. Their degree is still 10-20x higher than regular nodes
    (connections: (5, 50)), preserving the power-law tail that drives
    super spreader dynamics. At N=5000+ the full 500 cap becomes reachable.
    """
    all_graph_nodes = set(G.nodes())
    platform_nodes = len(all_graph_nodes)

    # Pre-compute a sampled target degree for each node from its spec range.
    # BUG 1 FIX: Sample from Uniform(min, min(max, platform_nodes-1)) instead of
    # always pinning to the minimum.
    target_degree = {}
    for gid in node_ids:
        atype = nodes[gid].agent_type
        min_conn, max_conn = AGENT_TYPE_CONFIG[atype]["connections"]
        # Phase 3: network_density_modifier scales connection targets
        scaled_min = int(min_conn * density_modifier)
        scaled_max = int(max_conn * density_modifier)
        effective_max = min(scaled_max, platform_nodes - 1)
        effective_min = min(scaled_min, effective_max)
        target_degree[gid] = int(rng.integers(effective_min, effective_max + 1))

    # Pass 1: Bring under-target nodes up to their sampled target
    # Process high-connectivity types first (they need the most edges added)
    priority_order = [
        AgentType.INFLUENCER, AgentType.FACT_CHECKER, AgentType.BOT,
        AgentType.REGULAR, AgentType.LURKER,
    ]
    for atype in priority_order:
        for gid in node_ids:
            if nodes[gid].agent_type != atype:
                continue
            target = target_degree[gid]
            current_degree = G.degree(gid)
            if current_degree < target:
                current_neighbors = set(G.neighbors(gid))
                current_neighbors.add(gid)
                candidates = list(all_graph_nodes - current_neighbors)
                if candidates:
                    needed = min(target - current_degree, len(candidates))
                    rng.shuffle(candidates)
                    for i in range(needed):
                        G.add_edge(gid, candidates[i])

    # Pass 2: Prune over-target nodes
    # ISSUE 10 FIX: Sort neighbors by "excess above target" descending, so we
    # preferentially remove edges to neighbors that are most over their own target.
    # This prevents pruning from undoing pass 1's work on influencers/FCs.
    for gid in node_ids:
        atype = nodes[gid].agent_type
        node_target = target_degree[gid]
        _, spec_max = AGENT_TYPE_CONFIG[atype]["connections"]
        hard_max = min(node_target, spec_max, platform_nodes - 1)
        current_degree = G.degree(gid)
        if current_degree > hard_max:
            # Sort neighbors: most over-target first (best candidates for removal)
            neighbors = list(G.neighbors(gid))
            rng.shuffle(neighbors)  # randomize among equal-excess nodes
            neighbors.sort(key=lambda nbr: -(G.degree(nbr) - target_degree[nbr]))
            excess = current_degree - hard_max
            removed = 0
            for nbr in neighbors:
                if removed >= excess:
                    break
                # Don't prune if it would drop the neighbor below their minimum
                nbr_atype = nodes[nbr].agent_type
                nbr_min, _ = AGENT_TYPE_CONFIG[nbr_atype]["connections"]
                if G.degree(nbr) <= nbr_min:
                    continue
                G.remove_edge(gid, nbr)
                removed += 1

        # Lurker hard cap: spec says 3-8 connections. Force-prune unconditionally
        # if still over 8, regardless of neighbor minimum. Lurker constraint is strict.
        if atype == AgentType.LURKER and G.degree(gid) > spec_max:
            neighbors = list(G.neighbors(gid))
            rng.shuffle(neighbors)
            neighbors.sort(key=lambda nbr: -(G.degree(nbr) - target_degree[nbr]))
            for nbr in neighbors:
                if G.degree(gid) <= spec_max:
                    break
                G.remove_edge(gid, nbr)


def generate_network_barabasi_albert(
    node_ids: list[int],
    nodes: list[Node],
    chamber_assignments: dict[int, int],
    chamber_centers: list[np.ndarray],
    rng: np.random.Generator,
    density_modifier: float = 1.0,
) -> nx.Graph:
    """
    Generate a Barabasi-Albert scale-free network (Twitter topology, Item 4).

    Parameters:
        node_ids: global IDs of nodes on this platform
        nodes: full node list (index by global ID)
        chamber_assignments: {global_id: chamber_idx}
        chamber_centers: list of 4D worldview centroids
        rng: platform-specific RNG
        density_modifier: Phase 3 sensitivity — scales target connection counts
    Returns:
        nx.Graph with node labels = global_ids
    """
    n = len(node_ids)
    if n < 4:
        G = nx.complete_graph(n)
        mapping = {i: node_ids[i] for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        return G

    m = max(1, int(3 * density_modifier))  # BA parameter scaled by density
    G_raw = nx.barabasi_albert_graph(n, m, seed=int(rng.integers(0, 2**31)))

    # Relabel nodes from 0..n-1 to global IDs
    mapping = {i: node_ids[i] for i in range(n)}
    G = nx.relabel_nodes(G_raw, mapping)

    # Enforce connection counts (Item 5)
    _enforce_connection_counts(G, node_ids, nodes, rng, density_modifier)

    _resolve_partitions_global(G, node_ids, rng)
    return G


def generate_network_watts_strogatz(
    node_ids: list[int],
    nodes: list[Node],
    chamber_assignments: dict[int, int],
    chamber_centers: list[np.ndarray],
    rng: np.random.Generator,
    density_modifier: float = 1.0,
) -> nx.Graph:
    """
    Generate a Watts-Strogatz small-world network (WhatsApp topology, Item 4).

    High clustering + short path lengths. Represents tightly-clustered friend groups.
    Parameters: k=6 nearest neighbors, p_rewire=0.15.
    """
    n = len(node_ids)
    if n < 4:
        G = nx.complete_graph(n)
        mapping = {i: node_ids[i] for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        return G

    k = min(max(2, int(6 * density_modifier)), n - 1)  # scaled by density
    if k % 2 != 0:
        k = max(2, k - 1)  # WS requires even k
    p_rewire = 0.15

    G_raw = nx.watts_strogatz_graph(n, k, p_rewire, seed=int(rng.integers(0, 2**31)))

    mapping = {i: node_ids[i] for i in range(n)}
    G = nx.relabel_nodes(G_raw, mapping)

    # Enforce connection counts
    _enforce_connection_counts(G, node_ids, nodes, rng, density_modifier)

    _resolve_partitions_global(G, node_ids, rng)
    return G


def generate_network_ring_lattice(
    node_ids: list[int],
    nodes: list[Node],
    chamber_assignments: dict[int, int],
    chamber_centers: list[np.ndarray],
    num_chambers: int,
    rng: np.random.Generator,
    density_modifier: float = 1.0,
) -> nx.Graph:
    """
    Generate a ring-lattice with bridge connections (Instagram topology, Item 4).

    Construction:
      1. Divide nodes into K clusters (by echo chamber assignment)
      2. Within each cluster: create ring lattice with k_nearest=4
      3. Add intra-cluster random edges: p_intra=0.15
      4. Add inter-cluster bridge edges: p_bridge=0.02
         Bridges preferentially connect to high-degree nodes (influencers)
    """
    n = len(node_ids)
    if n < 4:
        G = nx.complete_graph(n)
        mapping = {i: node_ids[i] for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        return G

    G = nx.Graph()
    G.add_nodes_from(node_ids)

    # Group nodes by chamber
    chamber_groups: dict[int, list[int]] = {}
    for gid in node_ids:
        cidx = chamber_assignments[gid]
        chamber_groups.setdefault(cidx, []).append(gid)

    k_nearest = max(2, int(4 * density_modifier))
    p_intra = 0.15 * density_modifier
    p_bridge = 0.02 * density_modifier

    # Step 1-2: Ring lattice within each cluster
    for cidx, members in chamber_groups.items():
        m = len(members)
        if m <= 1:
            continue
        # Ring lattice: connect each to k_nearest/2 neighbors on each side
        half_k = max(1, min(k_nearest // 2, (m - 1) // 2))
        for i, gid in enumerate(members):
            for j in range(1, half_k + 1):
                neighbor = members[(i + j) % m]
                G.add_edge(gid, neighbor)

        # Step 3: Intra-cluster random rewiring
        for i, gid in enumerate(members):
            for j in range(i + 1, m):
                if not G.has_edge(gid, members[j]) and rng.random() < p_intra:
                    G.add_edge(gid, members[j])

    # Step 4: Inter-cluster bridges (preferentially to influencers)
    all_ids = list(node_ids)
    for i, gid_a in enumerate(all_ids):
        for j in range(i + 1, len(all_ids)):
            gid_b = all_ids[j]
            if chamber_assignments[gid_a] == chamber_assignments[gid_b]:
                continue
            if G.has_edge(gid_a, gid_b):
                continue
            # Bridge probability, boosted if either is influencer
            bp = p_bridge
            if nodes[gid_a].agent_type == AgentType.INFLUENCER:
                bp *= 3.0
            if nodes[gid_b].agent_type == AgentType.INFLUENCER:
                bp *= 3.0
            if rng.random() < bp:
                G.add_edge(gid_a, gid_b)

    # Enforce connection counts
    _enforce_connection_counts(G, node_ids, nodes, rng, density_modifier)

    _resolve_partitions_global(G, node_ids, rng)
    return G


def generate_network_stochastic_block(
    node_ids: list[int],
    nodes: list[Node],
    chamber_assignments: dict[int, int],
    chamber_centers: list[np.ndarray],
    num_chambers: int,
    rng: np.random.Generator,
    density_modifier: float = 1.0,
) -> nx.Graph:
    """
    Generate a Stochastic Block Model network (Reddit topology, Item 4).

    Construction:
      - C communities (= num_echo_chambers)
      - p_within = 0.12 (connection probability within same community)
      - p_between = 0.005 (connection probability across communities)
      - Each community has 1-3 moderator nodes (fact-checker type)
    """
    n = len(node_ids)
    if n < 4:
        G = nx.complete_graph(n)
        mapping = {i: node_ids[i] for i in range(n)}
        G = nx.relabel_nodes(G, mapping)
        return G

    p_within = min(1.0, 0.12 * density_modifier)
    p_between = min(1.0, 0.005 * density_modifier)

    # Group nodes by chamber
    chamber_groups: dict[int, list[int]] = {}
    for gid in node_ids:
        cidx = chamber_assignments[gid]
        chamber_groups.setdefault(cidx, []).append(gid)

    # Build SBM using networkx stochastic_block_model
    # We need sizes and probability matrix
    communities = sorted(chamber_groups.keys())
    sizes = [len(chamber_groups[c]) for c in communities]
    num_blocks = len(communities)

    # Probability matrix
    p_matrix = []
    for i in range(num_blocks):
        row = []
        for j in range(num_blocks):
            if i == j:
                row.append(p_within)
            else:
                row.append(p_between)
        p_matrix.append(row)

    # Generate SBM with sequential node IDs
    G_raw = nx.stochastic_block_model(
        sizes, p_matrix, seed=int(rng.integers(0, 2**31))
    )

    # Build mapping from sequential to global IDs
    mapping = {}
    seq_idx = 0
    for c in communities:
        for gid in chamber_groups[c]:
            mapping[seq_idx] = gid
            seq_idx += 1

    G = nx.relabel_nodes(G_raw, mapping)

    # Enforce connection counts
    _enforce_connection_counts(G, node_ids, nodes, rng, density_modifier)

    _resolve_partitions_global(G, node_ids, rng)
    return G


def _resolve_partitions_global(G: nx.Graph, node_ids: list[int], rng: np.random.Generator):
    """Connect disconnected components with bridge edges (global ID version)."""
    components = list(nx.connected_components(G))
    if len(components) <= 1:
        return

    main = max(components, key=len)
    for comp in components:
        if comp == main:
            continue
        a = rng.choice(list(main))
        b = rng.choice(list(comp))
        G.add_edge(a, b)


def generate_platform_network(
    platform: Platform,
    node_ids: list[int],
    nodes: list[Node],
    chamber_assignments: dict[int, int],
    chamber_centers: list[np.ndarray],
    num_chambers: int,
    rng: np.random.Generator,
    density_modifier: float = 1.0,
) -> nx.Graph:
    """
    Dispatch to the correct topology generator for a given platform (Item 4).
    Returns a networkx Graph with global node IDs as labels.
    """
    topology = PLATFORM_CONFIG[platform]["topology"]

    if topology == "barabasi_albert":
        return generate_network_barabasi_albert(
            node_ids, nodes, chamber_assignments, chamber_centers, rng, density_modifier
        )
    elif topology == "watts_strogatz":
        return generate_network_watts_strogatz(
            node_ids, nodes, chamber_assignments, chamber_centers, rng, density_modifier
        )
    elif topology == "ring_lattice":
        return generate_network_ring_lattice(
            node_ids, nodes, chamber_assignments, chamber_centers, num_chambers, rng,
            density_modifier
        )
    elif topology == "stochastic_block":
        return generate_network_stochastic_block(
            node_ids, nodes, chamber_assignments, chamber_centers, num_chambers, rng,
            density_modifier
        )
    else:
        raise ValueError(f"Unknown topology: {topology}")


def wire_platform_connections(
    nodes: list[Node],
    platform: Platform,
    graph: nx.Graph,
    chamber_assignments: dict[int, int],
    rng: np.random.Generator,
) -> None:
    """
    Populate node.platform_connections[platform] and node.platform_edge_weights[platform]
    from the networkx graph. Also computes relationship strength based on worldview (Item 6).
    """
    for gid in graph.nodes():
        node = nodes[gid]
        neighbors = list(graph.neighbors(gid))
        node.platform_connections[platform] = neighbors

        weights = {}
        for neighbor_id in neighbors:
            neighbor = nodes[neighbor_id]
            same_chamber = (
                chamber_assignments.get(gid, -1) == chamber_assignments.get(neighbor_id, -2)
            )
            w = _compute_edge_weight(
                node.worldview_vector, neighbor.worldview_vector, same_chamber, rng
            )
            weights[neighbor_id] = w
        node.platform_edge_weights[platform] = weights


def generate_all_nodes(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[list[Node], dict[int, int], list[np.ndarray]]:
    """
    Create all Node objects with attributes (worldview, susceptibility, literacy, etc.)
    and assign multi-platform memberships.

    Returns:
        nodes: list of Node, indexed by global ID
        chamber_assignments: {global_id: chamber_idx}
        chamber_centers: list of 4D worldview centroids
    """
    n = config.network_size

    # Echo chamber setup (Item 6)
    chamber_centers = generate_worldview_clusters(config.num_echo_chambers, rng)
    raw_assignments = rng.integers(0, config.num_echo_chambers, size=n)
    chamber_assignments = {i: int(raw_assignments[i]) for i in range(n)}

    # Agent types
    agent_types = assign_agent_types(n, rng)

    # Item 30: Pre-assign age groups for all nodes
    age_groups_pool = []
    for ag, pct in AGE_GROUP_DISTRIBUTION.items():
        age_groups_pool.extend([ag] * int(n * pct))
    while len(age_groups_pool) < n:
        age_groups_pool.append("middle")
    rng.shuffle(age_groups_pool)

    # Create nodes
    nodes = []
    for i in range(n):
        atype = agent_types[i]
        chamber_idx = chamber_assignments[i]
        worldview = assign_worldview(chamber_centers[chamber_idx], rng)
        threshold = generate_credibility_threshold(atype, rng)
        susceptibility = generate_emotional_susceptibility(rng)
        literacy = generate_literacy_vector(rng)
        activity = generate_activity_profile(rng)

        # Item 30: Demographics
        age_group = age_groups_pool[i]
        a_param, b_param = AGE_GROUP_DIGITAL_NATIVITY[age_group]
        digital_nativity = float(rng.beta(a_param, b_param))

        # Item 30: Topic susceptibility correlation per spec §2.6
        if age_group == "young" and digital_nativity > 0.6:
            literacy.celebrity = min(1.0, literacy.celebrity * 1.2)
            literacy.campus = min(1.0, literacy.campus * 1.2)
        if age_group == "older" and digital_nativity < 0.4:
            literacy.health = max(0.0, literacy.health * 0.8)  # health literacy penalty

        node = Node(
            id=i,
            agent_type=atype,
            credibility_threshold=threshold,
            effective_threshold=threshold,
            worldview_vector=worldview,
            susceptibility=susceptibility,
            original_susceptibility=EmotionVector(
                fear=susceptibility.fear,
                outrage=susceptibility.outrage,
                humor=susceptibility.humor,
                curiosity=susceptibility.curiosity,
                urgency=susceptibility.urgency,
            ),
            literacy_vector=literacy,
            echo_chamber_idx=chamber_idx,
            hop_tendency=float(rng.beta(2, 5)),
            active_hours_profile=activity,
            base_lambda=PLATFORM_CONFIG[config.seed_platform]["arrival_rate"],
            age_group=age_group,
            digital_nativity=digital_nativity,
            bot_detection_intuition=digital_nativity * 0.3,
        )
        nodes.append(node)

    # Assign multi-platform memberships (Item 7) — uses age_group platform bias
    assign_multi_platform_memberships(nodes, config.active_platforms, rng)

    return nodes, chamber_assignments, chamber_centers


def generate_all_platform_networks(
    config: SimulationConfig,
    nodes: list[Node],
    chamber_assignments: dict[int, int],
    chamber_centers: list[np.ndarray],
    master_rng: np.random.Generator,
) -> dict[Platform, nx.Graph]:
    """
    Generate per-platform networks using per-platform seeds (Item 9).

    For reproducibility, each platform gets its own RNG derived from master_seed:
        platform_seed = master_seed + platform_index
    This ensures that adding/removing a platform doesn't change other platforms' networks.

    Returns: {Platform: nx.Graph}
    """
    platform_graphs: dict[Platform, nx.Graph] = {}

    for idx, platform in enumerate(config.active_platforms):
        # Per-platform seed derivation (Item 9)
        if config.master_seed is not None:
            platform_seed = config.master_seed + idx + 1  # +1 so main seed != first platform seed
        else:
            platform_seed = int(master_rng.integers(0, 2**31))
        platform_rng = np.random.default_rng(platform_seed)

        # Collect nodes on this platform
        node_ids = [n.id for n in nodes if platform in n.platforms]

        if not node_ids:
            # No nodes on this platform — create empty graph
            platform_graphs[platform] = nx.Graph()
            continue

        # Generate topology
        graph = generate_platform_network(
            platform, node_ids, nodes,
            chamber_assignments, chamber_centers,
            config.num_echo_chambers, platform_rng,
            config.network_density_modifier,
        )

        # Wire connections into node objects
        wire_platform_connections(
            nodes, platform, graph, chamber_assignments, platform_rng,
        )

        platform_graphs[platform] = graph

    return platform_graphs


# =============================================================================
# Rumor Creation
# =============================================================================

def create_seed_rumor(config: SimulationConfig, rng: np.random.Generator) -> Rumor:
    """Create the initial rumor (version 0)."""
    scenario_config = RUMOR_SCENARIOS[config.scenario]
    emotions = EmotionVector.from_dict(scenario_config["emotions"])
    content_quality = float(rng.uniform(0.3, 0.9))

    alignment = _generate_rumor_alignment(config.scenario, rng)

    # Platform fit for virality (using seed platform)
    media_idx = MEDIA_TYPES.index(config.media_type)
    platform_fit = PLATFORM_FIT_MATRIX[config.seed_platform][media_idx]
    emotional_impact = sum(emotions.as_array()) / 5.0
    virality = content_quality * platform_fit * (1 + emotional_impact * 0.3)

    rumor = Rumor(
        id=0,
        version=0,
        scenario=config.scenario,
        media_type=config.media_type,
        content_quality=content_quality,
        emotions=emotions,
        alignment_vector=alignment,
        origin_platform=config.seed_platform,
        origin_time=0.0,
        mutation_chain=[0],
        virality_score=float(np.clip(virality, 0.0, 5.0)),
    )
    return rumor


def _generate_rumor_alignment(scenario: str, rng: np.random.Generator) -> np.ndarray:
    """Generate a 4D alignment vector based on scenario topic (spec §4.7).

    Dims: [political (-1..1), health_trust (0..1), tech_trust (0..1), authority_trust (0..1)].
    """
    base_alignments = {
        # [political, health_trust, tech_trust, authority_trust]
        "celebrity": np.array([0.0, 0.5, 0.5, 0.3]),    # apolitical, moderate on all
        "financial": np.array([0.3, 0.4, 0.6, 0.2]),    # slight right, low authority trust
        "health": np.array([0.1, 0.2, 0.3, 0.3]),       # centrist, low health/tech/auth trust
        "campus": np.array([0.0, 0.5, 0.5, 0.5]),       # apolitical, moderate all
    }
    base = base_alignments.get(scenario, np.array([0.0, 0.5, 0.5, 0.5]))
    noise = rng.normal(0, 0.1, size=4)
    result = base + noise
    result[0] = np.clip(result[0], -1.0, 1.0)   # political
    result[1] = np.clip(result[1], 0.0, 1.0)     # health_trust
    result[2] = np.clip(result[2], 0.0, 1.0)     # tech_trust
    result[3] = np.clip(result[3], 0.0, 1.0)     # authority_trust
    return result


# =============================================================================
# Simulation Engine
# =============================================================================

class SimulationEngine:
    """
    Event-driven simulation of misinformation spread.

    Uses a priority queue (min-heap) to process events in chronological order.
    Each event triggers state changes and potentially spawns new events.
    Supports multi-platform with per-platform topologies and mechanics.
    """

    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.default_rng(config.master_seed)
        self.py_rng = random.Random(
            config.master_seed if config.master_seed is not None
            else int(self.rng.integers(0, 2**31))
        )

        # State
        self.nodes: list[Node] = []
        self.platform_graphs: dict[Platform, nx.Graph] = {}  # Phase 2: per-platform graphs
        self.graph: Optional[nx.Graph] = None  # legacy: seed_platform graph
        self.rumor_versions: dict[int, Rumor] = {}
        self.current_time: float = 0.0
        self.event_queue: list[SimEvent] = []  # min-heap
        self.next_rumor_id: int = 1

        # Tracking
        self.event_log: list[EventLogEntry] = []
        self.infection_timeline: list[tuple[float, float]] = []
        self.r0_timeline: list[tuple[float, float]] = []
        self.checkpoints: list[CheckpointSnapshot] = []
        self.next_checkpoint_idx: int = 0

        # Termination tracking
        self.r0_below_threshold_since: Optional[float] = None
        self.r0_peak: float = 0.0
        self.last_infection_time: float = 0.0

        # Correction tracking
        self.correction_active: bool = False
        self.correction_origin_time: Optional[float] = None
        self.correction_sent_to: set = set()

        # Counters
        self.total_shares: int = 0
        self.total_mutations: int = 0
        self.total_platform_hops: int = 0
        self.shares_in_window: int = 0
        self.window_start: float = 0.0

        # Per-platform share tracking for algorithmic amplification (sliding window)
        self.recent_shares_per_platform: dict[Platform, deque] = {
            p: deque() for p in Platform
        }
        # Per-platform total share counts (cumulative, for community notes trigger)
        self.total_shares_per_platform: dict[Platform, int] = {p: 0 for p in Platform}
        # Legacy alias
        self.recent_shares: deque = deque()

        # Reddit moderator action tracking (Item 12)
        self.mod_actions_per_mod: dict[int, list[float]] = {}  # {mod_id: [action_times]}

        # Group E tracking
        self.bots_detected: int = 0
        self.rewiring_events: int = 0
        self.super_spreader_events: int = 0
        self.crisis_active: bool = False
        self.crisis_pre_thresholds: dict[int, float] = {}  # for recovery
        self.bot_clusters: dict[int, list[int]] = {}  # {cluster_id: [node_ids]}
        self.bot_wave_active: dict[int, bool] = {}    # {cluster_id: currently_in_wave}
        self._last_sse_time: float = -9999.0  # last super spreader event time
        self._current_share_amplified: bool = False  # BUG 2: track if current share is amp-boosted
        # Amplification cooldown: {platform: cooldown_end_time}
        # Once algo amp fires, it can't re-fire on the same platform until boost_duration expires.
        # Spec §3.3: boost_duration = Exp(mu=30min). The boost IS the elevated exposure window.
        self._amp_cooldown_until: dict[Platform, float] = {p: 0.0 for p in Platform}
        self._termination_time: Optional[float] = None  # BUG 14: explicit termination timestamp

        # Phase 3: Detailed tracking (only active when config.detailed_tracking=True)
        self._detailed_next_sample_time: float = 60.0  # first sample at t=60s
        self._detailed_timelines: dict = {
            "time": [],
            "queue_length_avg": [],
            "attention_budget_mean_all": [],
            "attention_budget_mean_hubs": [],
            "attention_budget_p10_hubs": [],  # Phase 3.6: 10th percentile for hub nodes
            "fear_mean": [],
            "outrage_susceptibility_mean": [],
            "bot_survival_fraction": [],
            "cumulative_unfollows": [],
            "cumulative_seeks": [],
            "infection_per_chamber": [],  # list of dicts {chamber_idx: rate}
            "infection_per_platform": [],  # list of dicts {platform: rate}
            "infections_this_period": [],  # for time-of-day graph
            "believing_frac": [],
            "silent_believer_frac": [],
            "corrected_frac": [],
            "total_infected_frac": [],
            "utilization_per_platform": [],
        }
        self._detailed_infections_since_last: int = 0  # counter reset each snapshot
        self._total_seeks: int = 0
        self._total_unfollows: int = 0

    def setup(self):
        """Initialize networks, create seed rumor, schedule initial events."""
        # Generate all nodes with multi-platform memberships
        self.nodes, chamber_assignments, chamber_centers = generate_all_nodes(
            self.config, self.rng
        )

        # Generate per-platform networks (Item 4, Item 9)
        self.platform_graphs = generate_all_platform_networks(
            self.config, self.nodes, chamber_assignments, chamber_centers, self.rng
        )

        # Set legacy single-platform connections from seed_platform
        seed_plat = self.config.seed_platform
        self.graph = self.platform_graphs.get(seed_plat)
        for node in self.nodes:
            if seed_plat in node.platform_connections:
                node.connections = node.platform_connections[seed_plat]
                node.edge_weights = node.platform_edge_weights.get(seed_plat, {})
            else:
                node.connections = []
                node.edge_weights = {}

        # Phase 3: Apply literacy placement (herd immunity analysis)
        if (self.config.literacy_placement_strategy is not None
                and self.config.literacy_placement_pct > 0):
            _apply_literacy_placement(
                self.nodes,
                self.config.literacy_placement_strategy,
                self.config.literacy_placement_pct,
                self.config.literacy_placement_topic,
                self.platform_graphs,
                self.config.seed_platform,
                self.rng,
            )

        # Phase 3: Surgical counterfactual — immunize bridge nodes per-run
        if self.config.remove_top_n_bridges > 0:
            G = self.platform_graphs.get(seed_plat)
            if G is not None and G.number_of_nodes() > 0:
                bc = nx.betweenness_centrality(G)
                # Exclude bots and fact-checkers from bridge removal
                candidates = [
                    (nid, score) for nid, score in bc.items()
                    if self.nodes[nid].agent_type not in (AgentType.BOT, AgentType.FACT_CHECKER)
                ]
                candidates.sort(key=lambda x: -x[1])
                for nid, _ in candidates[:self.config.remove_top_n_bridges]:
                    self.nodes[nid].status = NodeStatus.IMMUNE
        if self.config.block_first_influencer:
            # Immunize the highest-degree influencer on the seed platform
            influencers = [
                n for n in self.nodes
                if n.agent_type == AgentType.INFLUENCER
                and seed_plat in n.platforms
                and n.status != NodeStatus.IMMUNE
            ]
            if influencers:
                # Highest degree = most connected = most likely to be infected first
                best = max(influencers,
                           key=lambda n: len(n.platform_connections.get(seed_plat, [])))
                best.status = NodeStatus.IMMUNE

        # Create seed rumor
        seed_rumor = create_seed_rumor(self.config, self.rng)
        self.rumor_versions[0] = seed_rumor

        # Phase 3.6 Fix A2: Reddit karma cold start boost.
        # karma_score=1 means only direct connections see it; seed gets rejected
        # by all, karma goes negative, rumor dies. Boost to 3 so it starts in
        # the "direct + some community" visibility tier.
        if seed_plat == Platform.REDDIT:
            seed_rumor.karma_score = 3

        # Select patient zero (must be on seed platform)
        patient_zero = self._select_patient_zero()
        seed_rumor.origin_node = patient_zero.id

        # Infect patient zero
        patient_zero.status = NodeStatus.BELIEVING
        patient_zero.infected_at = 0.0
        patient_zero.infected_on_platform = seed_plat
        patient_zero.rumor_version = 0
        self.last_infection_time = 0.0

        self._log_event(0.0, "rumor_seeded", patient_zero.id, platform=seed_plat, details={
            "scenario": self.config.scenario,
            "platform": seed_plat.value,
        })

        # Schedule patient zero's shares on the seed platform
        self._schedule_shares(patient_zero, seed_rumor, platform=seed_plat, is_seed=True)

        # Schedule periodic termination checks
        self._schedule_event(
            self.config.termination_check_interval,
            EventType.CHECK_TERMINATION,
            0
        )

        # Schedule checkpoints
        for ct in CHECKPOINT_TIMES:
            if ct <= self.config.max_time:
                self._schedule_event(ct, EventType.CHECKPOINT, 0, {"checkpoint_time": ct})

        # Schedule attention recovery events (every simulated hour)
        self._schedule_event(3600.0, EventType.ATTENTION_RECOVERY, 0)

        # Item 19: Emergency correction injection (if configured)
        if self.config.correction_injection_time is not None:
            self._schedule_event(
                self.config.correction_injection_time,
                EventType.EMERGENCY_CORRECTION,
                0,
            )

        # Item 32: Form bot clusters and schedule credibility ticks
        self._setup_bot_clusters()

        # Item 35: Schedule crisis event (if enabled)
        # BUG 12 FIX: Support explicit crisis_time/duration/intensity for testing
        if self.config.crisis_enabled:
            crisis_time = (self.config.crisis_time if self.config.crisis_time is not None
                           else float(self.rng.uniform(6 * 3600, 36 * 3600)))
            crisis_time = min(crisis_time, self.config.max_time - 3600)
            crisis_intensity = (self.config.crisis_intensity if self.config.crisis_intensity is not None
                                else float(self.rng.uniform(0.3, 0.8)))
            crisis_duration = (self.config.crisis_duration if self.config.crisis_duration is not None
                               else float(self.rng.exponential(4 * 3600)))
            self._schedule_event(crisis_time, EventType.CRISIS_START, 0, {
                "intensity": crisis_intensity,
                "duration": crisis_duration,
                "crisis_topic": self.config.scenario,
            })

        # Item 34: Schedule periodic super spreader checks
        # ISSUE D FIX: Reduced check interval from 300s to 60s so fast platforms
        # (Twitter, mu=30s service time) get checked before termination.
        # The 5-minute lookback window for recent shares is unchanged.
        self._schedule_event(60.0, EventType.SUPER_SPREADER_CHECK, 0)

    def _setup_bot_clusters(self):
        """Item 32: Assign bots to coordinated clusters."""
        bots = [n for n in self.nodes if n.agent_type == AgentType.BOT]
        if not bots:
            return

        num_clusters = int(self.rng.integers(1, 4))  # 1-3 clusters
        for bot in bots:
            bot.bot_cluster_id = int(self.rng.integers(0, num_clusters))

        # Build cluster lookup
        self.bot_clusters = {}
        for bot in bots:
            cid = bot.bot_cluster_id
            if cid not in self.bot_clusters:
                self.bot_clusters[cid] = []
            self.bot_clusters[cid].append(bot.id)

        # Initialize wave state
        for cid in self.bot_clusters:
            self.bot_wave_active[cid] = False

        # Schedule hourly bot credibility tick
        self._schedule_event(3600.0, EventType.BOT_CREDIBILITY_TICK, 0)

    def _select_patient_zero(self) -> Node:
        """Select the seed node based on persona config. Must be on seed_platform."""
        persona_config = SEED_PERSONA_CONFIG[self.config.seed_persona]
        target_type = persona_config["maps_to"]
        cred_mod = persona_config["credibility_modifier"]
        seed_plat = self.config.seed_platform
        is_isolated = len(self.config.active_platforms) == 1

        # Find nodes matching the target agent type AND on the seed platform
        # Phase 3.6 Fix A1: Exclude fact-checkers from seed candidates —
        # in isolated sims with few neighbors, a fact-checker neighbor
        # kills the chain before it can grow.
        candidates = [
            n for n in self.nodes
            if n.agent_type == target_type
            and seed_plat in n.platforms
            and n.agent_type != AgentType.FACT_CHECKER
        ]
        if not candidates:
            candidates = [
                n for n in self.nodes
                if seed_plat in n.platforms
                and n.agent_type != AgentType.FACT_CHECKER
            ]
        if not candidates:
            candidates = [n for n in self.nodes if seed_plat in n.platforms]
        if not candidates:
            candidates = self.nodes

        # Filter to nodes with connections — require ≥3 for isolated sims
        min_connections = 3 if is_isolated else 1
        connected = [
            n for n in candidates
            if len(n.platform_connections.get(seed_plat, [])) >= min_connections
        ]
        # Fallback: relax to ≥1 connection
        if not connected:
            connected = [
                n for n in candidates
                if len(n.platform_connections.get(seed_plat, [])) > 0
            ]
        pool = connected if connected else candidates

        # Phase 3.6 Fix A1b: In isolated sims, prefer seeds whose neighbors
        # are NOT dominated by fact-checkers. A seed with 3/7 FC neighbors
        # will have the chain killed before it can grow.
        if is_isolated and len(pool) > 1:
            def _fc_neighbor_ratio(node):
                nbrs = node.platform_connections.get(seed_plat, [])
                if not nbrs:
                    return 1.0
                fc_count = sum(1 for nid in nbrs if self.nodes[nid].agent_type == AgentType.FACT_CHECKER)
                return fc_count / len(nbrs)
            # Keep only nodes where FC ratio < 30% of neighbors
            good_seeds = [n for n in pool if _fc_neighbor_ratio(n) < 0.3]
            if good_seeds:
                pool = good_seeds

        # ISSUE 2 FIX: Use persona-specific seed offset so different personas
        # with the same maps_to type select DIFFERENT seed nodes.
        # Use sum of ordinals for a deterministic offset (not hash(), which
        # is randomized by PYTHONHASHSEED across processes).
        persona_offset = sum(ord(c) for c in self.config.seed_persona.value)
        persona_rng = random.Random(
            (self.config.master_seed or 0) + persona_offset
        )
        return persona_rng.choice(pool)

    def run(self) -> SimulationResult:
        """Run the full simulation until termination."""
        self.setup()

        while self.event_queue and self.current_time < self.config.max_time:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time

            if self.current_time > self.config.max_time:
                break

            # Phase 3: Record detailed snapshot every 60s (only when enabled)
            if (self.config.detailed_tracking
                    and self.current_time >= self._detailed_next_sample_time):
                self._record_detailed_snapshot()
                self._detailed_next_sample_time += 60.0

            self._process_event(event)

        return self._compile_results()

    def _record_detailed_snapshot(self):
        """Phase 3: Sample detailed metrics at the current simulation time."""
        t = self.current_time
        tl = self._detailed_timelines
        tl["time"].append(t)

        # Queue length (events pending per non-removed node, avg)
        active_nodes = sum(1 for n in self.nodes if n.status != NodeStatus.REMOVED)
        queue_per_node = len(self.event_queue) / max(active_nodes, 1)
        tl["queue_length_avg"].append(queue_per_node)

        # Attention budgets
        all_budgets = [n.attention_budget for n in self.nodes
                       if n.status != NodeStatus.REMOVED]
        tl["attention_budget_mean_all"].append(
            float(np.mean(all_budgets)) if all_budgets else 1.0)

        # Hub nodes: influencers + nodes with > 30 connections on seed platform
        seed_plat = self.config.seed_platform
        hub_budgets = [n.attention_budget for n in self.nodes
                       if n.status != NodeStatus.REMOVED
                       and (n.agent_type == AgentType.INFLUENCER
                            or len(n.platform_connections.get(seed_plat, [])) > 30)]
        tl["attention_budget_mean_hubs"].append(
            float(np.mean(hub_budgets)) if hub_budgets else 1.0)
        tl["attention_budget_p10_hubs"].append(
            float(np.percentile(hub_budgets, 10)) if len(hub_budgets) >= 2 else (hub_budgets[0] if hub_budgets else 1.0))

        # Emotional susceptibility
        fears = [n.susceptibility.fear for n in self.nodes
                 if n.status != NodeStatus.REMOVED]
        outrages = [n.susceptibility.outrage for n in self.nodes
                    if n.status != NodeStatus.REMOVED]
        tl["fear_mean"].append(float(np.mean(fears)) if fears else 0.0)
        tl["outrage_susceptibility_mean"].append(
            float(np.mean(outrages)) if outrages else 0.0)

        # Bot survival
        total_bots = sum(1 for n in self.nodes if n.agent_type == AgentType.BOT)
        alive_bots = sum(1 for n in self.nodes
                         if n.agent_type == AgentType.BOT and not n.detected)
        tl["bot_survival_fraction"].append(
            alive_bots / total_bots if total_bots > 0 else 1.0)

        # Phase 3.6: Per-platform bot survival
        bot_surv_plat = {}
        for plat in self.config.active_platforms:
            bots_on = [n for n in self.nodes if n.agent_type == AgentType.BOT and plat in n.platforms]
            alive_on = [n for n in bots_on if not n.detected]
            bot_surv_plat[plat] = len(alive_on) / len(bots_on) if bots_on else 1.0
        tl.setdefault("bot_survival_per_platform", []).append(bot_surv_plat)

        # Cumulative rewiring
        tl["cumulative_unfollows"].append(self._total_unfollows)
        tl["cumulative_seeks"].append(self._total_seeks)

        # Per-chamber infection rates
        chamber_rates = {}
        for cidx in range(self.config.num_echo_chambers):
            ch_nodes = [n for n in self.nodes if n.echo_chamber_idx == cidx]
            if ch_nodes:
                ch_infected = sum(1 for n in ch_nodes
                                  if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER))
                chamber_rates[cidx] = ch_infected / len(ch_nodes)
        tl["infection_per_chamber"].append(chamber_rates)

        # Per-platform infection rates
        plat_rates = {}
        for plat in self.config.active_platforms:
            plat_nodes = [n for n in self.nodes if plat in n.platforms]
            if plat_nodes:
                pi = sum(1 for n in plat_nodes
                         if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER))
                plat_rates[plat] = pi / len(plat_nodes)
        tl["infection_per_platform"].append(plat_rates)

        # Infections since last snapshot (for time-of-day histogram)
        tl["infections_this_period"].append(self._detailed_infections_since_last)
        self._detailed_infections_since_last = 0

        # Fix 6B: Per-status fractions for Graph #1
        total = len(self.nodes)
        tl["believing_frac"].append(
            sum(1 for n in self.nodes if n.status == NodeStatus.BELIEVING) / total)
        tl["silent_believer_frac"].append(
            sum(1 for n in self.nodes if n.status == NodeStatus.SILENT_BELIEVER) / total)
        tl["corrected_frac"].append(
            sum(1 for n in self.nodes if n.status == NodeStatus.CORRECTED) / total)
        tl["total_infected_frac"].append(self._compute_infection_rate())

        # Fix 8B: Per-platform utilization approximation
        util = {}
        window = 60.0
        cutoff = t - window
        for plat in self.config.active_platforms:
            plat_cfg = PLATFORM_CONFIG.get(plat, {})
            base_mu = plat_cfg.get("base_service_time", 30.0)
            # Count share events on this platform in the window
            recent_events = sum(
                1 for evt in self.event_log[-500:]
                if evt.platform == plat
                and evt.time >= cutoff
                and evt.event_type in ("node_infected", "rumor_rejected")
            )
            nodes_on_plat = sum(1 for n in self.nodes if plat in n.platforms)
            if nodes_on_plat > 0:
                util[plat] = min(1.0, (recent_events * base_mu) / (window * nodes_on_plat))
            else:
                util[plat] = 0.0
        tl["utilization_per_platform"].append(util)

        # Fix 23-Prep: Track all 5 emotion dimension means
        for dim in ["fear", "outrage", "humor", "curiosity", "urgency"]:
            vals = [getattr(n.susceptibility, dim) for n in self.nodes
                    if n.status != NodeStatus.REMOVED]
            tl.setdefault(f"{dim}_susceptibility_mean", []).append(
                float(np.mean(vals)) if vals else 0.0)

    def _process_event(self, event: SimEvent):
        """Route event to appropriate handler."""
        handlers = {
            EventType.PROCESS_MESSAGE: self._handle_process_message,
            EventType.SHARE_RUMOR: self._handle_share_rumor,
            EventType.GENERATE_CORRECTION: self._handle_generate_correction,
            EventType.SHARE_CORRECTION: self._handle_share_correction,
            EventType.ATTENTION_RECOVERY: self._handle_attention_recovery,
            EventType.CHECK_TERMINATION: self._handle_check_termination,
            EventType.CHECKPOINT: self._handle_checkpoint,
            # Phase 2 events
            EventType.WHATSAPP_SELF_CORRECTION: self._handle_whatsapp_self_correction,
            EventType.REDDIT_MOD_ACTION: self._handle_reddit_mod_action,
            EventType.STORY_EXPIRY_CHECK: self._handle_story_expiry_check,
            EventType.PLATFORM_HOP: self._handle_platform_hop,
            EventType.TWITTER_COMMUNITY_NOTE: self._handle_twitter_community_note,
            EventType.EMERGENCY_CORRECTION: self._handle_emergency_correction,
            # Group E events
            EventType.SUPER_SPREADER_CHECK: self._handle_super_spreader_check,
            EventType.CRISIS_START: self._handle_crisis_start,
            EventType.CRISIS_END: self._handle_crisis_end,
            EventType.BOT_WAVE: self._handle_bot_wave,
            EventType.BOT_CREDIBILITY_TICK: self._handle_bot_credibility_tick,
        }
        handler = handlers.get(event.event_type)
        if handler:
            handler(event)

    # -------------------------------------------------------------------------
    # Event Scheduling
    # -------------------------------------------------------------------------

    def _schedule_event(self, time: float, event_type: EventType, node_id: int,
                        data: dict = None, platform: Platform = None):
        """Add an event to the priority queue."""
        event = SimEvent(
            time=time, event_type=event_type, node_id=node_id,
            data=data or {}, platform=platform,
        )
        heapq.heappush(self.event_queue, event)

    def _get_node_connections_on_platform(self, node: Node, platform: Platform) -> list[int]:
        """Get a node's connections on a specific platform."""
        return node.platform_connections.get(platform, [])

    def _get_edge_weight_on_platform(self, node: Node, target_id: int, platform: Platform) -> float:
        """Get edge weight between node and target on a specific platform."""
        return node.platform_edge_weights.get(platform, {}).get(target_id, 0.5)

    def _schedule_shares(self, sender: Node, rumor: Rumor,
                         platform: Platform = None, is_seed: bool = False,
                         backfire: float = 0.0, forward_depth: int = 0):
        """
        Schedule share events from sender to all eligible connections on a platform.

        Item 1: effective_service_time = platform_base_service_time * agent_type_modifier
        """
        if platform is None:
            platform = self.config.seed_platform

        platform_config = PLATFORM_CONFIG[platform]
        base_service_time = platform_config["base_service_time"]  # Item 1
        agent_modifier = AGENT_TYPE_CONFIG[sender.agent_type]["service_time_modifier"]  # Item 1
        effective_mu = base_service_time * agent_modifier  # Item 1

        connections = self._get_node_connections_on_platform(sender, platform)

        for target_id in connections:
            target = self.nodes[target_id]
            if target.status == NodeStatus.REMOVED:
                continue

            edge_weight = self._get_edge_weight_on_platform(sender, target_id, platform)

            # Selective sharing per section 4.3: share if virality > (1 - edge_weight)
            # ISSUE 6 FIX: Deterministic gate — text content shares through strong ties
            # only, high-virality content shares through all ties.
            share_threshold = 1 - edge_weight
            if rumor.virality_score < share_threshold:
                continue  # don't share through weak ties for low-virality content

            # Service time: exponential with effective_mu (Item 1)
            delay = float(self.rng.exponential(effective_mu))

            # Apply seed persona credibility modifier to first hop
            cred_modifier = 1.0
            if is_seed:
                persona_config = SEED_PERSONA_CONFIG[self.config.seed_persona]
                cred_modifier = persona_config["credibility_modifier"]

            self._schedule_event(
                self.current_time + delay,
                EventType.SHARE_RUMOR,
                target_id,
                {
                    "sender_id": sender.id,
                    "rumor_version": rumor.version,
                    "edge_weight": edge_weight,
                    "credibility_modifier": cred_modifier,
                    "backfire": backfire,  # Items 23-24
                    "forward_depth": forward_depth,  # ISSUE C: per-chain depth
                },
                platform=platform,
            )

    # -------------------------------------------------------------------------
    # Event Handlers
    # -------------------------------------------------------------------------

    def _handle_share_rumor(self, event: SimEvent):
        """A rumor arrives at a node — decide believe/reject."""
        target = self.nodes[event.node_id]
        data = event.data
        sender_id = data["sender_id"]
        sender = self.nodes[sender_id]
        rumor_version = data["rumor_version"]
        edge_weight = data["edge_weight"]
        cred_modifier = data.get("credibility_modifier", 1.0)
        backfire = data.get("backfire", 0.0)  # Items 23-24
        forward_depth = data.get("forward_depth", 0)  # ISSUE C: per-chain depth
        event_platform = event.platform or self.config.seed_platform

        if rumor_version not in self.rumor_versions:
            return

        rumor = self.rumor_versions[rumor_version]

        # --- Platform-specific pre-checks (Items 10, 11, 12) ---

        # Item 11: Instagram story expiry — expired stories can't spread
        if event_platform == Platform.INSTAGRAM and rumor.expiry_time is not None:
            if self.current_time > rumor.expiry_time:
                return  # story expired, no new infections

            # Gradual decay: last 25% of TTL, reach fades linearly
            story_ttl = PLATFORM_CONFIG[Platform.INSTAGRAM]["story_ttl"]
            remaining_life = (rumor.expiry_time - self.current_time) / story_ttl
            if remaining_life < 0.25:
                reach_modifier = remaining_life / 0.25
                if self.rng.random() > reach_modifier:
                    return  # story fading, fewer people see it

        # Item 12: Reddit — quarantined/removed posts can't spread
        if event_platform == Platform.REDDIT:
            if rumor.quarantined or rumor.removed_by_mod:
                return
            # Reddit karma visibility tiers per spec §3.5.2
            if rumor.karma_score < 0:
                # Collapsed/hidden — only 10% visibility
                if self.rng.random() > 0.1:
                    return
            elif rumor.karma_score < 5:
                pass  # visible to direct connections only (normal behavior)
            elif rumor.karma_score < 20:
                pass  # visible to entire community (rising) — normal graph traversal covers this
            elif rumor.karma_score >= 50:
                # Cross-community visibility: equivalent to algorithmic amplification
                self._check_algorithmic_amplification(sender, rumor, platform=Platform.REDDIT)
            # karma 20-49: visible to adjacent communities + algorithmic boost (handled via normal spread)

        # Item 17: Twitter community note — if active, 50% auto-correction for viewers
        if (event_platform == Platform.TWITTER
                and self.correction_active
                and hasattr(rumor, '_community_note')
                and rumor._community_note):
            if self.rng.random() < 0.5:
                # Auto-corrected by community note
                if target.status == NodeStatus.UNAWARE:
                    # Pre-bunking from community note
                    target.effective_threshold *= 1.2
                    target.credibility_threshold *= 1.2
                return  # viewer sees note, doesn't believe

        # ISSUE 9 FIX: Track emotional fatigue for ALL nodes receiving high-emotion
        # messages, even if already believing. Nodes still experience emotional load
        # from continued exposure, which drives fatigue accumulation for hub nodes.
        if target.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER,
                              NodeStatus.CORRECTED, NodeStatus.IMMUNE, NodeStatus.REMOVED):
            # Still update emotional state (fatigue counters) for already-processed nodes
            self._update_emotional_state(target, rumor)
            return

        # Time of day check — is target active?
        current_hour = int((self.config.start_hour + self.current_time / 3600) % 24)
        activity_prob = target.active_hours_profile[current_hour]
        if self.rng.random() > activity_prob:
            retries = data.get("_retries", 0)
            if retries < 3:
                retry_delay = float(self.rng.exponential(1800))
                retry_data = dict(data)
                retry_data["_retries"] = retries + 1
                self._schedule_event(
                    self.current_time + retry_delay,
                    EventType.SHARE_RUMOR,
                    event.node_id,
                    retry_data,
                    platform=event_platform,
                )
            return

        # Attention budget check
        # attention_budget_enabled = structural toggle; attention_budget_toggle = sensitivity toggle
        _attn_active = self.config.attention_budget_enabled and self.config.attention_budget_toggle
        if _attn_active and target.attention_budget < 0.5:
            p_skip = 1.0 - (target.attention_budget / 0.5)
            if self.rng.random() < p_skip:
                self._log_event(self.current_time, "attention_skipped", target.id, sender_id,
                                platform=event_platform)
                return

        if _attn_active:
            target.attention_budget = max(0.0, target.attention_budget - self.config.attention_cost)

        target.times_exposed += 1

        # Item 26: Update emotional priming & fatigue before threshold computation
        self._update_emotional_state(target, rumor)

        # Item 22: Degrade literacy under sustained pressure
        self._degrade_literacy(target, rumor, sender, edge_weight)

        # --- Compute effective threshold ---
        effective_thresh = target.credibility_threshold

        # Item 21: Literacy as threshold boost (topic-specific)
        literacy_boost = self._apply_literacy_boost(target, rumor)
        effective_thresh *= literacy_boost

        # Item 3: Apply platform base credibility modifier
        platform_threshold_mod = PLATFORM_CONFIG[event_platform]["threshold_modifier"]
        effective_thresh *= platform_threshold_mod

        # Item 10: WhatsApp forwarded tag penalty (ISSUE C: per-chain depth)
        # Penalty applies when this message has been forwarded >= 5 times in its chain
        if event_platform == Platform.WHATSAPP and forward_depth >= 5:
            effective_thresh *= 1.3

        # Item 15: Scenario-platform affinity modifier
        scenario_config = RUMOR_SCENARIOS.get(rumor.scenario, {})
        platform_affinity = scenario_config.get("platform_affinity", {}).get(event_platform, 0.5)
        affinity_mod = 1.0 + (0.5 - platform_affinity) * 0.5
        effective_thresh *= affinity_mod

        # Spec §3.2: WhatsApp forwarded_tag credibility penalty.
        # Tagged messages ("Forwarded many times") get threshold *= 1.3 (harder to believe).
        if event_platform == Platform.WHATSAPP and rumor.forwarded_tag:
            effective_thresh *= 1.3

        if target.agent_type == AgentType.FACT_CHECKER:
            if cred_modifier != 1.0:
                effective_thresh *= (1.0 / cred_modifier)
            effective_thresh = float(np.clip(effective_thresh, 0.85, 0.999))
            target.effective_threshold = effective_thresh
        else:
            # 1. Trust decay
            effective_thresh *= (self.config.decay_rate ** target.times_exposed)

            # 2. Confirmation bias (4D worldview alignment, Item 25)
            alignment_score = self._compute_alignment(target, rumor)
            exposure_scaling = min(1.0, target.times_exposed / 3.0)
            bias_modifier = alignment_score * self.config.bias_strength * exposure_scaling
            effective_thresh *= (1 - bias_modifier)

            # 3. Seed persona credibility modifier (first hop only)
            if cred_modifier != 1.0:
                effective_thresh *= (1.0 / cred_modifier)

            # Items 23-24: Backfire — high-cred/high-literacy sender lowers receiver threshold
            if backfire > 0:
                effective_thresh *= (1 - backfire)

            # Fix 5C: High-literacy nodes get a hard threshold floor
            # MUST be after all modifiers (trust decay, bias, backfire) — see plan note
            literacy_val = getattr(target.literacy_vector, rumor.scenario, 0.5)
            if literacy_val > 0.7:
                literacy_floor = literacy_val * 0.4  # literacy 0.9 -> floor 0.36
                effective_thresh = max(effective_thresh, literacy_floor)

            effective_thresh = float(np.clip(effective_thresh, 0.05, 0.999))
            target.effective_threshold = effective_thresh
        target.effective_threshold = effective_thresh

        # --- Bernoulli share decision ---
        source_trust = self._get_sender_trust_modifier(sender, edge_weight, receiver=target)
        emotional_impact = rumor.emotions.dot(target.susceptibility)

        rumor_credibility = rumor.virality_score * (1 + emotional_impact * self.config.emotion_weight)
        rumor_credibility *= (source_trust / 1.5)

        if rumor_credibility > effective_thresh:
            excess_ratio = min(1.0, (rumor_credibility - effective_thresh) / max(effective_thresh, 0.1))
            believe_probability = 0.25 + 0.45 * excess_ratio
        else:
            ratio = rumor_credibility / max(effective_thresh, 0.01)
            believe_probability = ratio * 0.12

        believe_probability = float(np.clip(believe_probability, 0.01, 0.90))

        # Item 29: Rumor framing modifier — persuasive framing boosts believe_probability
        if self.config.framing_bonus_enabled:
            rumor_shape = self._select_rumor_shape(sender, rumor, edge_weight)
            framing_mod = RUMOR_FRAMING_MODIFIERS[rumor_shape]
            believe_probability *= (1 + framing_mod)
            believe_probability = min(believe_probability, 0.95)  # hard cap

        # Item 30: Demographic sharing modifier per spec §2.6
        # sharing_modifier = 1.0 + (0.3 * (1 - digital_nativity))
        # Low DN = share more without verifying; high DN = baseline
        sharing_modifier = 1.0 + (0.3 * (1 - target.digital_nativity))
        believe_probability *= sharing_modifier
        believe_probability = min(believe_probability, 0.95)

        # Phase 3: Global sharing probability modifier (sensitivity analysis)
        if self.config.sharing_probability_modifier != 1.0:
            believe_probability *= self.config.sharing_probability_modifier
            believe_probability = min(believe_probability, 0.95)

        # Item 31: Track rumor source count for reactive rewiring
        target.rumor_sources[sender_id] = target.rumor_sources.get(sender_id, 0) + 1

        # Item 32: Bot detection per share event
        if (self.config.bot_detection_enabled
                and sender.agent_type == AgentType.BOT and not sender.detected):
            self._check_bot_detection(sender, event_platform)
            if sender.detected:
                return  # bot was just detected, message blocked

        if self.rng.random() < believe_probability:
            # BUG 2 FIX: Pass amplified flag so _infect_node tracks organic vs amp shares
            self._current_share_amplified = data.get("_amplified", False)
            self._infect_node(target, sender, rumor, edge_weight, event_platform,
                              forward_depth=forward_depth)
            # Item 12: Reddit karma +1 on believe
            if event_platform == Platform.REDDIT:
                rumor.karma_score += 1
                self._check_reddit_moderator(rumor, event_platform)

            # Item 31: Seek mechanic — believing + high emotion → add echo chamber connections
            if self.config.rewiring_enabled:
                emotional_impact = rumor.emotions.dot(target.susceptibility)
                if emotional_impact > 0.7 and target.rewiring_events < 3:
                    if self.rng.random() < 0.10:
                        self._reactive_seek(target, event_platform)
        else:
            # Item 12: Reddit karma -1 on reject
            if event_platform == Platform.REDDIT:
                rumor.karma_score -= 1

            # Spec §4.13: Unfollow mechanic — 2+ messages from same source
            # (Spec says 3+ rumor messages, but in practice each node shares once,
            # so 2+ total messages including corrections is the functional equivalent)
            if (self.config.rewiring_enabled
                    and target.rumor_sources.get(sender_id, 0) >= 2
                    and target.rewiring_events < 3
                    and target.status in (NodeStatus.CORRECTED, NodeStatus.UNAWARE)):
                if self.rng.random() < 0.15:
                    self._reactive_unfollow(target, sender, event_platform)

            self._log_event(self.current_time, "rumor_rejected", target.id, sender_id,
                            platform=event_platform, details={
                "threshold": effective_thresh, "believe_prob": believe_probability,
            })

            if (target.agent_type == AgentType.FACT_CHECKER
                    and self.config.correction_enabled):
                target.status = NodeStatus.IMMUNE

                if not self.correction_active:
                    if self.config.correction_delay_override is not None:
                        # Phase 3: Sensitivity analysis override
                        correction_delay = self.config.correction_delay_override
                    else:
                        # ISSUE A FIX: On platforms with algo amp, FCs react faster
                        # to viral content. Halve delay (mu=7.5min vs 15min).
                        fc_mu = 900  # default: Exp(mu=15 min)
                        plat_cfg = PLATFORM_CONFIG.get(event_platform, {})
                        if plat_cfg.get("algorithmic_amplification", False):
                            fc_mu = 450  # halved for algo-amp platforms
                        # Spec §6.4.5: During crisis, FC correction delay increased by 2x
                        if self.crisis_active:
                            fc_mu *= 2
                        correction_delay = float(self.rng.exponential(fc_mu))
                    self._schedule_event(
                        self.current_time + correction_delay,
                        EventType.GENERATE_CORRECTION,
                        target.id,
                        {"rumor_version": rumor_version},
                        platform=event_platform,
                    )
                    self._log_event(self.current_time, "fact_checker_investigating", target.id,
                                    platform=event_platform, details={
                        "correction_eta": self.current_time + correction_delay,
                    })
                else:
                    self._schedule_correction_shares(target, platform=event_platform)
                    self._log_event(self.current_time, "fact_checker_relaying", target.id,
                                    platform=event_platform)

    def _infect_node(self, target: Node, sender: Node, rumor: Rumor,
                     edge_weight: float, platform: Platform = None,
                     forward_depth: int = 0):
        """Target node believes the rumor."""
        if platform is None:
            platform = self.config.seed_platform

        # BUG 11 FIX: Fact-checkers should NEVER be infected — they become IMMUNE
        if target.agent_type == AgentType.FACT_CHECKER:
            return

        # Silent believer vs active sharer
        if self.rng.random() < self.config.silent_believer_probability:
            target.status = NodeStatus.SILENT_BELIEVER
        else:
            target.status = NodeStatus.BELIEVING

        target.infected_by = sender.id
        target.infected_at = self.current_time
        target.infected_on_platform = platform
        target.rumor_version = rumor.version
        sender.downstream_infections += 1
        rumor.total_infections += 1
        self.last_infection_time = self.current_time
        self._detailed_infections_since_last += 1  # Phase 3: count for time-of-day graph

        self._log_event(self.current_time, "node_infected", target.id, sender.id,
                        platform=platform, details={
            "status": target.status.value,
            "rumor_version": rumor.version,
            "edge_weight": edge_weight,
        })

        # Record timeline point
        infection_rate = self._compute_infection_rate()
        self.infection_timeline.append((self.current_time, infection_rate))

        # Active sharers spread the rumor on the SAME platform they got it on
        if target.status == NodeStatus.BELIEVING:
            current_rumor = rumor
            if self.rng.random() < self.config.mutation_probability:
                current_rumor = self._mutate_rumor(rumor, target, platform)

            current_rumor.forward_count += 1  # global stat tracking

            # Spec §3.2: forward_count lives on the RUMOR OBJECT. Each forward
            # increments it. forwarded_tag = True when forward_count >= 5.
            # "Forwarded many times" label makes the message harder to believe.
            chain_depth = forward_depth + 1  # per-chain tracking for other uses
            is_deeply_forwarded = (platform == Platform.WHATSAPP
                                   and current_rumor.forward_count >= 5)

            # Update Rumor.forwarded_tag for reporting
            if is_deeply_forwarded and not current_rumor.forwarded_tag:
                current_rumor.forwarded_tag = True

            # Item 10: WhatsApp self-correction on forwarded messages
            # 5% chance node independently searches and self-corrects.
            # ISSUE F NOTE: In multi-platform, self-corrections may be rare because:
            # 1. WhatsApp chains need depth >= 5 (per-chain, not global)
            # 2. Multi-platform runs split nodes across platforms, so WA chains
            #    are shorter. This is expected behavior — fewer deep WA chains
            #    means fewer forwarded tags and fewer self-corrections.
            if (platform == Platform.WHATSAPP
                    and is_deeply_forwarded
                    and self.rng.random() < 0.05):
                self_correction_delay = float(self.rng.exponential(1800))  # mu=30min
                self._schedule_event(
                    self.current_time + self_correction_delay,
                    EventType.WHATSAPP_SELF_CORRECTION,
                    target.id,
                    {"rumor_version": current_rumor.version},
                    platform=platform,
                )

            # Item 11: Instagram story TTL — set expiry when first shared on Instagram
            if (platform == Platform.INSTAGRAM
                    and current_rumor.expiry_time is None
                    and current_rumor.media_type in ("reel", "image")):
                story_ttl = PLATFORM_CONFIG[Platform.INSTAGRAM]["story_ttl"]
                current_rumor.expiry_time = self.current_time + story_ttl

            # Items 23-24: Compute backfire effect for this sender's shares.
            # Receivers see a high-credibility/high-literacy sender believing,
            # which makes the rumor more believable (threshold reduction).
            backfire = self._compute_backfire(target, current_rumor)

            # BUG 13 FIX: Log backfire events when they trigger
            if backfire > 0:
                topic = current_rumor.scenario
                sender_lit = target.literacy_vector.get(topic)
                n_receivers = len(self._get_node_connections_on_platform(target, platform))
                self._log_event(self.current_time, "backfire", target.id,
                                platform=platform, details={
                    "agent_type": target.agent_type.value,
                    "literacy": round(sender_lit, 3),
                    "backfire_multiplier": round(backfire, 3),
                    "receivers": n_receivers,
                })

            self._schedule_shares(target, current_rumor, platform=platform,
                                  backfire=backfire, forward_depth=chain_depth)

            # Track shares for algorithmic amplification
            self.total_shares += 1
            self.total_shares_per_platform[platform] += 1
            self.recent_shares.append(self.current_time)

            # BUG 2 FIX: Only count ORGANIC shares (not amp-boosted) toward
            # amplification threshold to prevent cascade loop
            if not self._current_share_amplified:
                self.recent_shares_per_platform[platform].append(self.current_time)
                self._check_algorithmic_amplification(target, current_rumor, platform)

            # Item 16: Platform hopping — if node is on multiple platforms, may screenshot
            # the rumor and share it on another platform
            if len(target.platforms) > 1 and len(self.config.active_platforms) > 1:
                self._check_platform_hop(target, current_rumor, platform)

    def _check_platform_hop(self, node: Node, rumor: Rumor, source_platform: Platform):
        """
        Item 16: Check if a node will hop the rumor to another platform.

        hop_probability = base_hop_rate * virality * emotional_charge * hop_tendency
        Destination selected by topic-weighted platform affinity.
        """
        # Hop probability based on virality, emotional charge, and personal tendency.
        # Calibrated so ~10-20% of multi-platform infected nodes attempt a hop.
        emotional_charge = float(np.mean(rumor.emotions.as_array()))
        hop_prob = 0.5 * rumor.virality_score * emotional_charge * node.hop_tendency

        # Spec §3.4.2: Instagram stories about to expire get screenshotted more urgently
        if (source_platform == Platform.INSTAGRAM
                and rumor.expiry_time is not None
                and rumor.expiry_time > self.current_time):
            story_ttl = PLATFORM_CONFIG[Platform.INSTAGRAM]["story_ttl"]
            remaining_life = (rumor.expiry_time - self.current_time) / story_ttl
            if remaining_life < 1.0:  # only boost if story is decaying
                hop_prob *= (1 + (1 - remaining_life) * 0.5)

        # Clamp to reasonable range
        hop_prob = min(0.4, hop_prob)

        if self.rng.random() >= hop_prob:
            return

        # Select destination platform (topic-weighted)
        other_platforms = [p for p in node.platforms if p != source_platform and p in self.config.active_platforms]
        if not other_platforms:
            return

        # Weight by scenario platform affinity
        scenario_config = RUMOR_SCENARIOS.get(rumor.scenario, {})
        affinities = scenario_config.get("platform_affinity", {})

        weights = []
        for p in other_platforms:
            # Platform fit for media type
            media_idx = MEDIA_TYPES.index(rumor.media_type)
            pf = PLATFORM_FIT_MATRIX[p][media_idx]
            # Topic relevance (from platform_affinity)
            tr = affinities.get(p, 0.5)
            weights.append(pf * tr)

        total_weight = sum(weights)
        if total_weight <= 0:
            return
        weights = [w / total_weight for w in weights]

        dest_idx = self.rng.choice(len(other_platforms), p=weights)
        dest_platform = other_platforms[dest_idx]

        # Schedule the hop with some delay
        hop_delay = float(self.rng.exponential(300))  # avg 5 min delay for screenshots/reshare

        self._schedule_event(
            self.current_time + hop_delay,
            EventType.PLATFORM_HOP,
            node.id,
            {
                "rumor_version": rumor.version,
                "source_platform": source_platform.value,
                "dest_platform": dest_platform.value,
            },
            platform=dest_platform,
        )

    def _mutate_rumor(self, parent: Rumor, mutator: Node, platform: Platform = None) -> Rumor:
        """Create a mutated version of the rumor."""
        if platform is None:
            platform = self.config.seed_platform

        new_version = self.next_rumor_id
        self.next_rumor_id += 1
        self.total_mutations += 1

        # Mutate emotional profile
        parent_emotions = parent.emotions.as_array()
        shifts = self.rng.uniform(-0.15, 0.15, size=5)
        new_emotions_arr = np.clip(parent_emotions + shifts, 0.0, 1.0)
        new_emotions = EmotionVector(
            fear=float(new_emotions_arr[0]),
            outrage=float(new_emotions_arr[1]),
            humor=float(new_emotions_arr[2]),
            curiosity=float(new_emotions_arr[3]),
            urgency=float(new_emotions_arr[4]),
        )

        new_quality = float(np.clip(parent.content_quality + self.rng.uniform(-0.1, 0.1), 0.1, 1.0))

        alignment_shift = self.rng.normal(0, 0.05, size=4)  # 4D worldview per spec §4.7
        new_alignment = parent.alignment_vector + alignment_shift
        new_alignment[0] = np.clip(new_alignment[0], -1.0, 1.0)   # political
        new_alignment[1] = np.clip(new_alignment[1], 0.0, 1.0)    # health_trust
        new_alignment[2] = np.clip(new_alignment[2], 0.0, 1.0)    # tech_trust
        new_alignment[3] = np.clip(new_alignment[3], 0.0, 1.0)    # authority_trust

        # Recompute virality using the platform where mutation happened
        media_idx = MEDIA_TYPES.index(parent.media_type)
        platform_fit = PLATFORM_FIT_MATRIX[platform][media_idx]
        emotional_impact = float(np.mean(new_emotions_arr))
        virality = new_quality * platform_fit * (1 + emotional_impact * 0.3)

        original_emotions = self.rumor_versions[0].emotions.as_array()
        mutation_distance = float(np.linalg.norm(new_emotions_arr - original_emotions))

        new_rumor = Rumor(
            id=new_version,
            version=new_version,
            parent_version=parent.version,
            scenario=parent.scenario,
            media_type=parent.media_type,
            content_quality=new_quality,
            emotions=new_emotions,
            alignment_vector=new_alignment,
            origin_platform=platform,
            origin_node=mutator.id,
            origin_time=self.current_time,
            forward_count=0,  # spec §4.2: mutations reset forward_count to 0
            forwarded_tag=False,   # ISSUE C: forwarded_tag is per-chain (tracked via event data)
            mutation_chain=parent.mutation_chain + [new_version],
            mutation_distance=mutation_distance,
            virality_score=float(np.clip(virality, 0.0, 5.0)),
        )

        self.rumor_versions[new_version] = new_rumor

        self._log_event(self.current_time, "mutation", mutator.id, platform=platform, details={
            "parent_version": parent.version,
            "new_version": new_version,
            "mutation_distance": mutation_distance,
            "emotion_shift": shifts.tolist(),
        })

        return new_rumor

    def _handle_process_message(self, event: SimEvent):
        """Process a message from a node's queue (currently handled inline in share_rumor)."""
        pass  # Processing is handled directly in _handle_share_rumor

    def _handle_generate_correction(self, event: SimEvent):
        """A fact-checker generates a correction."""
        node = self.nodes[event.node_id]
        if node.agent_type != AgentType.FACT_CHECKER:
            return
        if self.correction_active:
            return

        self.correction_active = True
        self.correction_origin_time = self.current_time
        event_platform = event.platform or self.config.seed_platform

        self._log_event(self.current_time, "correction_generated", node.id,
                        platform=event_platform, details={
            "rumor_version": event.data.get("rumor_version", 0),
        })

        # Schedule correction shares on the platform where FC encountered the rumor
        self._schedule_correction_shares(node, platform=event_platform)

        # Item 17: Twitter community note — 60% chance, delay Exp(mu=2h) per spec §4.11
        # Spec §4.11: Trigger when high visibility (karma_score > 30). Since karma
        # only increments on Reddit, we use cumulative Twitter share count as the
        # equivalent engagement metric for triggering community notes on Twitter.
        twitter_shares = self.total_shares_per_platform.get(Platform.TWITTER, 0)
        if (Platform.TWITTER in self.config.active_platforms
                and twitter_shares > 30):
            if self.rng.random() < 0.6:
                note_delay = float(self.rng.exponential(7200))  # mu=2h per spec §4.11
                rumor_ver = event.data.get("rumor_version", 0)
                self._schedule_event(
                    self.current_time + note_delay,
                    EventType.TWITTER_COMMUNITY_NOTE,
                    node.id,
                    {"rumor_version": rumor_ver},
                    platform=Platform.TWITTER,
                )

    def _schedule_correction_shares(self, source: Node, platform: Platform = None):
        """Schedule correction propagation from a node on a specific platform."""
        if platform is None:
            platform = self.config.seed_platform

        platform_config = PLATFORM_CONFIG[platform]
        base_service_time = platform_config["base_service_time"]
        correction_speed = platform_config["correction_speed_modifier"]
        effective_mu = base_service_time / correction_speed

        connections = self._get_node_connections_on_platform(source, platform)

        for target_id in connections:
            if target_id in self.correction_sent_to:
                continue
            target = self.nodes[target_id]
            if target.status in (NodeStatus.REMOVED, NodeStatus.CORRECTED):
                continue

            self.correction_sent_to.add(target_id)
            delay = float(self.rng.exponential(effective_mu))
            self._schedule_event(
                self.current_time + delay,
                EventType.SHARE_CORRECTION,
                target_id,
                {"sender_id": source.id},
                platform=platform,
            )

    def _handle_share_correction(self, event: SimEvent):
        """A correction arrives at a node."""
        target = self.nodes[event.node_id]
        sender_id = event.data["sender_id"]
        event_platform = event.platform or self.config.seed_platform

        if target.status == NodeStatus.REMOVED:
            return

        # BUG 6 FIX: Count correction messages from same source for unfollow tracking
        target.rumor_sources[sender_id] = target.rumor_sources.get(sender_id, 0) + 1

        # Time of day check
        current_hour = int((self.config.start_hour + self.current_time / 3600) % 24)
        activity_prob = target.active_hours_profile[current_hour]
        if self.rng.random() > activity_prob:
            return

        if target.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER):
            # Correction effectiveness per §4.5:
            #   effectiveness = 0.8 * fatigue_rate ^ times_seen_before_this
            fatigue_rate = self.config.correction_fatigue_rate
            if self.rumor_versions:
                rumor = self.rumor_versions.get(target.rumor_version, self.rumor_versions[0])
                alignment = self._compute_alignment(target, rumor)
                if alignment > 0.5:
                    fatigue_rate = 0.5
                else:
                    fatigue_rate = 0.7

            fatigue_exponent = target.times_correction_seen
            target.times_correction_seen += 1
            effectiveness = 0.8 * (fatigue_rate ** fatigue_exponent)

            # Item 29: Correction framing modifier
            sender = self.nodes[sender_id]
            correction_shape = self._select_correction_shape(sender)
            correction_framing = CORRECTION_FRAMING_MODIFIERS[correction_shape]
            effectiveness *= (1 + correction_framing)

            # Item 30: Demographic correction receptivity per spec §2.6
            # correction_receptivity = 0.5 + (0.5 * digital_nativity)
            # High DN = more receptive; low DN = more resistant
            correction_receptivity = 0.5 + (0.5 * target.digital_nativity)
            effectiveness *= correction_receptivity

            effectiveness = min(effectiveness, 0.95)  # hard cap

            if self.rng.random() < effectiveness:
                target.status = NodeStatus.CORRECTED
                self._log_event(self.current_time, "node_corrected", target.id, sender_id,
                                platform=event_platform, details={
                    "effectiveness": effectiveness,
                })

                # Spec §4.13: Check unfollow for ALL sources that sent 2+ messages
                # Now that node is corrected, it may unfollow prolific rumor sources
                for src_id, msg_count in list(target.rumor_sources.items()):
                    if msg_count >= 2 and target.rewiring_events < 3:
                        if self.rng.random() < 0.15:
                            src_node = self.nodes[src_id]
                            self._reactive_unfollow(target, src_node, event_platform)

                if self.rng.random() < 0.3:
                    self._schedule_correction_shares(target, platform=event_platform)
            else:
                self._log_event(self.current_time, "correction_failed", target.id, sender_id,
                                platform=event_platform, details={
                    "effectiveness": effectiveness,
                    "times_seen": target.times_correction_seen,
                })

        elif target.status == NodeStatus.UNAWARE:
            sender = self.nodes[sender_id]
            correction_quality = 0.8 if sender.agent_type == AgentType.FACT_CHECKER else 0.4
            pre_bunk = 0.1 + 0.15 * correction_quality
            target.effective_threshold *= (1 + pre_bunk)
            target.credibility_threshold *= (1 + pre_bunk)

            self._log_event(self.current_time, "pre_bunked", target.id, sender_id,
                            platform=event_platform, details={
                "threshold_boost": pre_bunk,
            })

    def _handle_attention_recovery(self, event: SimEvent):
        """Periodic attention budget recovery for all nodes."""
        current_hour = int((self.config.start_hour + self.current_time / 3600) % 24)
        _attn_active = self.config.attention_budget_enabled and self.config.attention_budget_toggle

        for node in self.nodes:
            if node.status == NodeStatus.REMOVED:
                continue

            if _attn_active:
                activity = node.active_hours_profile[current_hour]
                recovery = self.config.attention_recovery_rate

                # Recovery rate depends on activity level
                if activity < 0.2:
                    recovery *= 1.5  # offline rest = slightly faster recovery
                elif activity > 0.7:
                    recovery *= 0.3  # peak hours = much slower recovery

                # Phase 3.6: Diminishing recovery — depleted nodes recover slower
                # This prevents full recovery after intense exposure periods
                recovery *= node.attention_budget  # budget 0.5 → 50% of base recovery
                node.attention_budget = min(1.0, node.attention_budget + recovery)

            # Item 26: Decay emotional priming (5% per hour)
            self._decay_priming(node)

        # Schedule next recovery
        self._schedule_event(self.current_time + 3600.0, EventType.ATTENTION_RECOVERY, 0)

    def _handle_check_termination(self, event: SimEvent):
        """Check if simulation should terminate early."""
        should_terminate, reason = self._check_termination_conditions()

        if should_terminate:
            # BUG 14 FIX: Record the actual termination time before clearing queue
            self._termination_time = self.current_time
            # Clear event queue to stop simulation
            self.event_queue.clear()
            self._log_event(self.current_time, "termination", 0, details={"reason": reason})
            return

        # Schedule next check
        self._schedule_event(
            self.current_time + self.config.termination_check_interval,
            EventType.CHECK_TERMINATION,
            0,
        )

    def _handle_checkpoint(self, event: SimEvent):
        """Save a checkpoint snapshot."""
        snapshot = self._create_checkpoint()
        self.checkpoints.append(snapshot)

    # -------------------------------------------------------------------------
    # Phase 2 Event Handlers (Items 10, 11, 12)
    # -------------------------------------------------------------------------

    def _handle_whatsapp_self_correction(self, event: SimEvent):
        """
        Item 10: A WhatsApp user who received a forwarded message independently
        searches for the truth and self-corrects.
        correction_quality = 0.3 (self-sourced, lower than FC).
        """
        target = self.nodes[event.node_id]
        if target.status not in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER):
            return

        # Self-correction effectiveness is lower than FC-sourced
        effectiveness = 0.3  # self-correction quality
        if self.rng.random() < effectiveness:
            target.status = NodeStatus.CORRECTED
            self._log_event(self.current_time, "whatsapp_self_correction", target.id,
                            platform=Platform.WHATSAPP, details={
                "effectiveness": effectiveness,
            })

    def _handle_reddit_mod_action(self, event: SimEvent):
        """
        Item 12: Reddit moderator takes action on a rumor post.
        Actions: remove (60%), pin correction (30%), quarantine (10%).
        """
        rumor_version = event.data.get("rumor_version", 0)
        if rumor_version not in self.rumor_versions:
            return

        rumor = self.rumor_versions[rumor_version]
        if rumor.removed_by_mod or rumor.quarantined:
            return  # already handled

        mod_id = event.node_id
        roll = self.rng.random()

        if roll < 0.60:
            # Option A: Remove post (60%)
            rumor.removed_by_mod = True
            rumor.karma_score = -999
            self._log_event(self.current_time, "reddit_mod_remove", mod_id,
                            platform=Platform.REDDIT, details={
                "rumor_version": rumor_version,
            })

        elif roll < 0.90:
            # Option B: Pin correction (30%) — 2x effectiveness correction to community
            self._log_event(self.current_time, "reddit_mod_pin_correction", mod_id,
                            platform=Platform.REDDIT, details={
                "rumor_version": rumor_version,
            })
            # Schedule corrections to all mod's connections with 2x effectiveness
            self._schedule_correction_shares(self.nodes[mod_id], platform=Platform.REDDIT)

        else:
            # Option C: Quarantine thread (10%)
            rumor.quarantined = True
            self._log_event(self.current_time, "reddit_mod_quarantine", mod_id,
                            platform=Platform.REDDIT, details={
                "rumor_version": rumor_version,
            })

    def _check_reddit_moderator(self, rumor: Rumor, platform: Platform):
        """
        Item 12: Check if Reddit moderators detect and act on a rumor.
        Detection probability scales with karma: p_detect = min(1.0, karma/100).
        """
        if platform != Platform.REDDIT:
            return
        if rumor.removed_by_mod or rumor.quarantined:
            return

        # Detection probability scales with karma (viral = more likely caught)
        p_detect = min(1.0, abs(rumor.karma_score) / 100.0)
        if self.rng.random() > p_detect:
            return

        # Find a moderator (fact-checker) on Reddit
        reddit_mods = [
            n for n in self.nodes
            if n.agent_type == AgentType.FACT_CHECKER
            and Platform.REDDIT in n.platforms
        ]
        if not reddit_mods:
            return

        mod = self.py_rng.choice(reddit_mods)

        # Detection delay: Exponential(mu=30min)
        detection_delay = float(self.rng.exponential(1800))

        # Check moderator fatigue
        if mod.id not in self.mod_actions_per_mod:
            self.mod_actions_per_mod[mod.id] = []
        mod_times = self.mod_actions_per_mod[mod.id]
        # Count actions in last hour
        recent = [t for t in mod_times if (self.current_time - t) < 3600]
        self.mod_actions_per_mod[mod.id] = recent  # clean up old entries
        if len(recent) >= 5:
            detection_delay *= 2.0  # fatigue: overwhelmed
        mod_times.append(self.current_time)

        self._schedule_event(
            self.current_time + detection_delay,
            EventType.REDDIT_MOD_ACTION,
            mod.id,
            {"rumor_version": rumor.version},
            platform=Platform.REDDIT,
        )

    def _handle_story_expiry_check(self, event: SimEvent):
        """Item 11: Periodic check for expired Instagram stories (currently unused,
        expiry is checked inline in _handle_share_rumor)."""
        pass

    def _handle_platform_hop(self, event: SimEvent):
        """
        Item 16: A node hops the rumor from one platform to another.

        The rumor arrives on the destination platform as a new seed from this node.
        Content quality gets a small perturbation (not a full mutation).
        Forward count resets to 0 (escapes WhatsApp limit on new platform).
        """
        node = self.nodes[event.node_id]
        data = event.data
        rumor_version = data["rumor_version"]
        dest_platform_str = data["dest_platform"]
        dest_platform = Platform(dest_platform_str)

        if rumor_version not in self.rumor_versions:
            return

        # Check node is still infected and on the dest platform
        if node.status not in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER):
            return
        if dest_platform not in node.platforms:
            return

        rumor = self.rumor_versions[rumor_version]
        self.total_platform_hops += 1

        # Create a slightly perturbed version for the new platform (NOT a mutation)
        # Content quality +-0.1, forward_count reset
        new_quality = float(np.clip(rumor.content_quality + self.rng.uniform(-0.1, 0.1), 0.1, 1.0))

        # Recompute virality for destination platform
        media_idx = MEDIA_TYPES.index(rumor.media_type)
        platform_fit = PLATFORM_FIT_MATRIX[dest_platform][media_idx]
        emotional_impact = float(np.mean(rumor.emotions.as_array()))
        new_virality = new_quality * platform_fit * (1 + emotional_impact * 0.3)

        # Use same rumor version but update platform-specific fields
        # (We don't create a new version for hops — same rumor, different platform)

        self._log_event(self.current_time, "platform_hop", node.id, platform=dest_platform, details={
            "source_platform": data["source_platform"],
            "dest_platform": dest_platform_str,
            "rumor_version": rumor_version,
        })

        # Schedule shares on the destination platform
        # ISSUE C: forward_depth resets to 0 on platform hop (new chain)
        self._schedule_shares(node, rumor, platform=dest_platform, forward_depth=0)

        # Spec §3.6: Correction follow probability after platform hop.
        # Corrections do NOT automatically follow the hop, but there's a 15% chance
        # IF a multi-platform node on the TARGET platform has seen the correction
        # on the SOURCE platform.
        source_platform = Platform(data["source_platform"])
        if self.correction_active:
            # Check if any multi-platform node on dest has seen correction on source
            bridge_seen_correction = False
            for n in self.nodes:
                if (n.status == NodeStatus.CORRECTED
                        and source_platform in n.platforms
                        and dest_platform in n.platforms
                        and n.id != node.id):
                    bridge_seen_correction = True
                    break
            if bridge_seen_correction and self.rng.random() < 0.15:
                # Correction follows the hop with Exp(mu=1h) delay
                corr_delay = float(self.rng.exponential(3600))  # mu=1h
                # Find a fact-checker or corrected node on dest platform to share correction
                correction_source = None
                for n in self.nodes:
                    if (n.status == NodeStatus.CORRECTED
                            and dest_platform in n.platforms):
                        correction_source = n
                        break
                if correction_source is not None:
                    self._schedule_event(
                        self.current_time + corr_delay,
                        EventType.SHARE_CORRECTION,
                        correction_source.id,
                        {"sender_id": correction_source.id},
                        platform=dest_platform,
                    )
                    self._log_event(self.current_time, "correction_follow_hop", node.id,
                                    platform=dest_platform, details={
                        "source_platform": source_platform.value,
                        "delay": corr_delay,
                    })

    def _handle_twitter_community_note(self, event: SimEvent):
        """
        Item 17: Twitter community note attached to a rumor.
        All future viewers on Twitter have 50% chance of auto-correction on receipt.
        """
        rumor_version = event.data.get("rumor_version", 0)
        if rumor_version not in self.rumor_versions:
            return

        rumor = self.rumor_versions[rumor_version]

        # Mark this rumor version as having a community note
        if not hasattr(rumor, '_community_note'):
            rumor._community_note = True

        self._log_event(self.current_time, "twitter_community_note", event.node_id,
                        platform=Platform.TWITTER, details={
            "rumor_version": rumor_version,
        })

    def _handle_emergency_correction(self, event: SimEvent):
        """
        Item 19: Emergency correction injection.
        ALL fact-checkers on ALL platforms simultaneously generate corrections.
        """
        self._log_event(self.current_time, "emergency_correction", 0, details={
            "injection_time": self.current_time,
        })

        self.correction_active = True
        self.correction_origin_time = self.current_time

        # All fact-checkers across all platforms generate corrections immediately
        for node in self.nodes:
            if node.agent_type != AgentType.FACT_CHECKER:
                continue
            if node.status == NodeStatus.REMOVED:
                continue

            node.status = NodeStatus.IMMUNE

            # Share corrections on ALL platforms this FC is on
            for plat in node.platforms:
                if plat in self.config.active_platforms:
                    self._schedule_correction_shares(node, platform=plat)

    # -------------------------------------------------------------------------
    # Group E: Network Dynamics & Agent Handlers (Items 30-36)
    # -------------------------------------------------------------------------

    def _reactive_unfollow(self, node: Node, source: Node, platform: Platform):
        """Item 31: Unfollow a node that repeatedly sent rumor messages."""
        connections = node.platform_connections.get(platform, [])
        if source.id not in connections:
            return

        connections.remove(source.id)
        # Also remove reverse edge on bidirectional platforms
        if PLATFORM_CONFIG[platform]["edge_bidirectional"]:
            src_conns = source.platform_connections.get(platform, [])
            if node.id in src_conns:
                src_conns.remove(node.id)

        # Remove from edge weights
        node.platform_edge_weights.get(platform, {}).pop(source.id, None)
        source.platform_edge_weights.get(platform, {}).pop(node.id, None)

        # Remove from networkx graph
        G = self.platform_graphs.get(platform)
        if G is not None and G.has_edge(node.id, source.id):
            G.remove_edge(node.id, source.id)

        node.rewiring_events += 1
        self.rewiring_events += 1
        self._total_unfollows += 1
        self._log_event(self.current_time, "unfollow", node.id, source.id,
                        platform=platform)

    def _reactive_seek(self, node: Node, platform: Platform):
        """Item 31: Add 1-2 weak-tie edges within same echo chamber (rabbit hole)."""
        chamber = node.echo_chamber_idx
        same_chamber_nodes = [
            n for n in self.nodes
            if n.echo_chamber_idx == chamber
            and n.id != node.id
            and n.id not in node.platform_connections.get(platform, [])
            and platform in n.platforms
        ]
        if not same_chamber_nodes:
            return

        num_new = min(self.rng.integers(1, 3), len(same_chamber_nodes))
        new_targets = self.py_rng.sample(same_chamber_nodes, k=num_new)

        for t in new_targets:
            strength = float(self.rng.uniform(0.2, 0.4))

            # Add to node's connections
            node.platform_connections.setdefault(platform, []).append(t.id)
            node.platform_edge_weights.setdefault(platform, {})[t.id] = strength

            # Bidirectional
            if PLATFORM_CONFIG[platform]["edge_bidirectional"]:
                t.platform_connections.setdefault(platform, []).append(node.id)
                t.platform_edge_weights.setdefault(platform, {})[node.id] = strength

            # Update networkx graph
            G = self.platform_graphs.get(platform)
            if G is not None:
                G.add_edge(node.id, t.id, weight=strength)

        node.rewiring_events += 1
        self.rewiring_events += 1
        self._total_seeks += 1
        self._log_event(self.current_time, "seek_connections", node.id,
                        platform=platform, details={"new_edges": num_new})

    def _check_bot_detection(self, bot: Node, platform: Platform):
        """
        Item 32: Check if a bot gets detected on this share event.
        base_rate = 0.001, escalates with activity, platform-specific multiplier.

        ISSUE 9 NOTE: Detection rate is intentionally low (~4% of bots per run).
        This models real-world bot detection: most bots evade detection, especially
        with < 10 shares/hour. Expected: 0-4 detections per 500-node run with ~35 bots.
        Only bots with 20+ shares/hour reliably trigger detection (10x escalation).
        """
        bot.shares_this_hour += 1
        detection_rate = 0.008

        # Activity-based escalation
        if bot.shares_this_hour > 20:
            detection_rate *= 15.0
        elif bot.shares_this_hour > 10:
            detection_rate *= 5.0

        # Platform-specific detection
        detection_rate *= BOT_DETECTION_PLATFORM_MULT.get(platform, 1.0)

        # Coordinated wave penalty: 3+ bots from same cluster shared within 60s
        if bot.bot_cluster_id is not None:
            cluster_bots = self.bot_clusters.get(bot.bot_cluster_id, [])
            recent_cluster_shares = 0
            for bid in cluster_bots:
                b = self.nodes[bid]
                if b.id != bot.id and b.infected_at is not None:
                    if abs(self.current_time - b.infected_at) < 60:
                        recent_cluster_shares += 1
            if recent_cluster_shares >= 2:  # 3+ total including this bot
                detection_rate *= 2.0

        # Phase 3: bot_detection_rate_multiplier (2.0 = "bots detected 1h earlier")
        detection_rate *= self.config.bot_detection_rate_multiplier

        if self.rng.random() < detection_rate:
            bot.detected = True
            bot.status = NodeStatus.REMOVED
            self.bots_detected += 1

            # Sever all edges
            for plat in bot.platforms:
                bot.platform_connections[plat] = []
                bot.platform_edge_weights[plat] = {}
                G = self.platform_graphs.get(plat)
                if G is not None and G.has_node(bot.id):
                    G.remove_edges_from(list(G.edges(bot.id)))

            self._log_event(self.current_time, "bot_detected", bot.id,
                            platform=platform, details={
                "cluster": bot.bot_cluster_id,
                "shares_this_hour": bot.shares_this_hour,
            })

    def _handle_bot_wave(self, event: SimEvent):
        """Item 32: Coordinated bot wave — all bots in cluster share rapidly."""
        cluster_id = event.data.get("cluster_id")
        if cluster_id is None:
            return

        cluster_bots = self.bot_clusters.get(cluster_id, [])
        rumor = self.rumor_versions.get(0)
        if rumor is None:
            return

        for bid in cluster_bots:
            bot = self.nodes[bid]
            if bot.detected or bot.status == NodeStatus.REMOVED:
                continue
            if bot.status not in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER):
                continue

            # Each bot shares within Exp(mu=30s) of trigger
            delay = float(self.rng.exponential(30))
            for plat in bot.platforms:
                if plat in self.config.active_platforms:
                    self._schedule_shares(bot, rumor, platform=plat)

        # Cooldown: Exp(mu=5min), then re-enable wave
        cooldown = float(self.rng.exponential(300))
        self.bot_wave_active[cluster_id] = False

    def _handle_bot_credibility_tick(self, event: SimEvent):
        """Item 32: Bot credibility grows 0.05/hour, capped at 0.6. Also reset shares_this_hour."""
        for node in self.nodes:
            if node.agent_type == AgentType.BOT and not node.detected:
                node.apparent_credibility = min(0.6, node.apparent_credibility + 0.05)
                node.shares_this_hour = 0  # reset hourly counter

        # Schedule next tick
        if self.current_time + 3600 < self.config.max_time:
            self._schedule_event(self.current_time + 3600.0, EventType.BOT_CREDIBILITY_TICK, 0)

    def _handle_super_spreader_check(self, event: SimEvent):
        """
        Item 34: Check if a super spreader event should trigger (spec §4.10).
        4 trigger conditions: engagement threshold, influencer amplification,
        emotional mutation spike, cross-platform hop.
        """
        min_gap = 1800  # minimum 30min between SSE triggers
        if (self.current_time - self._last_sse_time) <= min_gap:
            # Schedule next check and skip
            if self.current_time + 60 < self.config.max_time:
                self._schedule_event(self.current_time + 60.0, EventType.SUPER_SPREADER_CHECK, 0)
            return

        triggered = False

        # 1. Engagement threshold: >10% of nodes shared within last 5 minutes
        cutoff = self.current_time - 300
        recent = sum(1 for t in self.recent_shares if t > cutoff)
        threshold = self.config.network_size * 0.10
        if recent > threshold:
            self._last_sse_time = self.current_time
            self._trigger_super_spreader("engagement", recent)
            triggered = True

        # 2. Influencer amplification: any influencer shared in last 60s
        if not triggered:
            influencer_cutoff = self.current_time - 60
            recent_influencer_shares = sum(
                1 for e in self.event_log[-200:]  # check recent events
                if e.event_type == "node_infected"
                and e.time > influencer_cutoff
                and self.nodes[e.node_id].agent_type == AgentType.INFLUENCER
            ) if self.event_log else 0
            if recent_influencer_shares > 0:
                self._last_sse_time = self.current_time
                self._trigger_super_spreader("influencer", recent_influencer_shares)
                triggered = True

        # 3. Emotional mutation spike: any mutation with emotional_charge > 0.85
        if not triggered:
            for ver_id, rumor in self.rumor_versions.items():
                if ver_id == 0:
                    continue
                avg_emotion = float(np.mean(rumor.emotions.as_array()))
                if avg_emotion > 0.85 and rumor.origin_time > self.current_time - 300:
                    self._last_sse_time = self.current_time
                    self._trigger_super_spreader("emotion_spike", 1)
                    triggered = True
                    break

        # 4. Cross-platform hop (checked inline in _handle_platform_hop, but also verify here)
        if not triggered and self.total_platform_hops > 0:
            recent_hops = sum(
                1 for e in self.event_log[-100:]
                if e.event_type == "platform_hop" and e.time > self.current_time - 300
            ) if self.event_log else 0
            if recent_hops > 0:
                self._last_sse_time = self.current_time
                self._trigger_super_spreader("platform_hop", recent_hops)
                triggered = True

        # Schedule next check (every 60s)
        if self.current_time + 60 < self.config.max_time:
            self._schedule_event(self.current_time + 60.0, EventType.SUPER_SPREADER_CHECK, 0)

    def _trigger_super_spreader(self, trigger_type: str, affected_count: int):
        """Item 34: Apply super spreader boost — 5-10x arrival rate for affected duration."""
        self.super_spreader_events += 1
        boost = float(self.rng.uniform(5, 10))
        duration = float(self.rng.exponential(1800))  # mu=30min

        self._log_event(self.current_time, "super_spreader_event", 0, details={
            "trigger_type": trigger_type,
            "affected_count": affected_count,
            "boost": boost,
            "duration": duration,
        })

        # Boost: reduce service times for currently-scheduled shares (proxy for rate boost)
        # This is modeled by scheduling extra amplification events for top sharers
        rumor = self.rumor_versions.get(0)
        if rumor is None:
            return

        # Pick random believing nodes and give them extra shares
        believers = [n for n in self.nodes if n.status == NodeStatus.BELIEVING]
        num_extra = min(int(affected_count * 0.1), len(believers))
        if num_extra > 0 and believers:
            extra_sharers = self.py_rng.sample(believers, k=min(num_extra, len(believers)))
            for sharer in extra_sharers:
                for plat in sharer.platforms:
                    if plat in self.config.active_platforms:
                        self._schedule_shares(sharer, rumor, platform=plat)

    def _handle_crisis_start(self, event: SimEvent):
        """Item 35: Crisis begins — drop thresholds, boost fear/urgency susceptibility."""
        intensity = event.data.get("intensity", 0.5)
        duration = event.data.get("duration", 14400)
        crisis_topic = event.data.get("crisis_topic", self.config.scenario)

        self.crisis_active = True

        # Determine topic relevance
        for node in self.nodes:
            if crisis_topic == self.config.scenario:
                relevance = 1.0
            else:
                relevance = 0.1

            crisis_modifier = 1.0 + (intensity * relevance)

            # Save pre-crisis threshold for recovery
            self.crisis_pre_thresholds[node.id] = node.credibility_threshold

            # Drop threshold
            node.credibility_threshold *= (1 - crisis_modifier * 0.3)
            node.effective_threshold *= (1 - crisis_modifier * 0.3)
            node.credibility_threshold = max(0.05, node.credibility_threshold)
            node.effective_threshold = max(0.05, node.effective_threshold)

            # Boost fear and urgency susceptibility
            node.susceptibility.fear = min(1.0, node.susceptibility.fear + 0.2)
            node.susceptibility.urgency = min(1.0, node.susceptibility.urgency + 0.2)

            # Spec §2.6: middle age group gets financial literacy penalty during crisis
            if node.age_group == "middle":
                node.literacy_vector.financial = max(0.0, node.literacy_vector.financial * 0.85)

        self._log_event(self.current_time, "crisis_start", 0, details={
            "intensity": intensity,
            "duration": duration,
            "topic": crisis_topic,
        })

        # Schedule crisis end
        self._schedule_event(self.current_time + duration, EventType.CRISIS_END, 0, {
            "intensity": intensity,
        })

    def _handle_crisis_end(self, event: SimEvent):
        """Item 35: Crisis ends — thresholds recover over Exp(mu=2h) cooldown."""
        self.crisis_active = False

        # Recover thresholds to pre-crisis levels
        for node in self.nodes:
            pre_crisis = self.crisis_pre_thresholds.get(node.id)
            if pre_crisis is not None:
                # Partial recovery: trust decay accumulated during crisis is permanent
                # but threshold returns toward pre-crisis value
                recovery_fraction = min(1.0, float(self.rng.exponential(0.5)))
                node.credibility_threshold += (pre_crisis - node.credibility_threshold) * recovery_fraction
                node.effective_threshold = node.credibility_threshold

            # Remove fear/urgency boost (partial — some lasting effect)
            node.susceptibility.fear = max(
                node.original_susceptibility.fear,
                node.susceptibility.fear - 0.15
            )
            node.susceptibility.urgency = max(
                node.original_susceptibility.urgency,
                node.susceptibility.urgency - 0.15
            )

        self._log_event(self.current_time, "crisis_end", 0)

    # -------------------------------------------------------------------------
    # Computation Helpers
    # -------------------------------------------------------------------------

    def _compute_alignment(self, node: Node, rumor: Rumor) -> float:
        """Compute worldview alignment score between node and rumor (0.0-1.0).

        4D worldview per spec §4.7.
        max_distance = sqrt(4) * 2 = 4.0 (diagonal of [-1,1]^4 hypercube).

        When worldview_dimensions=2, only use first 2 dims (political, health_trust),
        with max_distance = sqrt(2+1) = ~2.24 (diagonal of [-1,1] x [0,1]).
        """
        node_wv = node.worldview_vector
        rumor_wv = rumor.alignment_vector
        if self.config.worldview_dimensions == 2:
            # Only use first 2 dimensions
            distance = float(np.linalg.norm(node_wv[:2] - rumor_wv[:2]))
            max_distance = np.sqrt(2**2 + 1**2)  # ~2.24: [-1,1] range + [0,1] range
        else:
            distance = float(np.linalg.norm(node_wv - rumor_wv))
            max_distance = 4.0  # spec §4.7: diagonal of [-1,1]^4 hypercube
        alignment = 1.0 - (distance / max_distance)
        return float(np.clip(alignment, 0.0, 1.0))

    def _get_sender_trust_modifier(self, sender: Node, edge_weight: float,
                                    receiver: Node = None) -> float:
        """Determine trust modifier based on sender type and tie strength."""
        # Item 32: Bot uses apparent_credibility (grows over time)
        if sender.agent_type == AgentType.BOT:
            base_trust = sender.apparent_credibility
            # Item 30: User-side bot detection via digital nativity
            if receiver is not None and self.rng.random() < receiver.bot_detection_intuition:
                base_trust = SENDER_TRUST_MODIFIERS["bot_detected"]  # 0.3x
            return base_trust
        if sender.agent_type == AgentType.INFLUENCER:
            return SENDER_TRUST_MODIFIERS["influencer"]
        if sender.agent_type == AgentType.FACT_CHECKER:
            # Fact-checker sharing a rumor = extremely credible (backfire)
            return SENDER_TRUST_MODIFIERS["fact_checker_shares_rumor"]
        if edge_weight > 0.7:
            return SENDER_TRUST_MODIFIERS["strong_tie"]
        if edge_weight < 0.3:
            return SENDER_TRUST_MODIFIERS["stranger"]
        return SENDER_TRUST_MODIFIERS["regular"]

    def _compute_infection_rate(self) -> float:
        """Compute current infection rate (believing + silent_believer / total)."""
        infected = sum(
            1 for n in self.nodes
            if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER)
        )
        return infected / len(self.nodes) if self.nodes else 0.0

    def _compute_r0(self) -> float:
        """Compute current effective reproduction number.

        Uses recently infected nodes (last hour) for dynamic R0.
        If no recent infections, R0 is 0 (rumor has stalled).
        """
        recently_infected = [
            n for n in self.nodes
            if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER)
            and n.infected_at is not None
            and (self.current_time - n.infected_at) < 3600  # last hour
        ]
        if not recently_infected:
            # No recent infections → R0 is effectively 0 (stalled)
            return 0.0

        return sum(n.downstream_infections for n in recently_infected) / len(recently_infected)

    # -------------------------------------------------------------------------
    # Items 21-29: Behavioral Depth Helpers
    # -------------------------------------------------------------------------

    def _apply_literacy_boost(self, node: Node, rumor: Rumor) -> float:
        """
        Item 21: Topic-specific literacy increases effective threshold.
        Returns a multiplier > 1.0 (higher literacy = harder to convince).
        """
        topic = rumor.scenario  # e.g. "celebrity", "financial", "health", "campus"
        base_literacy = node.literacy_vector.get(topic)
        # Literacy acts as resistance: 0.5 = neutral, 1.0 = very resistant
        # Scale: literacy 0.5 → no change (1.0x), literacy 0.9 → 1.2x, literacy 0.2 → 0.85x
        literacy_boost = 1.0 + (base_literacy - 0.5) * 1.5
        return max(0.5, min(2.0, literacy_boost))

    def _degrade_literacy(self, node: Node, rumor: Rumor, sender: Node, edge_weight: float):
        """
        Item 22: Literacy degrades under sustained pressure.
        effective_literacy = base * (0.97 ^ (exposures * pressure_modifier))

        Pressure increases from:
          - High-credibility sender (influencer, FC, strong tie)
          - High emotional charge (fear > 0.5)
          - Worldview alignment (confirmation bias)
        """
        topic = rumor.scenario
        base_literacy = node.literacy_vector.get(topic)

        # Compute pressure modifier
        pressure = 1.0

        # Source credibility pressure
        source_trust = self._get_sender_trust_modifier(sender, edge_weight)
        if source_trust > 1.5:
            pressure *= 1.3  # high-cred source increases pressure
        elif source_trust > 1.0:
            pressure *= 1.1

        # Emotional charge pressure (fear-driven)
        if rumor.emotions.fear > 0.5:
            pressure *= 1.2
        emotional_charge = float(np.mean(rumor.emotions.as_array()))
        if emotional_charge > 0.5:
            pressure *= 1.1

        # Worldview alignment pressure
        alignment = self._compute_alignment(node, rumor)
        if alignment > 0.6:
            pressure *= 1.15

        # Exponential decay: literacy_effective = base * (0.97 ^ (exposures * pressure))
        decay_rate = 0.97
        effective_literacy = base_literacy * (decay_rate ** (node.times_exposed * pressure))
        effective_literacy = max(0.1, effective_literacy)  # floor

        # Update the literacy vector for this topic
        setattr(node.literacy_vector, topic, effective_literacy)

    def _update_emotional_state(self, node: Node, rumor: Rumor):
        """
        Item 26: Update emotional priming and fatigue.

        Priming: short-term amplification when exposed to high-emotion messages.
        Fatigue: long-term dampening from repeated exposure (every 10 total messages).

        susceptibility[i] = original[i] + priming[i] - fatigue[i]
        Clamped to [original*0.3, original*1.5]

        emotional_dynamics_mode controls which parts are active:
          'both' (default): priming + fatigue
          'priming_only': priming only, no fatigue
          'fatigue_only': fatigue only, no priming
          'static': no emotional dynamics at all
        """
        mode = self.config.emotional_dynamics_mode
        if mode == 'static':
            return  # no emotional dynamics

        rumor_emotions = rumor.emotions.as_array()
        dims = ["fear", "outrage", "humor", "curiosity", "urgency"]

        # BUG 5 FIX: Track total messages for fatigue (not per-dimension needing 10 each)
        any_high_emotion = False
        for i, dim in enumerate(dims):
            emotion_val = rumor_emotions[i]

            if emotion_val > 0.5:
                any_high_emotion = True

                # Priming: boost susceptibility for this dimension
                if mode in ('both', 'priming_only'):
                    priming_boost = 0.05 * emotion_val
                    current_priming = getattr(node.emotional_priming, dim)
                    setattr(node.emotional_priming, dim, current_priming + priming_boost)

                # Track per-dimension count
                node.messages_processed_per_emotion[dim] += 1

        # Fatigue: every 10 total high-emotion messages, add permanent dampening
        # across ALL dimensions that had high emotion
        if any_high_emotion and mode in ('both', 'fatigue_only'):
            total_msgs = sum(node.messages_processed_per_emotion.values())
            if total_msgs > 0 and total_msgs % 10 == 0:
                for dim in dims:
                    if node.messages_processed_per_emotion[dim] > 0:
                        original_val = getattr(node.original_susceptibility, dim)
                        fatigue_increment = 0.05 * original_val
                        current_fatigue = getattr(node.emotional_fatigue, dim)
                        setattr(node.emotional_fatigue, dim, current_fatigue + fatigue_increment)

        # Update effective susceptibility for all dimensions
        for i, dim in enumerate(dims):
            original_val = getattr(node.original_susceptibility, dim)
            priming_val = getattr(node.emotional_priming, dim)
            fatigue_val = getattr(node.emotional_fatigue, dim)

            effective = original_val + priming_val - fatigue_val
            effective = max(original_val * 0.3, min(original_val * 1.5, effective))
            setattr(node.susceptibility, dim, effective)

    def _decay_priming(self, node: Node):
        """Item 26: Priming decays by 5% per hour (called from attention_recovery)."""
        for dim in ["fear", "outrage", "humor", "curiosity", "urgency"]:
            current = getattr(node.emotional_priming, dim)
            setattr(node.emotional_priming, dim, current * 0.95)

    def _select_rumor_shape(self, sender: Node, rumor: Rumor, edge_weight: float) -> RumorShape:
        """
        Item 28: Select message framing shape based on sender, rumor, and tie strength.

        Logic tree:
          Bot → BARE_FORWARD
          High emotion + early exposure → REACTION
          Influencer → ELABORATOR
          High threshold (barely passed) → SKEPTIC_SHARER
          Strong tie → PERSONAL_FRAME
          Default → weighted random
        """
        if sender.agent_type == AgentType.BOT:
            return RumorShape.BARE_FORWARD

        emotional_intensity = rumor.emotions.dot(sender.susceptibility)

        if emotional_intensity > 0.7 and sender.times_exposed <= 2:
            return RumorShape.REACTION

        if sender.agent_type == AgentType.INFLUENCER:
            return RumorShape.ELABORATOR

        if sender.effective_threshold > 0.6:
            return RumorShape.SKEPTIC_SHARER

        if edge_weight > 0.7:
            return RumorShape.PERSONAL_FRAME

        # Default: weighted random
        shapes = list(RumorShape)
        weights = [0.30, 0.20, 0.25, 0.10, 0.15]
        idx = self.rng.choice(len(shapes), p=weights)
        return shapes[idx]

    def _select_correction_shape(self, sender: Node) -> CorrectionShape:
        """Item 28: Select correction message shape based on sender status."""
        if sender.agent_type == AgentType.FACT_CHECKER:
            return CorrectionShape.DEBUNK

        if sender.status == NodeStatus.CORRECTED:
            # Previously believed, now corrected
            if self.rng.random() < 0.6:
                return CorrectionShape.RELUCTANT_WALKBACK
            return CorrectionShape.RELAY

        if sender.status == NodeStatus.UNAWARE:
            # Never believed, sharing the correction
            if self.rng.random() < 0.3:
                return CorrectionShape.TOLD_YOU_SO
            return CorrectionShape.RELAY

        return CorrectionShape.RELAY

    def _compute_backfire(self, sender: Node, rumor: Rumor) -> float:
        """
        Items 23-24: Compute combined backfire multiplier when a high-credibility
        or high-literacy sender shares the rumor.

        Returns a threshold reduction for receivers (0.0 = no backfire, up to 0.85).
        Receivers get: threshold *= (1 - backfire)
        """
        total_backfire = 0.0

        # Item 24: Source credibility backfire
        # High-credibility sender (FC, influencer) sharing = "if they believe it..."
        if sender.agent_type == AgentType.FACT_CHECKER:
            cred_backfire = min(0.7, sender.credibility_threshold * 0.5)
            total_backfire += cred_backfire
        elif sender.agent_type == AgentType.INFLUENCER:
            cred_backfire = min(0.5, sender.credibility_threshold * 0.3)
            total_backfire += cred_backfire

        # Item 23: Literacy-based backfire cascade
        # High-literacy node (>0.7 in topic) sharing = "even the expert fell for it"
        topic = rumor.scenario
        sender_literacy = sender.literacy_vector.get(topic)
        if sender_literacy > 0.7:
            lit_backfire = min(0.7, sender_literacy * 0.5)
            total_backfire += lit_backfire

        # Cap combined backfire at 0.85
        return min(0.85, total_backfire)

    def _check_algorithmic_amplification(self, sharer: Node, rumor: Rumor,
                                         platform: Platform = None):
        """Check if content should get algorithmically boosted (Twitter/Instagram).

        Spec §3.3: When shares in a sliding window exceed the engagement threshold,
        the message gets pushed to non-followers (explore/for-you page).
        boost_duration = Exp(mu=30min). During the boost window, no NEW amplification
        fires on the same platform (the existing boost is still active).
        """
        if platform is None:
            platform = self.config.seed_platform

        platform_config = PLATFORM_CONFIG[platform]
        if not platform_config["algorithmic_amplification"]:
            return

        # Cooldown: if a boost is still active on this platform, skip
        if self.current_time < self._amp_cooldown_until.get(platform, 0.0):
            return

        threshold_pct = platform_config["engagement_threshold_pct"]
        window_min = platform_config["engagement_window_min"]
        if threshold_pct is None or window_min is None:
            return

        # Count shares in window (per-platform tracking)
        window_seconds = window_min * 60
        cutoff = self.current_time - window_seconds
        platform_shares = self.recent_shares_per_platform[platform]
        while platform_shares and platform_shares[0] < cutoff:
            platform_shares.popleft()

        # Threshold is based on nodes on THIS platform (floor of 5)
        nodes_on_platform = sum(1 for n in self.nodes if platform in n.platforms)
        shares_in_window = len(platform_shares)
        threshold = max(5, int(nodes_on_platform * threshold_pct))

        if shares_in_window > threshold:
            # Phase 3: algorithmic_amplification_multiplier (0 = disable, >1 = stronger)
            if self.config.algorithmic_amplification_multiplier <= 0:
                return
            boost_multiplier = float(self.rng.uniform(3, 5))
            boost_duration = float(self.rng.exponential(1800))  # Exp(mu=30min)

            # Set cooldown: no new amplification on this platform until boost expires
            self._amp_cooldown_until[platform] = self.current_time + boost_duration

            # Pick unaware nodes ON THIS PLATFORM not already connected to sharer
            sharer_connections = set(self._get_node_connections_on_platform(sharer, platform))
            unaware = [
                n for n in self.nodes
                if n.status == NodeStatus.UNAWARE
                and platform in n.platforms
                and n.id not in sharer_connections
            ]
            boost_count = min(int(len(sharer_connections) * boost_multiplier), len(unaware))
            # Phase 3: Scale boost count by amplification multiplier
            boost_count = int(boost_count * self.config.algorithmic_amplification_multiplier)

            if boost_count > 0:
                targets = self.py_rng.sample(unaware, boost_count)
                for target in targets:
                    # Spread arrivals over the boost window
                    delay = float(self.rng.exponential(boost_duration / max(boost_count, 1)))
                    self._schedule_event(
                        self.current_time + delay,
                        EventType.SHARE_RUMOR,
                        target.id,
                        {
                            "sender_id": sharer.id,
                            "rumor_version": rumor.version,
                            "edge_weight": 0.1,
                            "credibility_modifier": 1.0,
                            "_amplified": True,  # mark as amp-boosted
                        },
                        platform=platform,
                    )

                self._log_event(self.current_time, "algorithmic_amplification", sharer.id,
                                platform=platform, details={
                    "shares_in_window": shares_in_window,
                    "boost_count": boost_count,
                    "boost_multiplier": boost_multiplier,
                    "boost_duration_min": boost_duration / 60,
                })

    # -------------------------------------------------------------------------
    # Termination Conditions
    # -------------------------------------------------------------------------

    def _check_termination_conditions(self) -> tuple[bool, str]:
        """Check all adaptive termination conditions. Returns (should_terminate, reason)."""
        # 1. Max time (hard cap)
        if self.current_time >= self.config.max_time:
            return True, "max_time"

        infection_rate = self._compute_infection_rate()
        r0 = self._compute_r0()

        # Record R0 timeline
        self.r0_timeline.append((self.current_time, r0))
        if r0 > self.r0_peak:
            self.r0_peak = r0

        # 2. Rumor dead: R0 < 0.1 for 30+ min AND no new infections in 60 min
        if r0 < 0.1:
            if self.r0_below_threshold_since is None:
                self.r0_below_threshold_since = self.current_time
            elif (self.current_time - self.r0_below_threshold_since >= 1800  # 30 min
                  and self.current_time - self.last_infection_time >= 3600):  # 60 min
                return True, "rumor_dead"
        else:
            self.r0_below_threshold_since = None

        # 3. Fully saturated: >95% of reachable nodes infected or corrected
        total_reached = sum(
            1 for n in self.nodes
            if n.status != NodeStatus.UNAWARE
        )
        if total_reached / len(self.nodes) > 0.95:
            return True, "saturated"

        # 4. Fully corrected: >90% of believers corrected AND spread rate < 0.5%/hr
        believers = sum(
            1 for n in self.nodes
            if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER)
        )
        corrected = sum(1 for n in self.nodes if n.status == NodeStatus.CORRECTED)
        if believers + corrected > 0 and corrected / (believers + corrected) > 0.9:
            # Check spread rate
            if len(self.infection_timeline) >= 2:
                recent = [t for t, r in self.infection_timeline if t > self.current_time - 3600]
                if len(recent) < self.config.network_size * 0.005:
                    return True, "corrected"

        return False, ""

    # -------------------------------------------------------------------------
    # Checkpoints & Results
    # -------------------------------------------------------------------------

    def _create_checkpoint(self) -> CheckpointSnapshot:
        """Create a snapshot of current simulation state."""
        status_counts = {}
        for status in NodeStatus:
            status_counts[status] = sum(1 for n in self.nodes if n.status == status)

        return CheckpointSnapshot(
            time=self.current_time,
            believing_count=status_counts.get(NodeStatus.BELIEVING, 0),
            silent_believer_count=status_counts.get(NodeStatus.SILENT_BELIEVER, 0),
            corrected_count=status_counts.get(NodeStatus.CORRECTED, 0),
            unaware_count=status_counts.get(NodeStatus.UNAWARE, 0),
            immune_count=status_counts.get(NodeStatus.IMMUNE, 0),
            removed_count=status_counts.get(NodeStatus.REMOVED, 0),
            infection_rate=self._compute_infection_rate(),
            r0_estimate=self._compute_r0(),
            total_mutations=self.total_mutations,
            platform_hops=0,  # Phase 1: single platform
        )

    def _classify_death_type(self) -> DeathType:
        """
        ISSUE 3 FIX: Classify how the rumor died using spec section 6.7.
        Check order: still_alive -> saturated -> corrected -> time_decayed
                     -> mutated_away -> starved (fallback)
        """
        r0 = self._compute_r0()
        corrected_count = sum(1 for n in self.nodes if n.status == NodeStatus.CORRECTED)
        believing_count = sum(
            1 for n in self.nodes
            if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER)
        )
        # Peak believers = max infected at any point (from timeline)
        peak_believers = 0
        for _, rate in self.infection_timeline:
            count_at_point = int(rate * len(self.nodes))
            if count_at_point > peak_believers:
                peak_believers = count_at_point
        if peak_believers == 0:
            peak_believers = believing_count  # fallback

        # 1. still_alive: termination_reason == "max_time"
        _, reason = self._check_termination_conditions()
        if reason == "max_time" or (self._termination_time is None and
                self.current_time >= self.config.max_time):
            return DeathType.STILL_ALIVE

        # 2. saturated: final_infection >= 0.80 * reachable_nodes
        infection_rate = self._compute_infection_rate()
        if infection_rate >= 0.80:
            return DeathType.SATURATED

        # ISSUE B FIX: Fizzle guard — runs that never took off (<5% infection)
        # cannot be "corrected"; they simply starved from lack of spreading.
        if infection_rate < 0.05:
            return DeathType.STARVED

        # 3. corrected: corrected_count >= 0.3 * peak_believers
        if peak_believers > 0 and corrected_count >= 0.3 * peak_believers:
            return DeathType.CORRECTED

        # 4. time_decayed: story decay platform + content outlived its shelf life
        # ISSUE C FIX: Instagram stories have a 24h TTL. Even non-story content
        # (text posts) decays on story-centric platforms. Trigger when:
        #   (a) A rumor version literally expired, OR
        #   (b) Instagram is the seed platform, the run lasted >2h (content aging),
        #       the rumor had initial traction (peak > 5%), and believers are declining.
        for plat in self.config.active_platforms:
            if PLATFORM_CONFIG[plat].get("story_decay", False):
                story_ttl = PLATFORM_CONFIG[plat].get("story_ttl")
                if story_ttl is not None:
                    # Check 1: Any rumor version literally expired
                    for rv in self.rumor_versions.values():
                        if (rv.expiry_time is not None
                                and self.current_time > rv.expiry_time):
                            return DeathType.TIME_DECAYED
                    # Check 2: seed platform is this story-decay platform,
                    # run > 2 hours, and rumor had meaningful traction.
                    # On story-decay platforms, content visibility naturally
                    # drops over time, contributing to rumor death.
                    actual_time = (self._termination_time
                                   if self._termination_time is not None
                                   else self.current_time)
                    if (plat == self.config.seed_platform
                            and actual_time > 7200  # > 2 hours
                            and peak_believers > int(0.05 * len(self.nodes))):
                        return DeathType.TIME_DECAYED

        # 5. mutated_away: max_mutation_distance > 0.35
        # ISSUE C FIX: Lowered from 1.5 to 0.35. Each mutation shifts emotions
        # by uniform(-0.15, 0.15) per dim; typical max distance after a chain
        # of mutations peaks around 0.25-0.45. Threshold must be achievable.
        if self.rumor_versions:
            max_dist = max(
                (rv.mutation_distance for rv in self.rumor_versions.values()),
                default=0.0,
            )
            if max_dist > 0.35:
                return DeathType.MUTATED_AWAY

        # 6. starved (fallback): R0 < 1 AND no active spreaders
        return DeathType.STARVED

    def _compile_results(self) -> SimulationResult:
        """Compile final simulation results."""
        infection_rate = self._compute_infection_rate()
        r0 = self._compute_r0()

        # Peak infection
        peak_rate = 0.0
        peak_time = 0.0
        for t, rate in self.infection_timeline:
            if rate > peak_rate:
                peak_rate = rate
                peak_time = t

        # Status counts
        believing = sum(1 for n in self.nodes if n.status == NodeStatus.BELIEVING)
        silent = sum(1 for n in self.nodes if n.status == NodeStatus.SILENT_BELIEVER)
        corrected = sum(1 for n in self.nodes if n.status == NodeStatus.CORRECTED)
        unaware = sum(1 for n in self.nodes if n.status == NodeStatus.UNAWARE)

        # Per-platform stats
        platform_infection_rates = {}
        platform_node_counts = {}
        for plat in self.config.active_platforms:
            plat_nodes = [n for n in self.nodes if plat in n.platforms]
            plat_count = len(plat_nodes)
            platform_node_counts[plat] = plat_count
            if plat_count > 0:
                plat_infected = sum(
                    1 for n in plat_nodes
                    if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER)
                )
                platform_infection_rates[plat] = plat_infected / plat_count
            else:
                platform_infection_rates[plat] = 0.0

        # Determine termination reason
        _, reason = self._check_termination_conditions()
        if not reason:
            reason = "max_time"

        # BUG 14 FIX: Use explicit termination time if recorded, else current_time
        actual_term_time = self._termination_time if self._termination_time is not None else self.current_time

        # Phase 3: Build detailed timelines and node snapshot if enabled
        detailed_timelines = None
        node_data_snapshot = None
        if self.config.detailed_tracking:
            detailed_timelines = {k: list(v) for k, v in self._detailed_timelines.items()}
            # Node data snapshot for network autopsy
            node_data_snapshot = {}
            for n in self.nodes:
                node_data_snapshot[n.id] = {
                    "status": n.status.value,
                    "agent_type": n.agent_type.value,
                    "age_group": n.age_group,
                    "echo_chamber_idx": n.echo_chamber_idx,
                    "infected_by": n.infected_by,
                    "infected_at": n.infected_at,
                    "infected_on_platform": n.infected_on_platform.value if n.infected_on_platform else None,
                    "downstream_infections": n.downstream_infections,
                    "rumor_version": n.rumor_version,
                    "credibility_threshold": n.credibility_threshold,
                    "platforms": [p.value for p in n.platforms],
                    "connections_count": {
                        p.value: len(conns)
                        for p, conns in n.platform_connections.items()
                    },
                }

        return SimulationResult(
            termination_reason=reason,
            termination_time=actual_term_time,
            final_infection_rate=infection_rate,
            peak_infection_rate=peak_rate,
            peak_time=peak_time,
            total_believing=believing,
            total_silent_believers=silent,
            total_corrected=corrected,
            total_unaware=unaware,
            total_mutations=self.total_mutations,
            total_platform_hops=self.total_platform_hops,
            total_bots_detected=self.bots_detected,
            total_rewiring_events=self.rewiring_events,
            total_super_spreader_events=self.super_spreader_events,
            crisis_occurred=self.crisis_active or any(
                e.event_type == "crisis_start" for e in self.event_log
            ),
            r0_final=self.r0_peak,
            death_type=self._classify_death_type(),
            infection_timeline=self.infection_timeline.copy(),
            r0_timeline=self.r0_timeline.copy(),
            checkpoints=self.checkpoints.copy(),
            event_log=self.event_log.copy(),
            platform_infection_rates=platform_infection_rates,
            platform_node_counts=platform_node_counts,
            detailed_timelines=detailed_timelines,
            node_data_snapshot=node_data_snapshot,
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    def _log_event(self, time: float, event_type: str, node_id: int,
                   target_id: Optional[int] = None, platform: Optional[Platform] = None,
                   details: dict = None):
        """Log an event for analysis and replay."""
        self.event_log.append(EventLogEntry(
            time=time,
            event_type=event_type,
            node_id=node_id,
            target_id=target_id,
            platform=platform or self.config.platform,
            details=details or {},
        ))


# =============================================================================
# Entry Point / Quick Test
# =============================================================================

def _fmt_duration(seconds: float) -> str:
    """ISSUE 5 FIX: Format duration — minutes if < 1h, hours otherwise."""
    if seconds < 3600:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def run_single_simulation(
    scenario: str = "celebrity",
    seed_platform: Platform = Platform.TWITTER,
    active_platforms: list[Platform] = None,
    network_size: int = 500,
    seed: Optional[int] = None,
    media_type: Optional[str] = None,
    start_hour: Optional[int] = None,
) -> SimulationResult:
    """Convenience function to run a single simulation with defaults.

    Args:
        media_type: Content format ("text", "image", "video", "reel").
            If None, uses platform-specific default from PLATFORM_DEFAULT_MEDIA
            (e.g. Instagram -> "reel", Twitter -> "text").
        start_hour: Simulation start hour (0-23). If None, uses default (10).
    """
    if active_platforms is None:
        active_platforms = [seed_platform]
    # ISSUE 10 + ISSUE 3: Resolve media_type from platform default if not explicit
    if media_type is None:
        media_type = PLATFORM_DEFAULT_MEDIA.get(seed_platform, "text")

    config = SimulationConfig(
        network_size=network_size,
        active_platforms=active_platforms,
        seed_platform=seed_platform,
        scenario=scenario,
        media_type=media_type,
        master_seed=seed,
        **({"start_hour": start_hour} if start_hour is not None else {}),
    )
    engine = SimulationEngine(config)
    return engine.run()


# =============================================================================
# Phase 3: Monte Carlo Runner
# =============================================================================

def _mc_worker(args: dict) -> dict:
    """
    Module-level worker for multiprocessing (Windows uses spawn — must be picklable).

    Runs a single simulation and returns a lightweight result dict.
    """
    # Reconstruct config from serialized args
    config_dict = args["config_dict"]
    run_index = args["run_index"]

    # Reconstruct enum values
    config_dict["seed_platform"] = Platform(config_dict["seed_platform"])
    config_dict["active_platforms"] = [Platform(p) for p in config_dict["active_platforms"]]

    config = SimulationConfig(**config_dict)
    engine = SimulationEngine(config)
    result = engine.run()

    # Return lightweight dict (strip event_log for memory)
    return {
        "run_index": run_index,
        "final_infection_rate": result.final_infection_rate,
        "termination_time": result.termination_time,
        "termination_reason": result.termination_reason,
        "r0_final": result.r0_final,
        "death_type": result.death_type.value,
        "peak_infection_rate": result.peak_infection_rate,
        "peak_time": result.peak_time,
        "total_mutations": result.total_mutations,
        "total_platform_hops": result.total_platform_hops,
        "total_bots_detected": result.total_bots_detected,
        "total_rewiring_events": result.total_rewiring_events,
        "total_super_spreader_events": result.total_super_spreader_events,
        "crisis_occurred": result.crisis_occurred,
        "infection_timeline": result.infection_timeline,
        "r0_timeline": result.r0_timeline,
        "checkpoints": result.checkpoints,
        "platform_infection_rates": {p.value: v for p, v in result.platform_infection_rates.items()},
        "platform_node_counts": {p.value: v for p, v in result.platform_node_counts.items()},
    }


def run_monte_carlo(
    n_runs: int = 1000,
    scenario: str = "celebrity",
    seed_platform: Platform = Platform.TWITTER,
    active_platforms: list = None,
    network_size: int = 500,
    base_seed: int = 42,
    config_overrides: dict = None,
    max_workers: int = None,
) -> MonteCarloResult:
    """
    Run a Monte Carlo batch of simulations (Phase 3, spec section 6).

    Each run uses master_seed = base_seed + run_index for reproducibility.
    Uses multiprocessing.Pool for n_runs > 50.

    Args:
        n_runs: Number of simulation runs
        scenario: Rumor scenario name
        seed_platform: Platform where rumor starts
        active_platforms: List of active platforms (None = [seed_platform])
        network_size: Number of nodes per run
        base_seed: Base seed for reproducibility
        config_overrides: Dict of SimulationConfig field overrides
        max_workers: Max parallel workers (None = cpu_count)

    Returns:
        MonteCarloResult with aggregated statistics
    """
    import time as _time
    import multiprocessing

    if active_platforms is None:
        active_platforms = [seed_platform]
    if config_overrides is None:
        config_overrides = {}

    # Resolve media_type from platform default if not in overrides
    if "media_type" not in config_overrides:
        config_overrides["media_type"] = PLATFORM_DEFAULT_MEDIA.get(seed_platform, "text")

    # Build serializable config dict for workers
    base_config = {
        "network_size": network_size,
        "seed_platform": seed_platform.value,  # serialize enum
        "active_platforms": [p.value for p in active_platforms],
        "scenario": scenario,
    }
    # Apply overrides (serialize enums if present)
    for key, val in config_overrides.items():
        if isinstance(val, Platform):
            base_config[key] = val.value
        elif isinstance(val, list) and val and isinstance(val[0], Platform):
            base_config[key] = [p.value for p in val]
        else:
            base_config[key] = val

    # Prepare worker args with permuted seed sequence (Fix 10)
    seed_rng = np.random.default_rng(base_seed)
    seed_sequence = seed_rng.permutation(n_runs * 10)[:n_runs] + base_seed
    worker_args = []
    for i in range(n_runs):
        cfg = base_config.copy()
        cfg["master_seed"] = int(seed_sequence[i])
        worker_args.append({"config_dict": cfg, "run_index": i})

    # Run simulations
    start_time = _time.perf_counter()
    raw_results = []

    if n_runs <= 50:
        # Sequential for small batches
        for i, args in enumerate(worker_args):
            r = _mc_worker(args)
            raw_results.append(r)
            if (i + 1) % 10 == 0:
                rates = [rr["final_infection_rate"] for rr in raw_results]
                mean = np.mean(rates)
                print(f"  [{i+1}/{n_runs}] running mean: {mean:.1%}")
    else:
        # Parallel with multiprocessing
        if max_workers is None:
            max_workers = max(1, multiprocessing.cpu_count() - 1)

        print(f"Monte Carlo: {n_runs} runs, {network_size} nodes, {max_workers} workers")
        with multiprocessing.Pool(max_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_mc_worker, worker_args)):
                raw_results.append(r)
                if (i + 1) % 50 == 0:
                    rates = [rr["final_infection_rate"] for rr in raw_results]
                    mean = np.mean(rates)
                    std = np.std(rates)
                    ci = 1.96 * std / np.sqrt(len(rates))
                    print(f"  [{i+1}/{n_runs}] mean: {mean:.1%} +/- {ci:.1%}")

    total_time = _time.perf_counter() - start_time

    # Sort by run_index for reproducibility
    raw_results.sort(key=lambda r: r["run_index"])

    # Aggregate results
    infection_rates = np.array([r["final_infection_rate"] for r in raw_results])
    termination_times = np.array([r["termination_time"] for r in raw_results])
    r0_values = np.array([r["r0_final"] for r in raw_results])

    # Death type counts
    death_counts = {}
    for r in raw_results:
        dt = r["death_type"]
        death_counts[dt] = death_counts.get(dt, 0) + 1

    # Running means for convergence plot
    running = []
    for i in range(1, len(infection_rates) + 1):
        subset = infection_rates[:i]
        m = np.mean(subset)
        s = np.std(subset)
        ci = 1.96 * s / np.sqrt(i) if i > 1 else 0
        running.append((i, float(m), float(m - ci), float(m + ci)))

    # Compute tipping points per run (max derivative of infection timeline)
    tipping_pts = []
    for r in raw_results:
        timeline = r["infection_timeline"]
        if len(timeline) >= 2:
            times = [t for t, _ in timeline]
            rates = [rate for _, rate in timeline]
            max_deriv_t = times[0]
            max_deriv = 0
            for j in range(1, len(times)):
                dt = times[j] - times[j-1]
                if dt > 0:
                    deriv = (rates[j] - rates[j-1]) / dt
                    if deriv > max_deriv:
                        max_deriv = deriv
                        max_deriv_t = (times[j] + times[j-1]) / 2
            tipping_pts.append(max_deriv_t)
        else:
            tipping_pts.append(0.0)
    tipping_points = np.array(tipping_pts)

    # Build lightweight SimulationResult list (no event_logs)
    lightweight_results = []
    for r in raw_results:
        sr = SimulationResult(
            termination_reason=r["termination_reason"],
            termination_time=r["termination_time"],
            final_infection_rate=r["final_infection_rate"],
            peak_infection_rate=r["peak_infection_rate"],
            peak_time=r["peak_time"],
            total_mutations=r["total_mutations"],
            total_platform_hops=r["total_platform_hops"],
            total_bots_detected=r["total_bots_detected"],
            total_rewiring_events=r["total_rewiring_events"],
            total_super_spreader_events=r["total_super_spreader_events"],
            crisis_occurred=r["crisis_occurred"],
            r0_final=r["r0_final"],
            death_type=DeathType(r["death_type"]),
            infection_timeline=r["infection_timeline"],
            r0_timeline=r["r0_timeline"],
            checkpoints=r["checkpoints"],
            platform_infection_rates={Platform(k): v for k, v in r["platform_infection_rates"].items()},
            platform_node_counts={Platform(k): v for k, v in r["platform_node_counts"].items()},
        )
        lightweight_results.append(sr)

    mc = MonteCarloResult(
        n_runs=n_runs,
        base_seed=base_seed,
        scenario=scenario,
        platform=seed_platform.value,
        network_size=network_size,
        infection_rates=infection_rates,
        termination_times=termination_times,
        r0_values=r0_values,
        tipping_points=tipping_points,
        death_type_counts=death_counts,
        running_means=running,
        results=lightweight_results,
        mean_infection=float(np.mean(infection_rates)),
        std_infection=float(np.std(infection_rates)),
        ci_95_lower=float(np.mean(infection_rates) - 1.96 * np.std(infection_rates) / np.sqrt(n_runs)),
        ci_95_upper=float(np.mean(infection_rates) + 1.96 * np.std(infection_rates) / np.sqrt(n_runs)),
        median_infection=float(np.median(infection_rates)),
        mean_r0=float(np.mean(r0_values)),
        mean_termination_time=float(np.mean(termination_times)),
        total_time_seconds=total_time,
        avg_time_per_run=total_time / n_runs,
        config_overrides=config_overrides,
    )

    print(f"\nMonte Carlo complete: {n_runs} runs in {total_time:.1f}s ({total_time/n_runs:.2f}s/run)")
    print(f"  Mean infection: {mc.mean_infection:.1%} +/- {mc.ci_95_upper - mc.mean_infection:.1%}")
    print(f"  Median: {mc.median_infection:.1%}, Mean R0: {mc.mean_r0:.2f}")
    print(f"  Death types: {mc.death_type_counts}")

    return mc


# =============================================================================
# Phase 3: Analysis Utility Functions
# =============================================================================

def compute_tipping_point(infection_timeline: list) -> float:
    """
    Compute tipping point: time of max derivative of infection rate (spec section 6.2).

    Args:
        infection_timeline: [(time, infection_rate), ...]
    Returns:
        Time in seconds at which infection was spreading fastest
    """
    if len(infection_timeline) < 2:
        return 0.0
    max_deriv = 0.0
    max_deriv_time = 0.0
    for i in range(1, len(infection_timeline)):
        t1, r1 = infection_timeline[i - 1]
        t2, r2 = infection_timeline[i]
        dt = t2 - t1
        if dt > 0:
            deriv = (r2 - r1) / dt
            if deriv > max_deriv:
                max_deriv = deriv
                max_deriv_time = (t1 + t2) / 2
    return max_deriv_time


def compute_point_of_no_return(mc_result: MonteCarloResult) -> float:
    """
    Compute "point of no return" (spec section 6.2): the infection % beyond which
    correction becomes ineffective.

    Algorithm: Sweep X from 5% to 80%. For each X, among runs where any early
    checkpoint exceeded X, check if >= 95% end with final_infection > 50%.
    The smallest such X is the point of no return.

    Returns:
        Threshold as a fraction (0.0-1.0), or 1.0 if none found.
    """
    for x_pct in range(5, 81):
        x = x_pct / 100.0
        # Find runs where any early checkpoint (1h, 2h, 4h) exceeded X
        exceeded_runs = []
        for result in mc_result.results:
            early_exceeded = False
            for cp in result.checkpoints:
                if cp.time <= 14400:  # first 3 checkpoints (1h, 2h, 4h)
                    if cp.infection_rate > x:
                        early_exceeded = True
                        break
            if early_exceeded:
                exceeded_runs.append(result.final_infection_rate)

        if len(exceeded_runs) >= 10:  # need enough data
            above_50 = sum(1 for r in exceeded_runs if r > 0.5)
            if above_50 / len(exceeded_runs) >= 0.95:
                return x
    return 1.0  # no point of no return found


def run_sensitivity_sweep(
    param_name: str,
    values: list,
    n_runs_per_value: int = 200,
    scenario: str = "celebrity",
    seed_platform: Platform = Platform.TWITTER,
    network_size: int = 500,
    base_seed: int = 42,
    **mc_kwargs,
) -> dict:
    """
    Run a sensitivity sweep over one parameter (spec section 6.9).

    Args:
        param_name: Name of SimulationConfig field to vary
        values: List of values to try
        n_runs_per_value: MC runs per value
    Returns:
        {value: MonteCarloResult}
    """
    results = {}
    for val in values:
        print(f"\n--- Sweep {param_name}={val} ---")
        overrides = {param_name: val}
        mc = run_monte_carlo(
            n_runs=n_runs_per_value,
            scenario=scenario,
            seed_platform=seed_platform,
            network_size=network_size,
            base_seed=base_seed,
            config_overrides=overrides,
            **mc_kwargs,
        )
        results[val] = mc
    return results


def run_counterfactual_analysis(
    baseline_result: SimulationResult,
    n_runs: int = 200,
    scenario: str = "celebrity",
    seed_platform: Platform = Platform.TWITTER,
    network_size: int = 500,
    base_seed: int = 42,
    **mc_kwargs,
) -> dict:
    """
    Run 10-scenario counterfactual analysis (spec section 6.8).

    Tier 1 (toggle-based): 7 scenarios disabling individual mechanics.
    Tier 2 (surgical): 3 scenarios using node_data_snapshot from detailed baseline.

    Args:
        baseline_result: A detailed-tracking SimulationResult for surgical counterfactuals
        n_runs: MC runs per scenario
    Returns:
        {scenario_label: MonteCarloResult}
    """
    results = {}

    # Run baseline MC
    print("\n=== Counterfactual: Baseline ===")
    results["baseline"] = run_monte_carlo(
        n_runs=n_runs, scenario=scenario, seed_platform=seed_platform,
        network_size=network_size, base_seed=base_seed, **mc_kwargs,
    )

    # Tier 1: Toggle-based counterfactuals
    toggle_scenarios = {
        "no_bots": {"bot_detection_enabled": False},
        "no_rewiring": {"rewiring_enabled": False},
        "no_corrections": {"correction_enabled": False},
        "no_attention_budget": {"attention_budget_toggle": False},
        "no_algo_amp": {"algorithmic_amplification_multiplier": 0.0},
        "no_framing": {"framing_bonus_enabled": False},
        "single_platform": {"active_platforms": [seed_platform]},
    }
    for label, overrides in toggle_scenarios.items():
        print(f"\n=== Counterfactual: {label} ===")
        results[label] = run_monte_carlo(
            n_runs=n_runs, scenario=scenario, seed_platform=seed_platform,
            network_size=network_size, base_seed=base_seed,
            config_overrides=overrides, **mc_kwargs,
        )

    # Tier 2: Surgical counterfactuals (need node_data_snapshot)
    snapshot = baseline_result.node_data_snapshot if baseline_result else None

    if snapshot:
        # 2a. Remove top 3 bridge nodes — computed per-run via betweenness centrality
        print("\n=== Counterfactual: remove_top_3_bridges (per-run betweenness) ===")
        results["remove_top_3_bridges"] = run_monte_carlo(
            n_runs=n_runs, scenario=scenario, seed_platform=seed_platform,
            network_size=network_size, base_seed=base_seed,
            config_overrides={"remove_top_n_bridges": 3},
            **mc_kwargs,
        )

        # 2b. First influencer rejected
        influencers_infected = [
            (nid, data["infected_at"])
            for nid, data in snapshot.items()
            if data["agent_type"] == "influencer" and data["infected_at"] is not None
        ]
        if influencers_infected:
            influencers_infected.sort(key=lambda x: x[1])
            first_inf_id = influencers_infected[0][0]
            print(f"\n=== Counterfactual: first_influencer_rejected (node {first_inf_id}) ===")
            results["first_influencer_rejected"] = run_monte_carlo(
                n_runs=n_runs, scenario=scenario, seed_platform=seed_platform,
                network_size=network_size, base_seed=base_seed,
                config_overrides={"block_first_influencer": True},
                **mc_kwargs,
            )

        # 2c. Bots detected 1h earlier
        print("\n=== Counterfactual: bots_detected_earlier ===")
        results["bots_detected_earlier"] = run_monte_carlo(
            n_runs=n_runs, scenario=scenario, seed_platform=seed_platform,
            network_size=network_size, base_seed=base_seed,
            config_overrides={"bot_detection_rate_multiplier": 2.0},
            **mc_kwargs,
        )

    return results


def compute_network_autopsy(result: SimulationResult) -> dict:
    """
    Analyze a detailed simulation run for network autopsy (spec section 6.8).

    Traces critical infection path, identifies bridge nodes, and finds deadliest mutation.

    Args:
        result: SimulationResult with detailed_tracking=True
    Returns:
        Dict with autopsy data for graphs #16, #17, #18
    """
    snapshot = result.node_data_snapshot
    if not snapshot:
        return {}

    # Trace critical infection path (longest downstream chain from patient zero)
    infection_tree = {}  # {node_id: [children]}
    for nid, data in snapshot.items():
        parent = data.get("infected_by")
        if parent is not None:
            infection_tree.setdefault(parent, []).append(nid)

    # Find patient zero (infected_by is None, infected_at = 0)
    patient_zero = None
    for nid, data in snapshot.items():
        if data["infected_at"] == 0.0:
            patient_zero = nid
            break

    # BFS to find critical path (longest path in infection tree)
    def find_longest_path(root):
        if root not in infection_tree or not infection_tree[root]:
            return [root]
        best = [root]
        for child in infection_tree[root]:
            child_path = find_longest_path(child)
            if len(child_path) + 1 > len(best):
                best = [root] + child_path
        return best

    critical_path = find_longest_path(patient_zero) if patient_zero is not None else []

    # Top spreaders (by downstream_infections)
    spreaders = sorted(
        [(nid, data["downstream_infections"]) for nid, data in snapshot.items()],
        key=lambda x: -x[1]
    )[:10]

    # Bridge nodes (nodes that spread across echo chambers)
    bridge_nodes = []
    for nid, data in snapshot.items():
        if data["downstream_infections"] > 0:
            children = infection_tree.get(nid, [])
            child_chambers = set(snapshot[c]["echo_chamber_idx"] for c in children if c in snapshot)
            if len(child_chambers) > 1:  # spread to multiple chambers
                bridge_nodes.append((nid, len(child_chambers), data["downstream_infections"]))
    bridge_nodes.sort(key=lambda x: -x[2])

    # Deadliest mutation (version with most infections)
    version_infections = {}
    for nid, data in snapshot.items():
        v = data.get("rumor_version")
        if v is not None:
            version_infections[v] = version_infections.get(v, 0) + 1
    deadliest_version = max(version_infections, key=version_infections.get) if version_infections else 0

    return {
        "critical_path": critical_path,
        "top_spreaders": spreaders,
        "bridge_nodes": bridge_nodes[:10],
        "deadliest_version": deadliest_version,
        "version_infections": version_infections,
        "infection_tree": infection_tree,
        "patient_zero": patient_zero,
    }


def _apply_literacy_placement(
    nodes: list,
    strategy: str,
    pct: float,
    topic: str,
    platform_graphs: dict,
    seed_platform: Platform,
    rng: np.random.Generator,
) -> list:
    """
    Apply literacy boost to selected nodes per placement strategy (spec section 6.3).

    Args:
        strategy: 'random', 'bridge', 'influencer', 'echo_seed'
        pct: Fraction of nodes to boost (0.0 to 1.0)
        topic: Topic to boost literacy for
    Returns:
        List of boosted node IDs
    """
    n_boost = max(1, int(len(nodes) * pct))
    candidates = [n for n in nodes if n.agent_type not in (AgentType.BOT,)]

    if strategy == "random":
        selected = list(rng.choice(len(candidates), size=min(n_boost, len(candidates)), replace=False))
        selected = [candidates[i] for i in selected]

    elif strategy == "bridge":
        # Nodes with connections across multiple chambers
        scored = []
        for n in candidates:
            conns = n.platform_connections.get(seed_platform, [])
            if conns:
                neighbor_chambers = set(nodes[nid].echo_chamber_idx for nid in conns if nid < len(nodes))
                scored.append((n, len(neighbor_chambers)))
        scored.sort(key=lambda x: -x[1])
        selected = [n for n, _ in scored[:n_boost]]

    elif strategy == "influencer":
        # Highest-degree nodes
        scored = [(n, len(n.platform_connections.get(seed_platform, []))) for n in candidates]
        scored.sort(key=lambda x: -x[1])
        selected = [n for n, _ in scored[:n_boost]]

    elif strategy == "echo_seed":
        # Select top nodes from each echo chamber (distributed)
        from collections import defaultdict
        chamber_nodes = defaultdict(list)
        for n in candidates:
            chamber_nodes[n.echo_chamber_idx].append(n)
        per_chamber = max(1, n_boost // max(len(chamber_nodes), 1))
        selected = []
        for cidx, cnodes in chamber_nodes.items():
            rng.shuffle(cnodes)
            selected.extend(cnodes[:per_chamber])
        selected = selected[:n_boost]
    else:
        selected = []

    # Apply literacy boost
    for n in selected:
        current = getattr(n.literacy_vector, topic, 0.5)
        boosted = min(1.0, current + 0.4)
        setattr(n.literacy_vector, topic, boosted)

    return [n.id for n in selected]


def run_herd_immunity_sweep(
    strategies: list = None,
    literacy_pcts: list = None,
    n_runs_per_cell: int = 100,
    scenario: str = "celebrity",
    seed_platform: Platform = Platform.TWITTER,
    network_size: int = 500,
    base_seed: int = 42,
    **mc_kwargs,
) -> dict:
    """
    Run herd immunity analysis: literacy placement strategies x percentages (spec section 6.3).

    Returns:
        {strategy: {pct: MonteCarloResult}}
    """
    if strategies is None:
        strategies = ["random", "bridge", "influencer", "echo_seed"]
    if literacy_pcts is None:
        literacy_pcts = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]

    results = {}
    for strat in strategies:
        results[strat] = {}
        for pct in literacy_pcts:
            print(f"\n--- Herd Immunity: {strat} @ {pct:.0%} ---")
            overrides = {
                "literacy_placement_strategy": strat,
                "literacy_placement_pct": pct,
                "literacy_placement_topic": scenario,
            }
            mc = run_monte_carlo(
                n_runs=n_runs_per_cell,
                scenario=scenario,
                seed_platform=seed_platform,
                network_size=network_size,
                base_seed=base_seed,
                config_overrides=overrides,
                **mc_kwargs,
            )
            results[strat][pct] = mc
    return results


def sanity_check_group_a():
    """
    GROUP A SANITY CHECK: Generate all 4 platform networks, print:
    - Node counts per platform
    - Agent type distribution
    - Avg connection count per agent type (verify ranges)
    - Echo chamber cluster assignments
    """
    import time as _time

    print("=" * 70)
    print("GROUP A SANITY CHECK: Multi-Platform Network Generation")
    print("=" * 70)

    config = SimulationConfig(
        network_size=500,
        active_platforms=[Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT],
        seed_platform=Platform.TWITTER,
        master_seed=42,
    )
    rng = np.random.default_rng(config.master_seed)

    start = _time.perf_counter()

    # Generate nodes with multi-platform memberships
    nodes, chamber_assignments, chamber_centers = generate_all_nodes(config, rng)

    # Generate all platform networks
    platform_graphs = generate_all_platform_networks(
        config, nodes, chamber_assignments, chamber_centers, rng
    )

    elapsed = _time.perf_counter() - start
    print(f"\nNetwork generation completed in {elapsed:.2f}s")

    # --- Node counts per platform ---
    print(f"\n--- Node counts per platform (total nodes: {len(nodes)}) ---")
    for plat in config.active_platforms:
        count = sum(1 for n in nodes if plat in n.platforms)
        print(f"  {plat.value:>10}: {count} nodes ({count / len(nodes):.1%})")

    # --- Multi-platform distribution ---
    print(f"\n--- Multi-platform distribution ---")
    from collections import Counter
    plat_count_dist = Counter(len(n.platforms) for n in nodes)
    for k in sorted(plat_count_dist.keys()):
        pct = plat_count_dist[k] / len(nodes)
        expected = MULTI_PLATFORM_DISTRIBUTION.get(k, 0)
        print(f"  {k} platform(s): {plat_count_dist[k]} nodes ({pct:.1%}) [expected ~{expected:.0%}]")

    # --- Agent type distribution ---
    print(f"\n--- Agent type distribution ---")
    type_counts = Counter(n.agent_type for n in nodes)
    for atype in AgentType:
        count = type_counts[atype]
        expected_pct = AGENT_TYPE_CONFIG[atype]["population_pct"]
        print(f"  {atype.value:>15}: {count} ({count / len(nodes):.1%}) [expected ~{expected_pct:.0%}]")

    # --- Avg connection count per agent type per platform ---
    print(f"\n--- Avg connections per agent type per platform ---")
    for plat in config.active_platforms:
        print(f"\n  Platform: {plat.value}")
        for atype in AgentType:
            plat_nodes = [
                n for n in nodes
                if n.agent_type == atype and plat in n.platforms
            ]
            if not plat_nodes:
                print(f"    {atype.value:>15}: (no nodes)")
                continue
            degrees = [len(n.platform_connections.get(plat, [])) for n in plat_nodes]
            avg_deg = sum(degrees) / len(degrees)
            min_deg = min(degrees)
            max_deg = max(degrees)
            expected_range = AGENT_TYPE_CONFIG[atype]["connections"]
            print(f"    {atype.value:>15}: avg={avg_deg:6.1f}  min={min_deg:4d}  max={max_deg:4d}  [spec: {expected_range[0]}-{expected_range[1]}]")

    # --- Echo chamber cluster assignments ---
    print(f"\n--- Echo chamber distribution ---")
    chamber_counts = Counter(n.echo_chamber_idx for n in nodes)
    for cidx in sorted(chamber_counts.keys()):
        center = chamber_centers[cidx]
        center_str = ", ".join(f"{v:.2f}" for v in center)
        print(f"  Chamber {cidx}: {chamber_counts[cidx]} nodes  center=[{center_str}]")

    # --- Graph topology stats ---
    print(f"\n--- Graph topology stats ---")
    for plat in config.active_platforms:
        G = platform_graphs[plat]
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        if n_nodes > 0:
            avg_degree = 2 * n_edges / n_nodes
            is_connected = nx.is_connected(G)
            # Clustering coefficient
            cc = nx.average_clustering(G)
            print(f"  {plat.value:>10}: {n_nodes} nodes, {n_edges} edges, "
                  f"avg_degree={avg_degree:.1f}, clustering={cc:.3f}, connected={is_connected}")
        else:
            print(f"  {plat.value:>10}: empty graph")

    # --- Quick single-run simulation test ---
    print(f"\n--- Quick simulation run (seed_platform=Twitter, all 4 active) ---")
    start = _time.perf_counter()
    result = run_single_simulation(
        scenario="celebrity",
        seed_platform=Platform.TWITTER,
        active_platforms=[Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT],
        network_size=500,
        seed=42,
    )
    elapsed = _time.perf_counter() - start
    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Termination: {result.termination_reason} at t={_fmt_duration(result.termination_time)}")
    print(f"  Final infection rate: {result.final_infection_rate:.1%}")
    print(f"  Believing: {result.total_believing}, Silent: {result.total_silent_believers}")
    print(f"  Corrected: {result.total_corrected}, Unaware: {result.total_unaware}")
    print(f"  R0 (final): {result.r0_final:.2f}")
    print(f"  Mutations: {result.total_mutations}")
    print(f"  Death type: {result.death_type.value}")
    print(f"  Events logged: {len(result.event_log)}")

    if result.platform_infection_rates:
        print(f"\n  Per-platform infection rates:")
        for plat, rate in result.platform_infection_rates.items():
            count = result.platform_node_counts.get(plat, 0)
            print(f"    {plat.value:>10}: {rate:.1%} ({count} nodes)")

    print("\n" + "=" * 70)
    print("GROUP A SANITY CHECK PASSED - no crashes, stats printed")
    print("=" * 70)


def sanity_check_group_b():
    """
    GROUP B SANITY CHECK: Platform-specific mechanics.
    Run single simulations on each platform individually,
    verify platform-specific behaviors fire correctly.
    """
    import time as _time

    print("=" * 70)
    print("GROUP B SANITY CHECK: Platform-Specific Mechanics")
    print("=" * 70)

    scenarios = ["celebrity", "financial", "health", "campus"]
    platforms = [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]

    for plat in platforms:
        print(f"\n--- {plat.value.upper()} (single-platform, celebrity scenario) ---")
        start = _time.perf_counter()
        result = run_single_simulation(
            scenario="celebrity",
            seed_platform=plat,
            active_platforms=[plat],
            network_size=500,
            seed=42,
        )
        elapsed = _time.perf_counter() - start

        print(f"  Time: {elapsed:.2f}s | Term: {result.termination_reason} at t={_fmt_duration(result.termination_time)}")
        print(f"  Infection: {result.final_infection_rate:.1%} | Corrected: {result.total_corrected} | Mutations: {result.total_mutations}")

        # Count platform-specific events
        wa_self_corr = sum(1 for e in result.event_log if e.event_type == "whatsapp_self_correction")
        mod_removes = sum(1 for e in result.event_log if e.event_type == "reddit_mod_remove")
        mod_pins = sum(1 for e in result.event_log if e.event_type == "reddit_mod_pin_correction")
        mod_quarantine = sum(1 for e in result.event_log if e.event_type == "reddit_mod_quarantine")
        amp_events = sum(1 for e in result.event_log if e.event_type == "algorithmic_amplification")

        if plat == Platform.WHATSAPP:
            print(f"  WhatsApp self-corrections: {wa_self_corr}")
        elif plat == Platform.REDDIT:
            print(f"  Reddit mod actions: remove={mod_removes}, pin={mod_pins}, quarantine={mod_quarantine}")
        elif plat in (Platform.TWITTER, Platform.INSTAGRAM):
            print(f"  Algorithmic amplification events: {amp_events}")

    # Test all 4 scenarios
    # ISSUE H NOTE: With seed=42, typically 3/4 scenarios show corrected=0.
    # This is expected at N=500: corrections require a fact-checker (5% of nodes)
    # to be exposed AND generate a correction before termination. Twitter's fast
    # cascade (mu=30s) often saturates or fizzles before any FC acts. A single
    # seed is not representative — Monte Carlo (Phase 3) will sample corrections
    # across hundreds of runs. Group D tests correction mechanics directly.
    print(f"\n--- All scenarios on Twitter ---")
    for scenario in scenarios:
        result = run_single_simulation(
            scenario=scenario,
            seed_platform=Platform.TWITTER,
            active_platforms=[Platform.TWITTER],
            network_size=500,
            seed=42,
        )
        print(f"  {scenario:>12}: infection={result.final_infection_rate:.1%}, "
              f"corrected={result.total_corrected}, death={result.death_type.value}")

    print("\n" + "=" * 70)
    print("GROUP B SANITY CHECK PASSED - no crashes, stats printed")
    print("=" * 70)


def sanity_check_group_c():
    """
    GROUP C SANITY CHECK: Cross-platform hopping and correction mechanics.
    Verify hops occur, corrections follow, emergency injection works.
    """
    import time as _time

    print("=" * 70)
    print("GROUP C SANITY CHECK: Cross-Platform & Correction Mechanics")
    print("=" * 70)

    # Test 1: Multi-platform run — verify hops (use seed=10 to differ from Group A)
    print(f"\n--- Test 1: Multi-platform with hops (seed=10) ---")
    start = _time.perf_counter()
    result = run_single_simulation(
        scenario="celebrity",
        seed_platform=Platform.TWITTER,
        active_platforms=[Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT],
        network_size=500,
        seed=10,
    )
    elapsed = _time.perf_counter() - start

    hop_events = sum(1 for e in result.event_log if e.event_type == "platform_hop")
    comm_notes = sum(1 for e in result.event_log if e.event_type == "twitter_community_note")
    wa_self_corr = sum(1 for e in result.event_log if e.event_type == "whatsapp_self_correction")

    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Infection: {result.final_infection_rate:.1%} | Corrected: {result.total_corrected}")
    print(f"  Platform hops: {hop_events}")
    print(f"  Community notes: {comm_notes}")
    print(f"  WhatsApp self-corrections: {wa_self_corr}")
    if result.platform_infection_rates:
        print(f"  Per-platform infection:")
        for plat, rate in result.platform_infection_rates.items():
            print(f"    {plat.value:>10}: {rate:.1%}")

    # Test 2: Emergency correction injection
    # BUG 4 FIX: Use WhatsApp (slower spread) + early injection to ensure it fires
    print(f"\n--- Test 2: Emergency correction at t=5min (WhatsApp) ---")
    config = SimulationConfig(
        network_size=500,
        active_platforms=[Platform.WHATSAPP],
        seed_platform=Platform.WHATSAPP,
        master_seed=42,
        correction_injection_time=300.0,  # 5 minutes
    )
    engine = SimulationEngine(config)
    result2 = engine.run()

    emergency_events = sum(1 for e in result2.event_log if e.event_type == "emergency_correction")
    print(f"  Infection: {result2.final_infection_rate:.1%}")
    print(f"  Corrected: {result2.total_corrected}")
    print(f"  Emergency correction events: {emergency_events}")
    print(f"  Duration: {result2.termination_time:.0f}s ({_fmt_duration(result2.termination_time)})")

    # Test 3: All 5 seed personas
    print(f"\n--- Test 3: All seed personas on Twitter ---")
    for persona in SeedPersona:
        config = SimulationConfig(
            network_size=500,
            active_platforms=[Platform.TWITTER],
            seed_platform=Platform.TWITTER,
            seed_persona=persona,
            master_seed=42,
        )
        engine = SimulationEngine(config)
        result3 = engine.run()
        print(f"  {persona.value:>15}: infection={result3.final_infection_rate:.1%}, "
              f"corrected={result3.total_corrected}")

    print("\n" + "=" * 70)
    print("GROUP C SANITY CHECK PASSED - no crashes, stats printed")
    print("=" * 70)


def sanity_check_group_d():
    """
    GROUP D SANITY CHECK: Behavioral depth mechanics.
    Verify literacy, emotional priming/fatigue, framing, and backfire all integrate.
    """
    import time as _time
    dims = ["fear", "outrage", "humor", "curiosity", "urgency"]

    print("=" * 70)
    print("GROUP D SANITY CHECK: Behavioral Depth Mechanics")
    print("=" * 70)

    # Test 1: Single-platform Twitter run — inspect nodes via engine
    print("\n--- Test 1: Twitter run with framing + backfire ---")
    start = _time.perf_counter()
    config1 = SimulationConfig(
        network_size=500,
        active_platforms=[Platform.TWITTER],
        seed_platform=Platform.TWITTER,
        scenario="celebrity",
        master_seed=42,
    )
    engine1 = SimulationEngine(config1)
    result = engine1.run()
    elapsed = _time.perf_counter() - start
    nodes = engine1.nodes  # access nodes from engine after run

    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Infection: {result.final_infection_rate:.1%} | Corrected: {result.total_corrected}")

    # Inspect nodes for emotional state changes
    primed_nodes = [n for n in nodes if any(
        getattr(n.emotional_priming, d) > 0.001 for d in dims
    )]
    fatigued_nodes = [n for n in nodes if any(
        getattr(n.emotional_fatigue, d) > 0.001 for d in dims
    )]
    infected_nodes = [n for n in nodes if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER)]

    print(f"  Infected: {len(infected_nodes)}")
    print(f"  Nodes with emotional priming > 0: {len(primed_nodes)}")
    print(f"  Nodes with emotional fatigue > 0: {len(fatigued_nodes)}")

    # Check backfire potential
    high_literacy_senders = sum(
        1 for n in nodes
        if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER)
        and n.literacy_vector.get("celebrity") > 0.7
    )
    high_cred_senders = sum(
        1 for n in nodes
        if n.status in (NodeStatus.BELIEVING, NodeStatus.SILENT_BELIEVER)
        and n.agent_type in (AgentType.FACT_CHECKER, AgentType.INFLUENCER)
    )
    print(f"  High-literacy infected (backfire sources): {high_literacy_senders}")
    print(f"  High-cred infected (FC/Influencer, backfire sources): {high_cred_senders}")

    # Test 2: Multi-platform run - verify all mechanics work together
    print("\n--- Test 2: Multi-platform with all Group D mechanics ---")
    start = _time.perf_counter()
    result2 = run_single_simulation(
        scenario="financial",
        seed_platform=Platform.TWITTER,
        active_platforms=[Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT],
        network_size=500,
        seed=99,
    )
    elapsed2 = _time.perf_counter() - start

    hop_events = sum(1 for e in result2.event_log if e.event_type == "platform_hop")
    corrections = result2.total_corrected

    print(f"  Completed in {elapsed2:.2f}s")
    print(f"  Infection: {result2.final_infection_rate:.1%} | Corrected: {corrections}")
    print(f"  Platform hops: {hop_events}")
    if result2.platform_infection_rates:
        for plat, rate in result2.platform_infection_rates.items():
            print(f"    {plat.value:>10}: {rate:.1%}")

    # Test 3: Verify emotional priming decays over time
    print("\n--- Test 3: Priming decay check (WhatsApp, lower spread) ---")
    start = _time.perf_counter()
    config3 = SimulationConfig(
        network_size=500,
        active_platforms=[Platform.WHATSAPP],
        seed_platform=Platform.WHATSAPP,
        scenario="health",
        master_seed=77,
    )
    engine3 = SimulationEngine(config3)
    result3 = engine3.run()
    elapsed3 = _time.perf_counter() - start
    nodes3 = engine3.nodes

    low_priming = sum(1 for n in nodes3 if all(
        getattr(n.emotional_priming, d) < 0.01 for d in dims
    ))
    any_priming = len(nodes3) - low_priming

    print(f"  Completed in {elapsed3:.2f}s")
    print(f"  Infection: {result3.final_infection_rate:.1%} | Corrected: {result3.total_corrected}")
    print(f"  Nodes with residual priming: {any_priming}")
    print(f"  Nodes with priming decayed to ~0: {low_priming}")

    # Test 4: 5-seed consistency check
    print("\n--- Test 4: 5-seed consistency with Group D ---")
    rates = []
    for s in range(5):
        r = run_single_simulation(
            scenario="celebrity",
            seed_platform=Platform.TWITTER,
            network_size=500,
            seed=s * 11 + 1,
        )
        rates.append(r.final_infection_rate)
        print(f"  Seed {s * 11 + 1:>3}: infection={r.final_infection_rate:.1%}")

    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    print(f"  Mean: {mean_rate:.1%}, Std: {std_rate:.1%}")

    print("\n" + "=" * 70)
    print("GROUP D SANITY CHECK PASSED - no crashes, stats printed")
    print("=" * 70)


def sanity_check_group_e():
    """
    GROUP E SANITY CHECK: Network dynamics, agents, demographics, crisis.
    """
    import time as _time

    print("=" * 70)
    print("GROUP E SANITY CHECK: Network Dynamics & Agent Systems")
    print("=" * 70)

    # Test 1: Demographics check — verify age group distribution
    print("\n--- Test 1: Demographics + Bot clusters on Twitter ---")
    start = _time.perf_counter()
    config1 = SimulationConfig(
        network_size=500,
        active_platforms=[Platform.TWITTER],
        seed_platform=Platform.TWITTER,
        scenario="celebrity",
        master_seed=42,
    )
    engine1 = SimulationEngine(config1)
    result1 = engine1.run()
    elapsed = _time.perf_counter() - start
    nodes = engine1.nodes

    age_counts = {"young": 0, "middle": 0, "older": 0}
    for n in nodes:
        age_counts[n.age_group] += 1
    print(f"  Completed in {elapsed:.2f}s")
    print(f"  Infection: {result1.final_infection_rate:.1%} | Corrected: {result1.total_corrected}")
    print(f"  Age distribution: {age_counts}")

    avg_nativity = {ag: 0.0 for ag in age_counts}
    for n in nodes:
        avg_nativity[n.age_group] += n.digital_nativity
    for ag in avg_nativity:
        if age_counts[ag] > 0:
            avg_nativity[ag] /= age_counts[ag]
    print(f"  Avg digital nativity: young={avg_nativity['young']:.2f}, "
          f"middle={avg_nativity['middle']:.2f}, older={avg_nativity['older']:.2f}")

    bots = [n for n in nodes if n.agent_type == AgentType.BOT]
    clusters = set(n.bot_cluster_id for n in bots if n.bot_cluster_id is not None)
    detected = sum(1 for n in bots if n.detected)
    print(f"  Bots: {len(bots)}, Clusters: {len(clusters)} (seed=42), Detected: {detected}")
    print(f"  Bots detected (result): {result1.total_bots_detected}")

    # ISSUE 8 FIX: Verify bot cluster Uniform(1,3) distribution across 20 seeds
    cluster_dist = []
    for bs in range(20):
        bc = SimulationConfig(
            network_size=500, active_platforms=[Platform.TWITTER],
            seed_platform=Platform.TWITTER, master_seed=bs,
        )
        be = SimulationEngine(bc)
        be.setup()
        bb = [n for n in be.nodes if n.agent_type == AgentType.BOT]
        nc = len(set(n.bot_cluster_id for n in bb if n.bot_cluster_id is not None))
        cluster_dist.append(nc)
    from collections import Counter
    print(f"  Bot cluster counts across 20 seeds: {dict(Counter(cluster_dist))}")

    # Test 2: Multi-platform with rewiring
    # Use seed=8 which reliably produces >50% infection + unfollows + seeks
    print("\n--- Test 2: Multi-platform with rewiring ---")
    start = _time.perf_counter()
    result2 = run_single_simulation(
        scenario="celebrity",
        seed_platform=Platform.TWITTER,
        active_platforms=[Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT],
        network_size=500,
        seed=8,
    )
    elapsed2 = _time.perf_counter() - start

    unfollow_events = sum(1 for e in result2.event_log if e.event_type == "unfollow")
    seek_events = sum(1 for e in result2.event_log if e.event_type == "seek_connections")
    sse_events = sum(1 for e in result2.event_log if e.event_type == "super_spreader_event")

    print(f"  Completed in {elapsed2:.2f}s")
    print(f"  Infection: {result2.final_infection_rate:.1%} | Corrected: {result2.total_corrected}")
    print(f"  Rewiring total: {result2.total_rewiring_events}")
    print(f"    Unfollows: {unfollow_events}, Seeks: {seek_events}")
    print(f"  Super spreader events: {sse_events}")
    print(f"  Platform hops: {result2.total_platform_hops}")
    if result2.platform_infection_rates:
        for plat, rate in result2.platform_infection_rates.items():
            print(f"    {plat.value:>10}: {rate:.1%}")

    # Test 3: Crisis system — force crisis at t=5min to guarantee it fires
    # BUG 12 FIX: Use explicit early crisis_time
    print("\n--- Test 3: Crisis system (forced at t=5min) ---")
    start = _time.perf_counter()
    config3 = SimulationConfig(
        network_size=500,
        active_platforms=[Platform.WHATSAPP],
        seed_platform=Platform.WHATSAPP,
        scenario="health",
        master_seed=55,
        crisis_enabled=True,
        crisis_time=300.0,       # 5 minutes
        crisis_duration=7200.0,  # 2 hours
        crisis_intensity=0.6,
    )
    engine3 = SimulationEngine(config3)
    result3 = engine3.run()
    elapsed3 = _time.perf_counter() - start

    crisis_starts = sum(1 for e in result3.event_log if e.event_type == "crisis_start")
    crisis_ends = sum(1 for e in result3.event_log if e.event_type == "crisis_end")

    print(f"  Completed in {elapsed3:.2f}s")
    print(f"  Infection: {result3.final_infection_rate:.1%} | Corrected: {result3.total_corrected}")
    print(f"  Crisis events: starts={crisis_starts}, ends={crisis_ends}")
    print(f"  Crisis occurred: {result3.crisis_occurred}")

    # Test 4: 5-seed consistency
    print("\n--- Test 4: 5-seed consistency with all Group E ---")
    rates = []
    for s in range(5):
        r = run_single_simulation(
            scenario="celebrity",
            seed_platform=Platform.TWITTER,
            network_size=500,
            seed=s * 13 + 1,
        )
        rates.append(r.final_infection_rate)
        print(f"  Seed {s * 13 + 1:>3}: infection={r.final_infection_rate:.1%}, "
              f"bots_det={r.total_bots_detected}, rewire={r.total_rewiring_events}")

    mean_rate = np.mean(rates)
    std_rate = np.std(rates)
    print(f"  Mean: {mean_rate:.1%}, Std: {std_rate:.1%}")

    print("\n" + "=" * 70)
    print("GROUP E SANITY CHECK PASSED - no crashes, stats printed")
    print("=" * 70)


def sanity_check_group_f():
    """
    GROUP F SANITY CHECK: Time-of-day effects, media type impact, and
    WhatsApp forward-limit mechanic.

    NOTE (Issue 4): Infection rates are highly bimodal (~30-40% of runs fizzle
    below 5%, remainder typically reach 50-90%). This is realistic (power-law
    cascade dynamics) but means:
      - Monte Carlo analysis (Phase 3) needs 1000+ runs for stable estimates
      - Website visualization (Phase 4-5) should either curate seeds or let
        users re-roll
      - Confidence intervals will be wide until N > 500 runs

    NOTE (Issue 5): Multi-platform runs only produce 2-3 death types (corrected,
    starved, occasionally saturated). When all 4 platforms are active, corrections
    converge from multiple sources (community notes + moderators + FCs), making
    saturation/mutation/decay very unlikely.  This is expected behavior — cross-
    platform correction is faster than single-platform.
    """
    import time as _time
    from collections import Counter

    print("=" * 70)
    print("GROUP F SANITY CHECK: Time-of-Day, Media Type, Forward Limit")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # TEST 1: Time-of-Day Wave Pattern (Issue 2)
    # -----------------------------------------------------------------------
    print("\n--- TEST 1: Time-of-Day Wave Pattern ---")

    # Aggregate hourly infections across 10 WhatsApp seeds starting at 10pm.
    # WhatsApp has the slowest service time (Exp(mu=10min)), so spread takes
    # several hours, making the hourly pattern visible.
    print("  Aggregate hourly infections (10 WhatsApp seeds, start_hour=22)\n")

    hourly_infections: dict[int, int] = {h: 0 for h in range(24)}
    for s in range(10):
        r22 = run_single_simulation(
            scenario="celebrity",
            seed_platform=Platform.WHATSAPP,
            active_platforms=[Platform.WHATSAPP],
            network_size=500,
            seed=s,
            start_hour=22,
        )
        for ev in r22.event_log:
            if ev.event_type == "node_infected":
                sim_hour = int((22 + ev.time / 3600) % 24)
                hourly_infections[sim_hour] += 1

    print("  Hour | Infections | Bar")
    print("  -----+------------+----")
    max_inf = max(hourly_infections.values()) if hourly_infections else 1
    for h in range(24):
        c = hourly_infections[h]
        bar = "#" * int(c / max(max_inf, 1) * 30)
        print(f"    {h:02d} | {c:>10} | {bar}")

    # Check: night hours (0-5) should have fewer infections than evening (17-21)
    night = sum(hourly_infections[h] for h in range(0, 6))
    peak = sum(hourly_infections[h] for h in range(17, 22))
    print(f"\n  Night (0-5 AM): {night} infections")
    print(f"  Peak  (5-10 PM): {peak} infections")
    if peak > night:
        print("  >> PASS: Peak hours > Night hours")
    else:
        print("  >> NOTE: Peak not > Night (small network saturates fast)")

    # Key comparison: 10am seed vs 3am seed, infection at t=2h on Twitter
    print("\n  Key validation: 10am vs 3am seed (5-seed avg, infection at t=2h)")
    for label, hour in [("10am", 10), ("3am", 3)]:
        rates_at_2h = []
        for s in range(5):
            r = run_single_simulation(
                scenario="celebrity",
                seed_platform=Platform.TWITTER,
                active_platforms=[Platform.TWITTER],
                network_size=500,
                seed=s + 100,
                start_hour=hour,
            )
            # Find infection rate closest to t=2h (7200s)
            rate_2h = 0.0
            for t, rate in r.infection_timeline:
                if t <= 7200:
                    rate_2h = rate
                else:
                    break
            rates_at_2h.append(rate_2h)
        avg = sum(rates_at_2h) / len(rates_at_2h)
        print(f"    {label:>4} seed: avg infection at t=2h = {avg:.1%}  "
              f"(per-seed: {', '.join(f'{x:.1%}' for x in rates_at_2h)})")

    # -----------------------------------------------------------------------
    # TEST 2: Media Type Impact on Virality (Issue 3)
    # -----------------------------------------------------------------------
    print("\n--- TEST 2: Media Type Impact on Virality ---")
    num_media_seeds = 10

    # Instagram: text vs image vs reel
    print(f"\n  Instagram ({num_media_seeds} seeds each):")
    ig_results = {}
    for mt in ["text", "image", "reel"]:
        rates = []
        for s in range(num_media_seeds):
            r = run_single_simulation(
                scenario="celebrity",
                seed_platform=Platform.INSTAGRAM,
                active_platforms=[Platform.INSTAGRAM],
                network_size=500,
                seed=s,
                media_type=mt,
            )
            rates.append(r.final_infection_rate)
        ig_results[mt] = rates
        import numpy as _np
        print(f"    {mt:>5}: mean={_np.mean(rates):.1%}, "
              f"median={_np.median(rates):.1%}, "
              f"std={_np.std(rates):.1%}")

    ig_reel_mean = sum(ig_results["reel"]) / len(ig_results["reel"])
    ig_text_mean = sum(ig_results["text"]) / len(ig_results["text"])
    if ig_reel_mean > ig_text_mean:
        print(f"  >> PASS: Instagram reel ({ig_reel_mean:.1%}) > text ({ig_text_mean:.1%})")
    else:
        print(f"  >> WARN: Instagram reel ({ig_reel_mean:.1%}) NOT > text ({ig_text_mean:.1%})")

    # Twitter: text vs reel
    print(f"\n  Twitter ({num_media_seeds} seeds each):")
    tw_results = {}
    for mt in ["text", "reel"]:
        rates = []
        for s in range(num_media_seeds):
            r = run_single_simulation(
                scenario="celebrity",
                seed_platform=Platform.TWITTER,
                active_platforms=[Platform.TWITTER],
                network_size=500,
                seed=s,
                media_type=mt,
            )
            rates.append(r.final_infection_rate)
        tw_results[mt] = rates
        import numpy as _np
        print(f"    {mt:>5}: mean={_np.mean(rates):.1%}, "
              f"median={_np.median(rates):.1%}, "
              f"std={_np.std(rates):.1%}")

    tw_text_mean = sum(tw_results["text"]) / len(tw_results["text"])
    tw_reel_mean = sum(tw_results["reel"]) / len(tw_results["reel"])
    if tw_text_mean > tw_reel_mean:
        print(f"  >> PASS: Twitter text ({tw_text_mean:.1%}) > reel ({tw_reel_mean:.1%})")
    else:
        print(f"  >> NOTE: Twitter text ({tw_text_mean:.1%}) not > reel ({tw_reel_mean:.1%}) "
              "(algo amp may compensate)")

    # -----------------------------------------------------------------------
    # TEST 3: WhatsApp Forward Limit Mechanic (Issue 8)
    # -----------------------------------------------------------------------
    print("\n--- TEST 3: WhatsApp Forward Limit Mechanic ---")

    tagged_versions = 0
    total_versions = 0
    max_fwd_count = 0
    wa_self_corr_total = 0
    wa_infection_rates = []

    for s in range(10):
        eng = SimulationEngine(SimulationConfig(
            network_size=500,
            active_platforms=[Platform.WHATSAPP],
            seed_platform=Platform.WHATSAPP,
            scenario="celebrity",
            master_seed=s,
        ))
        result = eng.run()
        wa_infection_rates.append(result.final_infection_rate)

        # Count self-corrections
        wa_sc = sum(1 for e in result.event_log
                    if e.event_type == "whatsapp_self_correction")
        wa_self_corr_total += wa_sc

        # Check rumor version forward counts and tags
        for ver, rumor in eng.rumor_versions.items():
            total_versions += 1
            if rumor.forwarded_tag:
                tagged_versions += 1
            if rumor.forward_count > max_fwd_count:
                max_fwd_count = rumor.forward_count

    import numpy as _np
    print(f"  Across 10 WhatsApp runs:")
    print(f"    Rumor versions total: {total_versions}, "
          f"tagged 'forwarded many times': {tagged_versions}")
    print(f"    Max forward_count across all versions: {max_fwd_count}")
    print(f"    WhatsApp self-corrections: {wa_self_corr_total}")
    print(f"    Mean infection rate: {_np.mean(wa_infection_rates):.1%}")
    if tagged_versions > 0:
        print(f"  >> PASS: Forward limit mechanic active - "
              f"{tagged_versions}/{total_versions} versions tagged")
        print(f"           Tagged messages get 1.3x threshold penalty (harder to believe)")
        print(f"           Self-corrections fire on tagged messages "
              f"(5% chance per forwarded share)")
    else:
        print("  >> WARN: No versions tagged — forward limit may not trigger")

    print("\n" + "=" * 70)
    print("GROUP F SANITY CHECK COMPLETE")
    print("=" * 70)


def final_validation():
    """
    FINAL VALIDATION: 50-seed diagnostic across ALL 4 platforms.
    Comparison table + 3 individual run traces.
    """
    import time as _time

    print("=" * 70)
    print("FINAL VALIDATION: 50-Seed Diagnostic Across All 4 Platforms")
    print("=" * 70)

    platforms = [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]
    num_seeds = 50
    scenario = "celebrity"

    # Collect per-platform stats
    stats = {}
    for plat in platforms:
        results = []
        start = _time.perf_counter()
        for s in range(num_seeds):
            r = run_single_simulation(
                scenario=scenario,
                seed_platform=plat,
                active_platforms=[plat],
                network_size=500,
                seed=s,
            )
            results.append(r)
        elapsed = _time.perf_counter() - start

        infection_rates = [r.final_infection_rate for r in results]
        corrected_counts = [r.total_corrected for r in results]
        durations = [r.termination_time for r in results]
        bots_detected = [r.total_bots_detected for r in results]
        rewiring_events = [r.total_rewiring_events for r in results]
        sse_events = [r.total_super_spreader_events for r in results]

        stats[plat] = {
            "mean_inf": np.mean(infection_rates),
            "median_inf": np.median(infection_rates),
            "std_inf": np.std(infection_rates),
            "mean_corr": np.mean(corrected_counts),
            "mean_bots": np.mean(bots_detected),
            "mean_rewire": np.mean(rewiring_events),
            "mean_sse": np.mean(sse_events),
            "mean_dur": np.mean(durations),
            "elapsed": elapsed,
            "results": results,
        }

    # Print comparison table
    print(f"\n{'Platform':>12} | {'Mean Inf':>9} | {'Median':>7} | {'Std':>6} | "
          f"{'Corr':>5} | {'Bots':>5} | {'Rewire':>7} | {'SSE':>4} | {'Duration':>10} | {'Time':>6}")
    print("-" * 100)
    for plat in platforms:
        s = stats[plat]
        print(f"{plat.value:>12} | {s['mean_inf']:>8.1%} | {s['median_inf']:>6.1%} | "
              f"{s['std_inf']:>5.1%} | {s['mean_corr']:>5.1f} | {s['mean_bots']:>5.1f} | "
              f"{s['mean_rewire']:>7.1f} | {s['mean_sse']:>4.1f} | {s['mean_dur']:>8.0f}s | "
              f"{s['elapsed']:>5.1f}s")

    # Multi-platform combined run (50 seeds)
    print(f"\n--- Multi-Platform Combined (all 4 platforms active) ---")
    multi_results = []
    start = _time.perf_counter()
    for s in range(num_seeds):
        r = run_single_simulation(
            scenario=scenario,
            seed_platform=Platform.TWITTER,
            active_platforms=platforms,
            network_size=500,
            seed=s,
        )
        multi_results.append(r)
    multi_elapsed = _time.perf_counter() - start

    multi_inf = [r.final_infection_rate for r in multi_results]
    multi_hops = [r.total_platform_hops for r in multi_results]
    multi_corr = [r.total_corrected for r in multi_results]
    multi_bots = [r.total_bots_detected for r in multi_results]
    multi_rewire = [r.total_rewiring_events for r in multi_results]
    multi_sse = [r.total_super_spreader_events for r in multi_results]

    print(f"  Mean Infection: {np.mean(multi_inf):.1%} (median {np.median(multi_inf):.1%}, "
          f"std {np.std(multi_inf):.1%})")
    print(f"  Mean Corrected: {np.mean(multi_corr):.1f}")
    print(f"  Mean Hops: {np.mean(multi_hops):.1f}")
    print(f"  Mean Bots Detected: {np.mean(multi_bots):.1f}")
    print(f"  Mean Rewiring: {np.mean(multi_rewire):.1f}")
    print(f"  Mean SSE: {np.mean(multi_sse):.1f}")
    print(f"  50 runs in {multi_elapsed:.1f}s ({multi_elapsed/50:.2f}s per run)")

    # Per-platform breakdown for multi-platform runs (3 traces)
    print(f"\n--- 3 Individual Multi-Platform Run Traces ---")
    for idx in [0, 25, 49]:
        r = multi_results[idx]
        print(f"\n  Run seed={idx}:")
        print(f"    Infection: {r.final_infection_rate:.1%} | Corrected: {r.total_corrected} | "
              f"Hops: {r.total_platform_hops} | Bots: {r.total_bots_detected} | "
              f"Rewire: {r.total_rewiring_events}")
        print(f"    Duration: {r.termination_time:.0f}s ({_fmt_duration(r.termination_time)}) | "
              f"Death: {r.death_type.value} | R0: {r.r0_final:.2f}")
        if r.platform_infection_rates:
            for plat, rate in r.platform_infection_rates.items():
                plat_count = r.platform_node_counts.get(plat, 0)
                print(f"      {plat.value:>10}: {rate:.1%} ({plat_count} nodes)")

    # Death type distribution across ALL runs (4 platforms x 50 + 50 multi = 250 total)
    print(f"\n--- Death Type Distribution (250 runs) ---")
    all_results = []
    for plat in platforms:
        all_results.extend(stats[plat]["results"])
    all_results.extend(multi_results)

    from collections import Counter
    death_counts = Counter(r.death_type.value for r in all_results)
    total_runs = len(all_results)
    for dtype, count in sorted(death_counts.items(), key=lambda x: -x[1]):
        print(f"  {dtype:>15}: {count:>4} ({count/total_runs:.1%})")
    print(f"  {'TOTAL':>15}: {total_runs}")
    distinct = len(death_counts)
    print(f"  Distinct death types: {distinct}")
    if distinct >= 4:
        print("  >> TARGET MET: 4+ death types appearing!")
    else:
        print(f"  >> TARGET NOT MET: only {distinct} death types (need 4+)")

    # Per-platform death type breakdown
    print(f"\n--- Death Type by Platform ---")
    for plat in platforms:
        plat_deaths = Counter(r.death_type.value for r in stats[plat]["results"])
        parts = ", ".join(f"{d}={c}" for d, c in sorted(plat_deaths.items(), key=lambda x: -x[1]))
        print(f"  {plat.value:>12}: {parts}")
    multi_deaths = Counter(r.death_type.value for r in multi_results)
    parts = ", ".join(f"{d}={c}" for d, c in sorted(multi_deaths.items(), key=lambda x: -x[1]))
    print(f"  {'multi':>12}: {parts}")

    print("\n" + "=" * 70)
    print("FINAL VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    final_validation()
