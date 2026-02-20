# "How Fast Does a Rumor Become Truth?"
## Multi-Platform Misinformation Cascade Simulation
### Complete Project Specification v2.1

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Agent Architecture](#2-agent-architecture)
3. [Platform Models](#3-platform-models)
4. [Behavioral Mechanics](#4-behavioral-mechanics)
5. [Network Structure](#5-network-structure)
6. [Analysis Metrics](#6-analysis-metrics)
7. [Website Visualization](#7-website-visualization)
8. [Simulation & Modeling Concepts Map](#8-simulation--modeling-concepts-map)
9. [Output Deliverables](#9-output-deliverables)
10. [Tech Stack & Performance Targets](#10-tech-stack--performance-targets)
11. [Build Order & Timeline](#11-build-order--timeline)

---

## 1. Project Overview

### 1.1 Concept
Simulate how misinformation spreads through interconnected social media platforms modeled as queuing networks. A rumor is seeded on one platform, spreads through nodes (people) with different behaviors, mutates as it travels, hops across platforms, and races against corrections — all driven by probability distributions and random number generation.

### 1.2 Rumor Scenarios (User Selectable)
| Scenario | Topic Tag | Emotional Profile |
|----------|-----------|-------------------|
| "Celebrity X confirmed dead" | celebrity | fear: 0.4, outrage: 0.2, humor: 0.1, curiosity: 0.8, urgency: 0.6 |
| "Bank Y shutting down tomorrow" | financial | fear: 0.9, outrage: 0.5, humor: 0.0, curiosity: 0.3, urgency: 1.0 |
| "Drinking warm water cures disease Z" | health | fear: 0.6, outrage: 0.2, humor: 0.1, curiosity: 0.5, urgency: 0.4 |
| "University announcing surprise holiday" | campus | fear: 0.0, outrage: 0.0, humor: 0.3, curiosity: 0.7, urgency: 0.5 |

Users may also create a **custom rumor** via free-text input with manual emotion sliders:
```
Custom rumor flow:
    1. User types free text (e.g., "My professor is secretly a vampire")
    2. User adjusts 5 emotion sliders (fear, outrage, humor, curiosity, urgency) manually
    3. User selects topic tag from dropdown or types custom tag
    4. Custom text becomes the v0 seed message for the message generation system
    5. Mutations modify fragments of the original text
```

### 1.3 Media Type
Each rumor instance carries a format tag: `text`, `image`, `video`, or `reel`. Virality is NOT determined by format alone — it's:

```
virality_score = content_quality × platform_fit × emotional_impact

content_quality = random(0.1 to 1.0)  # how well-crafted the content is
```

**Platform Fit Matrix:**

| | Text | Image | Video | Reel |
|----------|------|-------|-------|------|
| WhatsApp | 0.9 | 0.7 | 0.5 | 0.3 |
| Twitter/X| 1.0 | 0.8 | 0.6 | 0.4 |
| Instagram| 0.3 | 0.9 | 0.7 | 1.0 |
| Reddit | 0.9 | 0.6 | 0.4 | 0.2 |

Content quality changes on mutation — sometimes improves, sometimes degrades.

### 1.4 Simulation Scale

```
Network size (configurable):
    Small:  500 nodes   → fast iteration, website visualization default
    Medium: 2,000 nodes → balanced fidelity, Monte Carlo default
    Large:  10,000 nodes → full statistical accuracy, Python-only (too heavy for browser)

Website visualization:
    500 nodes per platform (shared across all 4 platforms in multi-platform view)
    D3 force-directed graph performance limit
    Full-detail rendering on focused platform, simplified color-only on thumbnails

Python simulation:
    Scales to 10,000 nodes for Monte Carlo batches
    Network size is a configurable parameter for sensitivity analysis
```

### 1.5 Simulation Time & Adaptive Duration

```
Simulated time span: ADAPTIVE (up to 48 hours max)
    → captures time-of-day effects, morning waves, overnight stalls
    → terminates early when rumor dynamics resolve

Time resolution: 1 simulated second per event-driven tick
    → events processed in priority queue order, not fixed timesteps

Adaptive termination conditions (ANY triggers end):
    1. max_time reached: 48h (hard cap, safety net)
    2. rumor_dead: R₀ < 0.1 for 30+ consecutive simulated minutes
       AND no new infections in last 60 simulated minutes
    3. fully_saturated: >95% of reachable nodes infected or corrected
    4. fully_corrected: >90% of believers have been corrected
       AND active spread rate < 0.5% per hour

Early termination logging:
    termination_reason: "max_time" | "rumor_dead" | "saturated" | "corrected"
    termination_time: float (actual simulated time at end)
    
    Monte Carlo runs with different durations are STILL comparable:
    Final infection rate is measured at termination, whenever that is
    Time-to-death becomes a meaningful metric (not just "48h")

Checkpoint snapshots (for Monte Carlo consistency):
    Save simulation state at fixed checkpoints: t=1h, 2h, 4h, 8h, 12h, 24h, 48h
    Monte Carlo analysis can compare across runs at the SAME checkpoint time
    even if some runs ended early
    
    Example output: "At t=4h: mean infection = 31% ± 2.4%"
                    "At termination: mean infection = 58% ± 3.1% (avg duration: 11.2h)"

Speed-sensitive scenarios (emerge naturally from emotional profiles):
    Celebrity death: avg termination at ~6h (burns fast, corrects fast)
    Health misinformation: avg termination at ~36h (slow burn, hard to correct)
    Financial panic: avg termination at ~4h (extreme urgency, crashes fast)
    Campus rumor: avg termination at ~18h (moderate pace)

Website visualization time mapping:
    Playback duration adapts to simulation duration:
    display_seconds = min(60, simulation_hours × 1.25)
    → 6h sim = 7.5s playback, 36h sim = 45s playback, 48h sim = 60s playback
    Speed controls scale this: 0.5x, 1x, 2x, 5x, 10x
    Pause + scrub allows jumping to any simulated timestamp
```

### 1.6 Seed Persona (Patient Zero Selection)

```
The "who starts the rumor" choice affects BOTH network position AND initial credibility.
Configurable via dropdown in website, parameter in Python.

Seed personas (not new agent types — they modify the INITIAL seed event only):

| Persona | Maps To Agent Type | Initial Credibility Modifier | Tone |
|---------|-------------------|------------------------------|------|
| Random Person | Regular User | 1.0x (baseline) | Casual, uncertain |
| News Channel | Influencer | 2.5x (institutional trust) | Authoritative, breaking-news |
| Blogger/Creator | Regular User | 1.3x (moderate following) | Opinionated, personal brand |
| Celebrity | Influencer | 1.8x (fame but not news authority) | Casual, off-topic credibility |
| Anonymous Tip | Bot | 0.7x (no identity = less trust) | Cryptic, conspiratorial |

How it works:
    1. User selects persona from dropdown
    2. Simulation selects a node matching the agent type column
    3. The FIRST wave of receivers gets the credibility modifier applied:
        receiver.effective_threshold *= (1 / credibility_modifier)
        Higher modifier = lower effective threshold = easier to believe
    4. The seed message in the feed uses the persona's tone
    5. After the first hop (patient zero → first receivers), all subsequent
       spread uses the STANDARD source credibility mechanics from §4.6
    
    The persona is a STARTING FLAVOR, not a permanent simulation type.

Default: "Random Person"
Monte Carlo: uses the SAME persona across all runs for consistency
```

### 1.7 Reproducibility & Random Seeds

```
Every simulation run is seeded with a master random seed:
    master_seed: int (configurable, default = None for random)
    
    When master_seed is set:
        All numpy/random calls derive from this seed
        Network generation, node attributes, event outcomes are deterministic
        Two runs with same seed + same parameters = identical results
    
    When master_seed is None:
        A base_seed is generated ONCE from system entropy at batch start
        Each Monte Carlo run gets a unique seed: base_seed + run_index
        base_seed and all derived seeds are LOGGED for reproducibility after the fact
        This means: a batch is reproducible if you record the base_seed

Split screen / multi-platform mode:
    All platforms share the same master_seed for node attribute generation
    Platform parameters differ → differences are purely structural, not random noise
```

### 1.8 Simplifying Assumptions

```
Documented assumptions for academic transparency:

1. Equal platform sampling: All platforms simulated at equal node count (500 in browser,
   2K in Python). This represents an equal-sized sample of each platform's local network
   neighborhood, not the relative sizes of the platforms themselves.

2. Limited network dynamism: Network rewiring is constrained to small reactive adjustments
   (unfollowing aggressive sharers, seeking behavior after belief — see §4.13). Full
   dynamic network evolution (organic follower growth, algorithmic recommendations of
   new connections) is not modeled. Max 3 rewiring events per node.

3. Single rumor with attention budget: Only one rumor + one correction track at a time.
   The attention budget system (§4.12) models finite cognitive bandwidth, but competing
   rumors and general content noise are not simulated. Infection rates should be
   interpreted as upper bounds compared to a realistic multi-content environment.

4. Homogeneous time zones: All nodes share the same 24-hour activity cycle (with
   individual shifts). A global simulation would need multiple time zone offsets.

5. No platform-specific UI effects: We model platform mechanics (algorithms, forward
   limits, moderation) but not UI-specific behaviors (Instagram's grid layout affecting
   visibility, Reddit's comment threading affecting engagement depth).

6. Single-language: All nodes communicate in the same language. Cross-language spread
   dynamics are not modeled.

7. Binary belief model: Nodes believe or don't. In reality, people hold beliefs with
   varying confidence. Partially addressed by the silent believer mechanic.

8. Simplified demographics: The demographic layer (§2.6) captures age group and digital
   nativity as modifiers on existing mechanics. Full socioeconomic, education, geographic,
   and cultural modeling is out of scope. The two-variable demographic model is a
   lightweight proxy for population heterogeneity.

9. Message framing approximation: The framing bonus (§4.15) uses a small modifier based
   on message shape archetype, not natural language analysis. The actual persuasiveness
   of generated text is not computed — the shape category serves as a proxy for
   framing quality.

10. Bot coordination simplified: Bot networks use coordinated wave mechanics (§4.16) but
    do not model strategic content planning, long-term credibility farming beyond a
    simple time-based growth curve, or adaptive evasion of detection algorithms.
```

---

## 2. Agent Architecture

### 2.1 Five Agent Types

| Agent Type | Population % | Credibility Threshold | Connections | Behavior |
|------------|-------------|----------------------|-------------|----------|
| **Regular User** | 60% | Uniform(0.3, 0.8) | 5-15 | Standard Bernoulli share trial |
| **Influencer** | 5% | Uniform(0.2, 0.5) | 100-500 | Faster processing, massive reach |
| **Fact-Checker** | 3% | Fixed at 0.95 | 50-200 | Slow but always produces corrections |
| **Bot** | 7% | Fixed at 0.01 | 30-100 | Shares almost everything instantly |
| **Lurker** | 25% | Uniform(0.8, 0.99) | 3-8 | Rarely shares, mostly silent believers |

**Service Time Resolution:**
```
Agent types and platforms BOTH define service times. Resolution:
    effective_service_time = platform_base_service_time × agent_type_modifier

Agent type modifiers (multiplied against platform base):
    Regular User: 1.0x (baseline)
    Influencer:   0.2x (processes 5x faster — this is their job)
    Fact-Checker:  3.0x (processes 3x slower — doing verification)
    Bot:           0.02x (near-instant — automated)
    Lurker:        6.0x (processes very slowly — low engagement)

Example on Twitter (base μ=30sec):
    Regular User:  Exp(μ=30s)
    Influencer:    Exp(μ=6s)
    Fact-Checker:  Exp(μ=90s)
    Bot:           Exp(μ=0.6s)
    Lurker:        Exp(μ=180s)

Example on WhatsApp (base μ=10min):
    Regular User:  Exp(μ=10min)
    Influencer:    Exp(μ=2min)
    Fact-Checker:  Exp(μ=30min)
    Bot:           Exp(μ=12s)
    Lurker:        Exp(μ=60min)
```

### 2.2 Per-Node Attributes
Every node in the network carries:

```
node = {
    id: int,
    type: "regular" | "influencer" | "fact_checker" | "bot" | "lurker",
    
    # Core
    credibility_threshold: float,       # base threshold for sharing
    connections: list[node_id],         # adjacency list
    edge_weights: dict[node_id: float], # relationship strength per connection
    
    # Worldview (4D — for echo chambers + confirmation bias)
    worldview_vector: [float, float, float, float],
    # Dimensions:
    #   [0] political:       -1.0 (left) to 1.0 (right)
    #   [1] health_trust:     0.0 (skeptic of mainstream health) to 1.0 (trusts medical establishment)
    #   [2] tech_trust:       0.0 (tech paranoid) to 1.0 (early adopter, trusts tech companies)
    #   [3] authority_trust:  0.0 (anti-establishment) to 1.0 (trusts institutions)
    
    # Emotional susceptibility (DYNAMIC — see §4.14)
    susceptibility: {
        fear: float,                    # Generated from Beta(2, 5) distribution
        outrage: float,
        humor: float,
        curiosity: float,
        urgency: float
    },
    original_susceptibility: {          # Immutable copy of initial values (for priming caps + fatigue floors)
        fear: float, outrage: float, humor: float, curiosity: float, urgency: float
    },
    emotional_priming: {                # Current priming boost per emotion (decays over time, see §4.14)
        fear: float, outrage: float, humor: float, curiosity: float, urgency: float
    },
    emotional_fatigue: {                # Cumulative fatigue per emotion (permanent, see §4.14)
        fear: float, outrage: float, humor: float, curiosity: float, urgency: float
    },
    messages_processed_per_emotion: {   # Counter for fatigue threshold (see §4.14)
        fear: int, outrage: int, humor: int, curiosity: int, urgency: int
    },
    
    # Literacy (for herd immunity)
    literacy_vector: {
        health: float,
        financial: float,
        political: float,
        celebrity: float,
        campus: float,
        tech: float
    },
    
    # Multi-platform membership
    platforms: list[str],               # e.g. ["twitter", "whatsapp"] — which platforms this node exists on
    platform_node_ids: dict[str: int],  # mapping of platform → node_id on that platform
    
    # Cross-platform behavior
    hop_tendency: float,                # 0.0-1.0, how likely this node is to cross-post
                                        # Generated from Beta(2, 5) — most people low, few chronic cross-posters
    
    # Attention budget (see §4.12)
    attention_budget: float,            # 0.0-1.0, starts at 1.0, depleted by message processing
    
    # Demographics (see §2.6)
    demographic: {
        age_group: "young" | "middle" | "older",
        digital_nativity: float         # 0.0-1.0, generated from age-specific Beta distributions
    },
    
    # Time behavior
    active_hours_profile: list[float],  # 24-value array, activity probability per hour
    base_λ: float,                      # base arrival rate
    
    # Reactive rewiring (see §4.13)
    rewiring_events: int,               # counter, max 3 per simulation
    unfollowed: list[node_id],          # nodes this node has unfollowed during simulation
    
    # Bot-specific (see §4.16, only for type == "bot")
    bot_cluster_id: int | None,         # which coordinated bot network this belongs to
    apparent_credibility: float,        # starts at 0.3, builds over time
    shares_this_hour: int,              # activity counter for detection probability
    detected: bool,                     # True if platform has detected and removed this bot
    
    # State (changes during simulation)
    status: "unaware" | "believing" | "silent_believer" | "corrected" | "immune" | "removed",
    times_exposed: int,                 # for trust decay
    times_correction_seen: int,         # for correction fatigue
    effective_threshold: float,         # current threshold after all modifiers
    queue: deque,                       # message queue (max 1,000 — overflow drops oldest)
    infected_by: node_id | None,
    infected_at: float | None,
    downstream_infections: int,
    rumor_version: int                  # which mutation they received
}
```

### 2.3 Rumor Object Schema
Every rumor instance (including mutations) carries:

```
rumor = {
    id: int,                            # unique rumor identifier
    version: int,                       # mutation version number (0 = original)
    parent_version: int | None,         # parent rumor id for mutation chain
    
    # Content
    scenario: str,                      # "celebrity" | "financial" | "health" | "campus" | "political" | "tech"
    media_type: str,                    # "text" | "image" | "video" | "reel"
    content_quality: float,             # 0.1-1.0, changes on mutation
    
    # Emotional profile
    emotions: {
        fear: float,
        outrage: float,
        humor: float,
        curiosity: float,
        urgency: float
    },
    
    # Alignment (for confirmation bias — 4D worldview space)
    alignment_vector: [float, float, float, float],  # 4D worldview alignment
    # Matches node worldview dimensions: [political, health_trust, tech_trust, authority_trust]
    
    # Spread tracking
    origin_platform: str,               # platform where this version was created
    origin_node: node_id,               # node that created/mutated this version
    origin_time: float,                 # simulation time of creation
    forward_count: int,                 # how many times THIS version has been forwarded
                                        # NOTE: lives on rumor object, NOT node
                                        # Mutations reset forward_count to 0
    total_infections: int,              # downstream infections caused by this version
    
    # Platform-specific fields
    forwarded_tag: bool,                # WhatsApp: True if forward_count >= 5
    expiry_time: float | None,          # Instagram stories: current_time + 24h TTL, None for non-story
    karma_score: int,                   # Reddit: upvote/downvote score, starts at 1
    
    # Mutation tracking
    mutation_chain: list[int],          # list of version ids from original to this
    mutation_distance: float,           # euclidean distance from original emotional profile
    
    # Virality (computed)
    virality_score: float,              # content_quality × platform_fit × emotional_impact
    
    # Display (for message generation system)
    display_text: str | None            # generated message text for feed/tooltip (generated on demand)
}
```

### 2.4 Multi-Platform Node Identity

```
Cross-platform membership:
    Each node is assigned to 1-4 platforms probabilistically:
    
    platform_membership_probabilities:
        1 platform:  45% of nodes
        2 platforms: 40% of nodes
        3 platforms: 13% of nodes
        4 platforms:  2% of nodes
    
    Platform assignment is weighted by agent type:
        Regular users: equal probability across all platforms
        Influencers: biased toward Twitter (0.4) and Instagram (0.3)
        Bots: biased toward Twitter (0.5) and Reddit (0.3)
        Fact-checkers: biased toward Twitter (0.4) and Reddit (0.3)
        Lurkers: biased toward Reddit (0.3) and Instagram (0.3)
    
    A node on multiple platforms:
        Has SEPARATE connections on each platform (different graph per platform)
        Has the SAME worldview, susceptibility, literacy across platforms
        Can trigger platform hops: sees rumor on platform A, screenshots to platform B
        Is the ONLY mechanism for cross-platform spread

    Implementation:
        Each platform maintains its own networkx graph
        Nodes shared across platforms are linked by a global_id
        node.platforms = ["twitter", "whatsapp"]
        node.platform_node_ids = {"twitter": 42, "whatsapp": 107}
```

### 2.5 Message Generation System

Procedural text generation to make abstract propagation visible as realistic social media messages. This is primarily a **narrative layer** for educational immersion — making the math tangible. However, the selected message shape also applies a small **framing modifier** (see §4.15) to the sharing probability, bridging the narrative and mechanical layers.

**2.5.1 Message Archetypes (5 Rumor Shapes)**

```
Each message generated for the feed, tooltip, or node inbox selects an archetype
based on sender attributes + simulation state:

Shape 1: BARE FORWARD
    Trigger: Bot, weak-tie share, low emotional investment
    Structure: Just the content, no framing
    Examples: "[forwarded] Celebrity X found dead in hotel room"
              "⚠️ Bank Y closing all branches tomorrow"
    Length: 1 line
    Media: May include media card if media_type != text

Shape 2: REACTION
    Trigger: High emotion, strong-tie share, early exposure
    Structure: Emotional exclamation + optional one-line take
    Examples: "WHAT omg is this real??", "no way 💀💀💀", "I'm literally shaking rn"
              "bro. BRO.", "has anyone else seen this???"
    Length: 1-2 lines
    Media: Rarely includes media card (pure emotional response)

Shape 3: PERSONAL FRAME
    Trigger: Regular user, moderate emotion, sharing with context
    Structure: Personal connection + claim + optional call to action
    Examples: "my cousin works at the hospital and confirmed this is real"
              "someone I trust just told me Bank Y is shutting down, check your accounts"
    Length: 2-4 lines
    Media: Often references media without showing it ("look at this screenshot...")

Shape 4: ELABORATOR
    Trigger: Influencer, high literacy node who cracked, mutation event
    Structure: Extended take with analysis/speculation + fake details
    Examples: "Ok so I've been reading about this for an hour and here's what I think..."
              "Thread: Why the Celebrity X story is bigger than you think 🧵"
    Length: 4-8 lines
    Media: May include media card with analysis framing

Shape 5: SKEPTIC SHARER
    Trigger: High-threshold node who barely passed, conflicted share
    Structure: Hedge + content + request for verification
    Examples: "I don't know if this is true but has anyone verified this?"
              "take this with a grain of salt but my friend sent me this and..."
    Length: 2-4 lines
    Media: Sometimes includes media with skeptical framing

CORRECTION SHAPES (4 types):
    Correction Shape 1 — DEBUNK (fact-checker origin):
        "⚠️ FACT CHECK: The claim about Celebrity X is FALSE. Here's what actually happened..."
    Correction Shape 2 — RELAY (node forwarding correction):
        "turns out that thing about Bank Y was fake, someone debunked it"
    Correction Shape 3 — TOLD YOU SO (skeptic who never believed):
        "called it. knew this was fake from the start"
    Correction Shape 4 — RELUCTANT WALKBACK (previously believing node):
        "ok apparently I was wrong about that... my bad for spreading it"
```

**2.5.2 Shape Selection Logic**

```
def select_shape(sender, receiver, rumor, edge):
    if sender.type == "bot":
        return Shape.BARE_FORWARD
    
    emotional_intensity = dot_product(rumor.emotions, sender.susceptibility)
    
    if emotional_intensity > 0.7 and sender.times_exposed <= 2:
        return Shape.REACTION
    
    if sender.type == "influencer":
        return Shape.ELABORATOR
    
    if sender.effective_threshold > 0.6:  # barely passed threshold
        return Shape.SKEPTIC_SHARER
    
    if edge.relationship_strength > 0.7:  # strong tie
        return Shape.PERSONAL_FRAME
    
    # Default: weighted random
    weights = [0.3, 0.2, 0.25, 0.1, 0.15]  # bare, reaction, personal, elaborator, skeptic
    return weighted_random(shapes, weights)
```

**2.5.3 Three-Layer Phrase Pool System**

```
LAYER 1: Standalone micro-messages (complete, no assembly needed)
    Used by: Shape 1, Shape 2
    Count: ~80-100 across all emotions
    
    fear pool:     "omg", "I'm literally shaking", "this can't be real",
                   "no way", "WHAT", "💀💀💀", "I refuse to believe this",
                   "not Celebrity X 😭😭", "I'm scared ngl"
    outrage pool:  "THIS IS INSANE", "how is nobody talking about this",
                   "I'm so angry rn", "we need to do something",
                   "absolute disgrace", "I can't believe they're getting away with this"
    humor pool:    "lmaooo no way", "this is the funniest thing",
                   "💀 I can't", "bro", "you can't make this up",
                   "the timeline is unhinged today"
    curiosity pool:"wait is this actually real?", "hold on what",
                   "huh???", "👀👀👀", "has anyone confirmed this",
                   "I need more info", "source?"
    urgency pool:  "RIGHT NOW", "hurry", "don't wait",
                   "this is happening TODAY", "CHECK YOUR ACCOUNT",
                   "move fast", "🚨🚨🚨"
    
    Selection: pick from pool matching the rumor's DOMINANT emotion
    (highest value in rumor.emotions dict)

LAYER 2: Sentence-level building blocks (combinable fragments)
    Used by: Shape 3, 4, 5
    Count: ~30-40 per category, ~200-250 total
    
    personal_frames (scenario-agnostic):
        "my cousin works at [relevant_place] and confirmed this"
        "someone I trust just told me"
        "I didn't want to believe it but"
        "so apparently"
        "I just heard from a friend that"
        "a guy at work was saying"
        "my [family_member] just sent me this"
        "ok so I've been reading about this and"
    
    claim_details (PER SCENARIO):
        celebrity:  "they said it happened last night",
                    "the family hasn't made a statement yet",
                    "there are photos from the hospital",
                    "their last post was 6 hours ago, nothing since"
        financial:  "the bank's website is down",
                    "ATMs aren't dispensing cash",
                    "employees were told not to come in tomorrow",
                    "there's a line outside the branch already"
        health:     "a doctor posted about it on Twitter",
                    "the WHO hasn't denied it",
                    "my neighbor tried it and said it worked",
                    "the government is hiding the real numbers"
        campus:     "someone from admin told my friend",
                    "there's a notice going around",
                    "check the university portal",
                    "the professors were told this morning"
    
    amplifiers (scenario-agnostic):
        "and nobody is talking about it"
        "the media is covering it up"
        "this is being deleted everywhere"
        "they don't want you to know"
        "share this before it gets taken down"
    
    skeptic_hedges (scenario-agnostic):
        "I don't know if this is true but"
        "take this with a grain of salt"
        "can anyone verify this?"
        "this seems off but"
        "not sure I believe this however"
    
    call_to_actions (scenario-agnostic):
        "stay safe everyone"
        "check your accounts NOW"
        "tell your family"
        "spread the word"
        "be careful out there"

LAYER 3: Word-level swaps (prevent identical messages)
    Within any sentence, certain words randomly vary:
    "my cousin" / "my friend" / "someone I know" / "a guy at work"
    "confirmed" / "verified" / "said" / "swore"
    "look at this" / "check this out" / "have you seen this"
    
    Total authoring: ~400-500 fragments across all layers
    Output space: 10,000+ perceptually unique messages from combinatorial variety
```

**2.5.4 Platform Tone Modifiers**

```
Same message archetype, different voice per platform:

WhatsApp tone:
    Informal, family-group style
    Frequent "forwarded" tags and "⚠️" icons
    Hindi/English code-switching markers (contextual)
    "Good morning 🙏" openers on elaborated messages

Twitter/X tone:
    Short, punchy, hashtag-ready
    Thread format for elaborators ("🧵 1/")
    Quote-tweet framing for reactions
    "@" mentions and trending indicators

Instagram tone:
    Story-style urgency ("swipe up", "link in bio")
    Visual-first language ("look at this", "watch this")
    Emoji-heavy, influencer voice

Reddit tone:
    Longer, more analytical
    "Edit:" and "Update:" prefixes
    "Citation needed" in skeptic shapes
    Subreddit context: "r/[community_name]"
    Karma-aware: popular posts get "trending 🔥" indicator
```

**2.5.5 Media Type in Messages**

```
Messages reference media type naturally through text, not by rendering actual media:

How media type affects message text:
    text:  message IS the content — no media reference needed
    image: "look at this screenshot", "check this photo", "the image is going around"
    video: "did you see that video of...", "watch this 😳"
    reel:  "that reel about [topic] is everywhere", "someone made a reel about this"

Media card display (in feed panel — SIMPLE styled HTML elements, not actual media):
    Image card:   Colored gradient rectangle (tint by dominant emotion), 🖼️ icon, bold claim text
    Video card:   Same gradient, ▶ play button, duration badge, "BREAKING" tag for high-urgency
    Reel card:    Vertical 9:16 aspect ratio, 🎬 icon, view count (Instagram-origin only)
    Screenshot card: Nested card-in-card showing source platform post (platform hop artifact)
    
    Implementation: ~30-50 lines of CSS per card type
    These are simple styled HTML elements, not generated graphics
```

**2.5.6 Message Generation Triggers**

```
Messages are generated at these events:
    1. Rumor share: node shares rumor to connection(s)
       → generate message based on shape selection + sender attributes
    2. Correction generation: fact-checker creates correction
       → generate Correction Shape 1 (debunk)
    3. Correction relay: node forwards correction
       → generate Correction Shape 2 (relay) or Shape 3 (told you so)
    4. Correction acceptance: believing node gets corrected
       → generate Correction Shape 4 (reluctant walkback)
    5. Mutation: rumor mutates at a node
       → generate Shape 4 (elaborator) with new fake detail
    6. Platform hop: rumor crosses platforms
       → generate message with screenshot card + platform context
    7. Group broadcast (WhatsApp): share to group
       → generate one message with group context header
    8. Algorithmic amplification: content goes trending
       → no new message, but engagement numbers update on existing cards

Messages are generated ON DEMAND:
    Feed panel: generated as events occur during simulation (curated highlights only)
    Edge tooltip: generated on hover (not pre-computed)
    Node inbox: generated when node is clicked (not pre-computed for all nodes)
    
    This means most messages in a 500-node simulation are never generated —
    only the ones the user actually sees. Saves computation and memory.
```

---

### 2.6 Lightweight Demographic Layer

Adds population heterogeneity through two attributes that modify existing mechanics, without introducing new agent types or standalone systems.

```
Demographic assignment on node creation:
    node.demographic.age_group:
        "young"  (18-30):  40% of nodes
        "middle" (30-50):  35% of nodes
        "older"  (50+):    25% of nodes
    
    node.demographic.digital_nativity:
        young:   Beta(7, 3) → mean 0.7, skewed high
        middle:  Beta(5, 5) → mean 0.5, symmetric
        older:   Beta(3, 7) → mean 0.3, skewed low

How demographics modify EXISTING mechanics (modifiers only, not new systems):

1. Platform distribution bias (modifies §2.4 multi-platform assignment):
    young:  Twitter 0.30, Instagram 0.40, Reddit 0.20, WhatsApp 0.10
    middle: Twitter 0.25, Instagram 0.20, Reddit 0.15, WhatsApp 0.40
    older:  Twitter 0.10, Instagram 0.10, Reddit 0.05, WhatsApp 0.75
    
    → Older users concentrate on WhatsApp (realistic)
    → Young users spread across platforms (realistic)
    
    Applied during multi-platform assignment: platform weights are
    multiplied by demographic weights before normalization

2. Sharing behavior modifier (modifies §4.3 selective sharing):
    sharing_modifier = 1.0 + (0.3 × (1 - digital_nativity))
    → Low digital nativity = MORE likely to share without verifying
    → High digital nativity = baseline sharing rate
    
    Applied to: effective_share_probability *= sharing_modifier

3. Correction receptivity (modifies §4.5 correction fatigue):
    correction_receptivity = 0.5 + (0.5 × digital_nativity)
    → High digital nativity = more receptive to corrections
    → Low digital nativity = more resistant ("I know what I saw")
    
    Applied to: correction_effectiveness *= correction_receptivity

4. Bot detection intuition (modifies §4.16 bot behavior):
    bot_detection_intuition = digital_nativity × 0.3
    → When receiver receives from a bot sender:
       receiver has bot_detection_intuition probability of recognizing it
       If recognized: apply 0.3x trust modifier (§4.6 "Bot (if detected)")
    → High digital nativity users spot bots more often

5. Topic susceptibility correlation (modifies existing literacy vectors):
    older + low_digital_nativity (digital_nativity < 0.4):
        literacy_vector.health *= 0.8  (health literacy penalty)
    young + high_digital_nativity (digital_nativity > 0.6):
        literacy_vector.celebrity *= 1.2  (celebrity/campus literacy bonus)
        literacy_vector.campus *= 1.2
    middle + any:
        literacy_vector.financial *= 0.85 during crisis events (§6.4.5)
    
    These are SMALL modifiers on existing literacy vectors, not new systems
    Applied once during node initialization (except financial crisis modifier)

Implementation:
    2 new fields per node (age_group, digital_nativity)
    5 modifier lookups in existing code paths
    Does NOT add new agent types or new mechanics
    Just makes existing mechanics slightly demographic-aware
```

---

## 3. Platform Models

### 3.1 Four Platform Configurations

| Parameter | WhatsApp | Twitter/X | Instagram | Reddit |
|-----------|----------|-----------|-----------|--------|
| **Topology** | Watts-Strogatz (clustered groups) | Barabási-Albert (scale-free hubs) | Ring-lattice clusters with random bridges | Stochastic Block Model (community blocks) |
| **Arrival Rate (λ)** | Poisson(λ=2/min) | Poisson(λ=15/min) | Poisson(λ=5/min) | Poisson(λ=8/min) |
| **Base Service Time (μ)** | Exponential(μ=10min) | Exponential(μ=30sec) | Exponential(μ=5min) | Exponential(μ=2min) |
| **Base Credibility** | Higher (personal trust) | Lower (public skepticism) | Medium | Varies by community |
| **Special Mechanic** | Forward limit (max 5), group broadcast | Retweet = broadcast, algorithmic amp | Story decay (24hr), reel virality | Upvote threshold, moderator intervention |
| **Correction Speed** | Slow (no central authority) | Fast (community notes) | Medium | Fast (moderators) |

Note: Actual service time per node = Base Service Time × Agent Type Modifier (see §2.1).

### 3.2 Group Dynamics (WhatsApp Specific)
WhatsApp uses **batch processing** — one share to a group hits ALL members simultaneously:

```
When a node shares to a WhatsApp group:
    all members receive the message at the SAME timestep
    group_size = random(5, 50)
    "group admin" nodes exist: higher forward rate, acts as local influencer
    forward_limit = 5 (after 5 forwards, message is tagged "forwarded many times")
    tagged messages get credibility penalty: threshold *= 1.3 (harder to believe)

Forward count tracking:
    forward_count lives on the RUMOR OBJECT, not the node
    Each forward increments rumor.forward_count
    Mutations RESET forward_count to 0 (new version = new message)
    This means: a mutated rumor escapes the forward limit penalty
```

### 3.3 Algorithmic Amplification (Twitter/Instagram)
High-engagement content gets boosted by the platform algorithm:

```
At each timestep, for trending content:
    if shares_in_last_window > engagement_threshold:
        reach_multiplier = 3x to 5x
        message gets pushed to non-followers (explore page / for-you)
        duration of boost: Exponential(μ=30min)
    
    engagement_threshold scales with network size:
        Twitter: (network_size × 0.10) retweets in 5 minutes
        Instagram: (network_size × 0.20) shares in 10 minutes
        → 500 nodes: Twitter=50, Instagram=100
        → 2000 nodes: Twitter=200, Instagram=400
    
    This creates positive feedback loops:
        more shares → algorithm boost → more shares → bigger boost
```

### 3.4 Instagram Mechanics

**3.4.1 Instagram Network Topology (Ring Clusters with Bridges)**
```
Construction algorithm:
    1. Divide N_instagram nodes into K clusters (K = num_echo_chambers)
    2. Within each cluster: create a ring lattice
       → each node connects to its k_nearest = 4 neighbors on the ring
    3. Add intra-cluster random edges:
       → p_intra = 0.15 (rewiring probability, similar to Watts-Strogatz)
    4. Add inter-cluster bridge edges:
       → p_bridge = 0.02 (sparse connections between clusters)
       → bridges preferentially connect to high-degree nodes (influencers)
    
    Parameters:
        k_nearest: 4 (ring lattice neighbor count)
        p_intra_rewire: 0.15
        p_bridge: 0.02
    
    Properties:
        High clustering within follower circles
        Low diameter due to bridge shortcuts
        Influencer nodes act as cluster hubs
        Models Instagram's follower-cluster + explore-page structure
```

**3.4.2 Instagram Story Decay (24-Hour Expiry)**
```
Instagram content has a time-to-live (TTL) based on format:

    story_ttl = 24 simulated hours (86,400 simulated seconds)
    reel_ttl  = None (reels persist indefinitely)
    post_ttl  = None (posts persist indefinitely)
    
Decay mechanics:
    When a rumor is shared as a story on Instagram:
        rumor.expiry_time = current_time + story_ttl
        
    At each timestep, for each active story-format rumor:
        if current_time > rumor.expiry_time:
            rumor status on Instagram = "expired"
            nodes can NO LONGER forward this version
            nodes who already believed are NOT un-believed (damage is done)
            BUT no new infections from this version on this platform
    
    Gradual decay visibility:
        remaining_life = (expiry_time - current_time) / story_ttl
        if remaining_life < 0.25 (last 6 hours):
            reach_modifier = remaining_life / 0.25  # fades from 1.0 to 0.0
            → fewer people see it as it approaches expiry
    
    Interaction with platform hopping:
        A story about to expire gets screenshotted more urgently
        hop_probability *= (1 + (1 - remaining_life) * 0.5)  # urgency boost near expiry

Death classification:
    If ALL active versions on Instagram have expired AND no hops occurred:
        death_type = "Time Decayed"
```

### 3.5 Reddit Mechanics

**3.5.1 Reddit Network Topology (Stochastic Block Model)**
```
Construction algorithm:
    1. Define C communities (C = num_echo_chambers, default 4-6)
    2. Use Stochastic Block Model (SBM):
       → p_within = 0.12  (connection probability within same community)
       → p_between = 0.005 (connection probability across communities)
    3. Each community has 1-3 moderator nodes (fact-checker type)
    4. Within each community, degree distribution follows power law:
       → some "power users" have high karma/connections
       → most users are low-engagement lurkers

    Parameters:
        p_within: 0.12
        p_between: 0.005
        mod_count_per_community: Uniform(1, 3)
    
    Properties:
        Strong community boundaries (subreddits)
        Very sparse inter-community links (cross-posting is rare)
        Moderators act as gatekeepers within each community
        Models Reddit's subreddit isolation structure
```

**3.5.2 Reddit Upvote Threshold & Moderator Intervention**
```
Upvote/visibility system:
    Each rumor on Reddit has a karma_score starting at 1
    When a node "believes" and shares: karma_score += 1
    When a node rejects or downvotes: karma_score -= 1
    
    Visibility tiers based on karma:
        karma < 5:    visible only to direct connections (new/controversial)
        karma 5-20:   visible to entire community (rising)
        karma > 20:   visible to adjacent communities + algorithmic boost (hot/trending)
        karma > 50:   cross-community visibility, equivalent to algorithmic amplification
    
    Negative karma:
        karma < 0:    post is collapsed/hidden
        reach_modifier = 0.1 (only 10% of normal visibility)
        This is Reddit's built-in immune system

Moderator intervention:
    Moderator nodes (type = fact_checker within Reddit communities) have special powers:
    
    detection_probability: starts at 0.0, increases with karma:
        p_detect = min(1.0, karma_score / 100)
        → viral content is MORE likely to be caught
    
    detection_delay: Exponential(μ=30min) after detection probability triggers
    
    When moderator detects rumor:
        Option A (60%): Remove post → karma set to -999, reach_modifier = 0
        Option B (30%): Pin correction → all community members see correction with 2x effectiveness  
        Option C (10%): Quarantine thread → no new shares, existing believers unaffected
    
    Moderator fatigue:
        After moderating 5+ posts in a short window:
        detection_delay increases by 2x (overwhelmed)
```

### 3.6 Platform Hopping
Rumors jump between platforms via "screenshot culture":

```
Enhanced hop mechanics:
    Base hop formula:
        hop_probability = base_hop_rate × virality_score × emotional_charge × node.hop_tendency
    
    base_hop_rate: Poisson(λ=0.1/min)
    node.hop_tendency: per-node attribute from Beta(2, 5) distribution
        → most people: 0.1-0.3 (rarely cross-post)
        → chronic cross-posters: 0.7-0.9 (screenshot everything)
    
    When a hop occurs:
        A node that exists on 2+ platforms screenshots the rumor
        Rumor arrives on new platform as a NEW seed
        Content quality may change: random perturbation ±0.1

    Topic-weighted destination selection:
        Target platform chosen probabilistically, NOT randomly:
        
        hop_weight[platform] = platform_fit[rumor.media_type][platform] × topic_relevance[platform]
        
        Topic Relevance Matrix:
        |           | Financial | Health | Celebrity | Campus | Political | Tech |
        |-----------|-----------|--------|-----------|--------|-----------|------|
        | WhatsApp  | 0.9       | 0.8    | 0.5       | 0.7    | 0.4       | 0.3  |
        | Twitter   | 0.7       | 0.5    | 0.9       | 0.4    | 0.9       | 0.8  |
        | Instagram | 0.2       | 0.4    | 0.8       | 0.6    | 0.2       | 0.3  |
        | Reddit    | 0.6       | 0.7    | 0.4       | 0.5    | 0.8       | 0.9  |
        
        → Financial panic heavily favors hopping to WhatsApp (family groups)
        → Celebrity death favors Twitter and Instagram
        → Tech rumors favor Reddit and Twitter

    Correction follow probability:
        CRITICAL: Corrections do NOT automatically follow the hop
        BUT there is a small independent chance:
        
        correction_follow_probability = 0.15 (15-20%)
        
        Condition: at least one multi-platform node on the TARGET platform
                   must have already seen the correction on the SOURCE platform
        
        If condition met AND random() < correction_follow_probability:
            Correction arrives on target platform after delay: Exponential(μ=1h)
        
        If condition NOT met: correction cannot follow (no bridge node has seen it)
        
        This creates a realistic spectrum:
            Sometimes the rumor arrives alone and rages unchecked
            Sometimes the correction follows quickly
            Sometimes there's a 2-hour gap (rumor head start)

    Tracked separately:
        hop_chain: list of (source_platform, target_platform, time, node)
        cross_platform_infection: % of total spread caused by hops
        correction_follow_gap: time between rumor hop and correction arrival (if any)
```

---

## 4. Behavioral Mechanics

### 4.1 Trust Decay Over Time
Repeated exposure erodes skepticism:

```
effective_threshold = base_threshold × (decay_rate ^ times_exposed)

decay_rate = 0.85 (configurable)

Example:
    Base threshold: 0.7 (skeptical)
    1st exposure: 0.700 → rejects
    2nd exposure: 0.595 → rejects
    3rd exposure: 0.506 → rejects
    4th exposure: 0.430 → SHARES

Interaction: trust decay is ACCELERATED by:
    - source credibility (trusted sender = faster decay)
    - emotional charge (fear-based = faster decay)
    - confirmation bias (aligned worldview = faster decay)
```

### 4.2 Mutation
Rumors evolve as they spread:

```
On each forward, with probability p_mutate = 0.05:
    rumor_v(n+1) = mutate(rumor_v(n))
    
    Mutation changes:
        emotional_profile: shift each emotion by random(±0.15), clamped to [0.0, 1.0]
        content_quality: shift by random(±0.1), clamped to [0.1, 1.0]
        topic_alignment: slight drift in worldview alignment vector
        
    Mutation tracking:
        version_number: increments on each mutation
        mutation_chain: list of all versions with parent pointers
        mutation_distance: euclidean distance from original emotional profile
        forward_count: RESETS to 0 (new version = new message, escapes WhatsApp limit)
        
    Effects:
        A health rumor might mutate into a political rumor
        Fear might increase with mutations (sensationalism → survival bias)
        Content quality can improve OR degrade
        Mutated versions may penetrate echo chambers the original couldn't
```

### 4.3 Selective Sharing (Strong vs Weak Ties)
People don't share to everyone equally:

```
Each edge has a relationship_strength (Granovetter's model):
    strong_tie: weight > 0.7 (close friends, family) → share everything
    weak_tie: weight 0.3-0.7 (acquaintances) → share only high-virality content
    distant_tie: weight < 0.3 (followers, strangers) → share only viral content

Sharing decision per edge:
    if virality_score > (1 - edge_weight):
        share to this connection
    else:
        skip this connection

Key insight: weak ties actually spread rumors FURTHER across the network
because they bridge different echo chambers. Strong ties keep rumors
circulating within the same bubble.
```

### 4.4 Read But Don't Act (Silent Believers)
Three-state belief model:

```
When a node "believes" the rumor:
    60% probability → SILENT BELIEVER
        status = "silent_believer"
        believes the rumor internally
        does NOT share it forward
        BUT will confirm if someone asks
        counted in final infection stats
        R₀ calculation must account for these
    
    40% probability → ACTIVE SHARER
        status = "believing"  
        shares to connections based on selective sharing rules
        contributes to forward spread

Silent believers matter because:
    they represent real-world "dark spread"
    they won't show up in share counts but ARE infected
    they can be activated later (if directly asked, or on a different platform)
```

### 4.5 Correction Fatigue
Diminishing returns on corrections:

```
correction_effectiveness = base_effectiveness × (fatigue_rate ^ times_correction_seen)

fatigue_rate = 0.7

Example:
    1st correction seen: 80% chance of being corrected → works
    2nd correction: 56% → maybe works
    3rd correction: 39% → probably ignored
    5th correction: 19% → "I'm tired of seeing these fact-checks"

Interaction with confirmation bias:
    If correction CONTRADICTS worldview:
        fatigue_rate drops to 0.5 (fatigues faster)
    If correction ALIGNS with worldview:
        fatigue_rate stays at 0.7 (fatigues slower)
    
    This models the real "backfire effect" where corrections
    can actually REINFORCE belief in some populations
```

### 4.6 Source Credibility Inheritance (with Backfire Cascade)
WHO sends the message matters as much as WHAT it says:

```
effective_share_probability = base_probability × sender_trust_modifier

sender_trust_modifier per sender type:
    Influencer who shares: 2.0x (if they believe it, must be true)
    Close friend (strong tie): 1.5x
    Regular acquaintance: 1.0x
    Stranger/distant: 0.6x
    Bot (if detected): 0.3x
    Known fact-checker who shares rumor: 3.0x (ultimate credibility)

The SOURCE CREDIBILITY backfire cascade:
    When a HIGH-CREDIBILITY SENDER (fact-checker, respected influencer)
    finally cracks and shares the rumor:
    
    backfire_multiplier = min(0.7, node.credibility_threshold * 0.5)
    # Capped at 0.7 to prevent threshold going negative
    All receivers get: threshold *= (1 - backfire_multiplier)
    
    This models: "If even the DOCTOR believes it, it must be true"
    A single high-credibility conversion can trigger a cascade

    DISTINCT from literacy-based backfire (§6.4.4) — this triggers on
    SENDER TYPE, not literacy level. Both can stack:
    
    total_backfire = min(0.85, source_credibility_backfire + literacy_backfire)
    # Capped to prevent threshold from going to zero
```

### 4.7 Confirmation Bias
Worldview alignment reduces resistance (4D worldview space):

```
Each rumor has: rumor.alignment_vector = [float, float, float, float]
Each node has: node.worldview_vector = [float, float, float, float]
# Dimensions: [political, health_trust, tech_trust, authority_trust]

alignment_score = 1 - (euclidean_distance_4d(node.worldview, rumor.alignment) / max_distance_4d)
# max_distance_4d = sqrt(4) × 2 = 4.0 (diagonal of [-1,1]⁴ hypercube)
# normalized to 0.0 (opposite worldview) to 1.0 (perfectly aligned)

bias_modifier = alignment_score × bias_strength  # bias_strength = 0.4

effective_threshold *= (1 - bias_modifier)
# Result clamped to [0.001, 0.999] for numerical stability

Example:
    Node worldview: [0.8, 0.3, 0.6, 0.4]  (right-leaning, health-skeptic, tech-friendly, moderate trust)
    Rumor alignment: [0.7, 0.4, 0.5, 0.3]  (slightly right, moderate health, moderate tech, low trust)
    euclidean_distance = sqrt((0.1)² + (0.1)² + (0.1)² + (0.1)²) = 0.2
    alignment_score = 1 - (0.2 / 4.0) = 0.95 (very aligned)
    bias_modifier = 0.95 × 0.4 = 0.38
    threshold drops from 0.7 to 0.434 → much easier to convince

4D worldview advantages over 2D:
    → Someone can be health-skeptic BUT trust institutions (overlapping communities)
    → Political alignment doesn't force health alignment (independent dimensions)
    → Echo chambers form as natural clusters in 4D space:
       people cluster on dimensions that matter to them
       but vary independently on dimensions that don't
    → A mutation drifting in the health_trust dimension can unlock
       a health-skeptic community without changing political alignment
    
Interaction with mutation:
    As rumors mutate, their alignment_vector drifts in 4D space
    A rumor can mutate INTO alignment with a new echo chamber
    A political rumor can shift to trigger health-skeptics or tech-paranoid clusters
    This is how misinformation crosses ideological boundaries
```

### 4.8 Emotional Charge System
Multi-dimensional emotional resonance:

```
Each rumor carries:
    rumor.emotions = {fear, outrage, humor, curiosity, urgency}  # each 0.0-1.0

Each node carries:
    node.susceptibility = {fear, outrage, humor, curiosity, urgency}  # each 0.0-1.0

emotional_impact = dot_product(rumor.emotions, node.susceptibility)
                 = Σ(rumor.emotion_i × node.susceptibility_i)

This modifies sharing probability:
    effective_share_prob = base_prob × (1 + emotional_impact × emotion_weight)
    emotion_weight = 0.3

On mutation:
    Each emotion shifts by random(±0.15)
    Clamped to [0.0, 1.0]
    Rumors tend to evolve toward higher emotional charge
    (mutations with higher charge spread more → survival bias)
```

### 4.9 Time of Day Effect
Per-node activity profiles:

```
Base activity curve (probability of being online):
    hours[0-5]   = 0.05 - 0.10  (very low, insomniacs/timezones)
    hours[6-8]   = 0.30 - 0.60  (morning scroll)
    hours[9-11]  = 0.40 - 0.50  (work breaks)
    hours[12-13] = 0.65 - 0.75  (lunch peak)
    hours[14-16] = 0.35 - 0.45  (afternoon)
    hours[17-21] = 0.70 - 0.90  (evening peak)
    hours[22-23] = 0.30 - 0.50  (declining)

Each node gets a personal shift:
    node.time_shift = random(-3, +3) hours  # night owls vs early risers
    node.activity_profile = roll(base_curve, node.time_shift)

At each timestep:
    node.current_λ = node.base_λ × node.activity_profile[current_hour]

Effect on simulation:
    Rumors seeded at 10pm spread fast initially → STALL at 2am
    Morning brings a 2nd wave when people wake up and check phones
    Creates realistic wave patterns with plateaus during dead hours
```

### 4.10 Super Spreader Events
Emergent, not random — triggered by other mechanics interacting:

```
Trigger conditions (any of these):
    1. Engagement threshold crossed:
       shares > (network_size × 0.10) within 5 minutes → flagged as trending
       # Scales proportionally so SSE triggers at similar infection %
       
    2. Influencer amplification:
       An influencer shares → automatic 3x reach (algorithmic amplification)
       
    3. Emotional mutation spike:
       A mutation causes emotional_charge to exceed 0.85 → organic viral burst
       
    4. Cross-platform hop to bigger platform:
       WhatsApp → Twitter = audience explosion

When triggered:
    arrival_rate_boost = 5x to 10x for affected nodes
    boost_duration = Exponential(μ=30min)
    
Logging:
    event_type: "engagement" | "influencer" | "emotion_spike" | "platform_hop"
    time: float
    trigger_node: node_id
    affected_nodes: count
    downstream_impact: additional infections caused by this event
```

### 4.11 Correction Launch & Propagation
Corrections are NOT automatic — they emerge from fact-checker nodes:

```
Correction trigger:
    A fact-checker node receives the rumor through normal spread mechanics
    Fact-checkers have credibility_threshold = 0.95, so they almost never believe
    Instead of believing, they GENERATE a correction:
    
    correction_generation_delay = Exponential(μ=15min)  # research/verification time
    → fact-checkers are slow but reliable

Correction propagation:
    Once generated, the correction spreads as a SEPARATE message:
        correction.type = "correction"
        correction.credibility = 0.8 (base, modified by source trust)
        correction.reach = same as the fact-checker's normal connections
        correction.platform = same platform where fact-checker saw the rumor
    
    Correction does NOT automatically hop platforms:
        Each platform must have its own fact-checker encounter the rumor
        CRITICAL: this is why cross-platform spread is so dangerous
        A rumor can hop to a platform with no fact-checkers present
        (Exception: correction_follow_probability from §3.6)

Correction mechanics at the receiver:
    When a node receives a correction:
        if node.status == "believing" or node.status == "silent_believer":
            apply correction_effectiveness (see §4.5 Correction Fatigue)
            if correction succeeds: node.status = "corrected"
            if correction fails: node.times_correction_seen += 1
        if node.status == "unaware":
            apply pre-bunking effect (see platform-attached corrections above)
        if node.status == "corrected":
            ignore (already corrected)

Platform-specific correction speed modifiers:
    Twitter:   correction_speed_modifier = 1.5x (community notes, quote tweets)
    Reddit:    correction_speed_modifier = 1.3x (moderator pins, top comments)
    Instagram: correction_speed_modifier = 0.8x (no native correction mechanism)
    WhatsApp:  correction_speed_modifier = 0.5x (no central authority, peer-to-peer only)

Platform-attached correction mechanics:
    Some platforms allow corrections to be attached DIRECTLY to rumor content,
    bypassing the need for person-to-person correction propagation:

    Twitter Community Notes:
        When a correction exists on Twitter AND rumor.karma_score > 30 (high visibility):
            community_note_probability = 0.6
            community_note_delay = Exponential(μ=2h)
            
            When community note attaches:
                ALL future viewers of this rumor version on Twitter see correction simultaneously
                correction_effectiveness for attached note = 0.5
                    (weaker than direct fact-check — people often ignore the small label)
                BUT it's passive: doesn't require viewer to seek out the correction
                
                Implementation: every new node that receives this rumor version on Twitter
                has a 50% chance of being auto-corrected on receipt
    
    Reddit Sticky (formalized from §3.5.2):
        Moderator pin → all community members see correction with 2x effectiveness
        This IS platform-attached: correction travels WITH the content
    
    Instagram:
        No platform-attached correction mechanism
        Corrections can only spread person-to-person
        This is WHY Instagram is the most vulnerable platform for misinformation
        → should show up clearly in Monte Carlo platform comparison
    
    WhatsApp:
        No platform-attached correction mechanism (peer-to-peer only)
        BUT: WhatsApp "search the web" prompts on forwarded messages:
            When forwarded_tag = True:
                5% chance node independently searches and self-corrects
                self_correction_delay = Exponential(μ=30min)
                Modeled as: auto-generated correction with source = "self"

Pre-bunking effect (refined):
    When a correction reaches an UNAWARE node:
        pre_bunking_effectiveness = 0.1 + (0.15 × correction_quality)
        correction_quality depends on source:
            fact-checker source: correction_quality = 0.8
            regular node relay: correction_quality = 0.4
            self-correction (WhatsApp search): correction_quality = 0.3
        node.effective_threshold *= (1 + pre_bunking_effectiveness)
    
    → pre-bunking from a fact-checker is more effective than from a random person
    → replaces the flat 1.2x threshold boost with a source-quality-dependent model

Emergency correction (configurable trigger for experiments):
    correction_injection_time: float (default = None, disabled)
    When triggered: ALL fact-checkers on ALL platforms simultaneously generate corrections
    Used for: "what if a news outlet publishes a debunk at time T?"
```

### 4.12 Attention Budget System
Models finite cognitive bandwidth — nodes can't process unlimited messages:

```
Every node starts with:
    node.attention_budget = 1.0 (full attention)

When a node PROCESSES a message (reads from queue):
    attention_cost = 0.02 per message processed
    node.attention_budget -= attention_cost

When attention_budget < 0.3:
    node starts SKIPPING messages in queue:
    p_skip = 1 - (attention_budget / 0.3)
    → at budget 0.15: skips 50% of queued messages
    → at budget 0.05: skips 83% of queued messages
    → at budget 0.0: skips all messages (completely tuned out)
    
    Models: "I've seen too much about this today, I'm scrolling past"
    
    Skipped messages are marked as "attention_skipped" in logs
    They do NOT count as exposures for trust decay
    They DO remain in queue (can be processed if attention recovers)

Attention recovery:
    recovery_rate = 0.1 per simulated hour
    node.attention_budget = min(1.0, attention_budget + recovery_rate × dt)
    
    → full recovery from empty: ~10 hours (overnight rest cycle)
    → partial recovery during afternoon lull: enough to resume processing

Interaction with time of day:
    During low-activity hours (node.activity_profile < 0.2):
        recovery_rate doubled → 0.2 per hour (people mentally reset when offline)
    During peak hours (node.activity_profile > 0.7):
        recovery_rate halved → 0.05 per hour (constant stimulation prevents rest)

Effect on simulation:
    Early spread: everyone has full attention, spread is fast
    Mid simulation: heavy-traffic nodes (influencers, hub nodes) start ignoring messages
    Late simulation: peripheral nodes still have attention (low exposure = budget preserved)
    Creates natural diminishing returns without needing competing content
    Nodes in viral clusters get saturated faster than peripheral nodes
    
    Attention budget interacts with queue overflow (§5.3):
        A node can have a full queue AND low attention
        → messages pile up AND get skipped = double bottleneck

Implementation: ~20 lines of code in the message processing loop
```

### 4.13 Reactive Network Rewiring
Limited, targeted network changes in response to rumor dynamics:

```
Two rewiring behaviors, both constrained to max 3 rewiring events per node:

UNFOLLOW MECHANIC:
    Trigger: node receives 3+ rumor messages from the SAME source node
    Condition: node.status == "corrected" OR node rejected all messages from that source
    
    p_unfollow = 0.15
    if random() < p_unfollow:
        remove edge (node, source)
        node.rewiring_events += 1
        node.unfollowed.append(source)
        log: "Node #X unfollowed Node #Y (rumor fatigue)" + timestamp
    
    Effect: aggressive sharers lose their audience over time
    Creates natural dampening — the most prolific spreaders become less effective

SEEK MECHANIC:
    Trigger: node transitions to "believing" status
    Condition: emotional_impact > 0.7 (high emotional charge)
    
    p_seek = 0.10
    if random() < p_seek:
        add 1-2 new weak-tie edges to random nodes in SAME echo chamber
        new_edges have relationship_strength = Uniform(0.2, 0.4) (always weak ties)
        node.rewiring_events += 1
        log: "Node #X seeking more info — added 2 connections (rabbit hole)" + timestamp
    
    Effect: believers expand their echo chamber exposure slightly
    Models "falling down the rabbit hole" — seeking confirming sources

Constraints:
    Max rewiring events per node: 3 (prevents runaway topology changes)
    New edges are always weak ties (weight 0.2-0.4)
    Removed edges are logged for network autopsy (§6.8)
    Rewiring events are marked on the timeline (§7.6) as a new event type
    
    Network remains ~95% static — only the most dramatic behavioral responses
    trigger rewiring, and only a small fraction of nodes ever rewire

Edge handling:
    Removed edges: fully deleted from adjacency list, no longer traversable
    Added edges: immediately active for message passing
    Both tracked in: network_autopsy.rewiring_log = [{type, time, node, target}]

Interaction with other mechanics:
    Unfollowing an influencer has outsized impact (removes high-reach edge)
    Seeking during a backfire cascade amplifies the cascade (more echo chamber exposure)
    Unfollow + attention budget together model realistic "tuning out":
        first skip messages (attention), then unfollow if it persists (rewiring)
```

### 4.14 Emotional Priming & Fatigue
Susceptibility changes dynamically based on message exposure:

```
Two opposing effects on node.susceptibility, tracked per emotion dimension:

PRIMING (short-term amplification):
    When a node processes a message with high emotion_i (rumor.emotions[i] > 0.5):
        priming_boost = 0.05
        node.emotional_priming[emotion_i] += priming_boost × rumor.emotions[emotion_i]
        
        Current susceptibility after priming:
            node.susceptibility[i] = node.original_susceptibility[i] 
                                     + node.emotional_priming[i]
                                     - node.emotional_fatigue[i]
        
        Cap: node.susceptibility[i] <= node.original_susceptibility[i] × 1.5
        Floor: node.susceptibility[i] >= node.original_susceptibility[i] × 0.3
    
    Models: "I just saw three scary messages — now I'm more scared/primed"
    
    Priming decay:
        Every simulated hour: node.emotional_priming[i] *= 0.95
        → priming fades if not reinforced by new messages
        → half-life of ~14 hours: significant within a simulation, but not permanent

FATIGUE (long-term dampening):
    Counter: node.messages_processed_per_emotion[i] tracks messages processed
             where rumor.emotions[i] > 0.5
    
    When counter reaches a multiple of 10:
        node.emotional_fatigue[i] += (1 - 0.95) × node.original_susceptibility[i]
        fatigue_increment = 0.05 × original value per 10 high-emotion messages
    
    Fatigue does NOT recover (permanent desensitization)
    Floor: susceptibility cannot drop below original × 0.3
    
    Models: "I've seen so many fear messages I'm numb to it now"

Net effect over time:
    Short burst of fear messages:
        → susceptibility spikes (priming dominates)
        → then gradually fades if no reinforcement
    Sustained fear bombardment:
        → initial spike (priming)
        → plateau as priming and fatigue approach equilibrium
        → long-term decline as fatigue accumulates permanently
    
    Different nodes hit fatigue at different times:
        Central hub nodes in viral clusters → desensitized fastest (high message volume)
        Peripheral nodes → never get fatigued (too few messages to trigger)
        This creates heterogeneous susceptibility across the network

Interaction with emotional charge system (§4.8):
    emotional_impact still uses dot_product(rumor.emotions, node.susceptibility)
    But now node.susceptibility is dynamic, not static
    → emotional impact changes over the course of the simulation
    → the same rumor message has different impact on the same node at t=1h vs t=12h
```

### 4.15 Message Framing Bonus
Bridges the message generation system (§2.5) and the simulation decision engine:

```
When a message is generated, the selected shape contributes a framing_modifier
to the sharing probability at the receiver's decision point:

Shape framing modifiers:
    Shape 1 (Bare Forward):     framing_modifier = 0.00 (no persuasive framing)
    Shape 2 (Reaction):         framing_modifier = 0.05 (emotional contagion)
    Shape 3 (Personal Frame):   framing_modifier = 0.15 (personal credibility boost)
    Shape 4 (Elaborator):       framing_modifier = 0.10 (detail = believability)
    Shape 5 (Skeptic Sharer):   framing_modifier = 0.20 (skeptic endorsement = most persuasive)
    
    Correction shapes:
    Correction Shape 1 (Debunk):     correction_framing_boost = 0.15 (authoritative)
    Correction Shape 2 (Relay):      correction_framing_boost = 0.05 (casual)
    Correction Shape 3 (Told You So): correction_framing_boost = 0.00 (off-putting)
    Correction Shape 4 (Walkback):    correction_framing_boost = 0.10 (peer credibility)

Applied at the receiver's decision point:
    For rumor messages:
        effective_share_probability *= (1 + framing_modifier)
    For correction messages:
        correction_effectiveness *= (1 + correction_framing_boost)

Why these values:
    Bare forwards are least persuasive (no context, no personal endorsement)
    Personal frames add "my cousin confirmed" social proof
    Skeptic sharers are MOST persuasive: "even the skeptic believes it"
    
    The modifier is small (0-20%) — it doesn't override core mechanics
    but creates a measurable difference in Monte Carlo outcomes
    
    A run where more Personal Frame messages are generated (because of
    strong-tie-heavy network topology) will have slightly higher infection
    than a run dominated by Bare Forwards (bot-heavy network)

Interaction with mutations:
    Mutations change emotional_profile → which changes shape selection (§2.5.2)
    → which changes framing_modifier → which changes sharing probability
    A mutation that increases emotional intensity → triggers more Reaction shapes
    → framing_modifier goes from 0.15 (Personal) to 0.05 (Reaction) → less persuasive
    
    This creates a subtle feedback: highly emotional mutations spread via emotional
    contagion (high emotion_weight) but with less persuasive framing (lower framing modifier)
    More analytical mutations spread with more persuasive framing but less emotional punch

framing_modifier is stored on the message object for logging and tooltip display:
    message.framing_modifier = 0.15
    Tooltip: "Framing bonus: +15% (Personal Frame)"
```

### 4.16 Coordinated Bot Networks & Platform Detection
Replaces simple bot behavior with coordinated networks and platform countermeasures:

```
A) COORDINATED BOT BEHAVIOR:

Bot network formation (during network generation):
    bot_cluster_count = Uniform(1, 3)  # 1-3 independent bot networks
    All bot nodes randomly assigned to a cluster: node.bot_cluster_id = Uniform(0, cluster_count-1)
    
    Intra-cluster connections:
        Bots within the same cluster are densely connected to each other
        p(edge between bots in same cluster) = 0.8 (near-complete graph within cluster)
        This enables rapid internal coordination
    
    Target assignment:
        Each bot cluster is assigned a target echo chamber (random)
        Bots preferentially connect to nodes in their target chamber:
            p(edge to target chamber node) = 0.3 (high for bots)
            p(edge to non-target chamber node) = 0.05 (low)

Coordinated wave mechanic:
    When the first bot in a cluster shares the rumor:
        wave_trigger = True for all bots in the same cluster
        Each bot in the cluster shares within: Exponential(μ=30sec) of trigger
        Creates a burst of simultaneous shares from multiple accounts
        This models real bot network coordination — synchronized amplification
    
    Wave cooldown between coordinated waves:
        cooldown = Exponential(μ=5min)
        During cooldown, bots still share individually (at normal bot rate)
        After cooldown, next trigger event can start a new wave
        → bots pulse in waves, not continuous spam

Bot credibility building:
    bot.apparent_credibility starts at 0.3 (strangers don't trust unknown accounts)
    Growth over time: bot.apparent_credibility += 0.05 per simulated hour
    Capped at 0.6 (never fully trusted, but not instantly dismissed)
    
    This means:
        Bots that encounter the rumor early (t=0-2h) are LESS effective (low trust)
        Bots that encounter it later (t=6h+) are MORE effective (had time to build trust)
        This models real bot networks that farm credibility before campaigns
    
    Applied as: sender_trust_modifier for bots = bot.apparent_credibility (replaces fixed 0.3x)

B) PLATFORM DETECTION:

Detection probability (per bot, checked each time bot shares):
    base_detection_rate = 0.001 per share event (very low baseline)
    
    Activity-based escalation:
        if bot.shares_this_hour > 10:
            detection_rate *= 3.0  (suspicious activity pattern)
        if bot.shares_this_hour > 20:
            detection_rate *= 10.0 (obvious bot behavior)
    
    Platform-specific detection multiplier:
        Twitter:   1.5x (better bot detection algorithms)
        Reddit:    1.3x (moderator + algorithmic detection)
        Instagram: 1.0x (moderate detection)
        WhatsApp:  0.3x (end-to-end encryption, very hard to detect)
    
    Coordinated wave penalty:
        If 3+ bots from the same cluster share within 60 seconds:
            detection_rate *= 2.0 for all bots in that cluster
            (platform detects coordinated behavior pattern)

When detected:
    node.detected = True
    node.status = "removed"
    All edges from this node are severed (removed from all adjacency lists)
    Any messages in transit from this node are cancelled (removed from receiver queues)
    All future messages from this node are blocked
    
    Log: "Bot #X (cluster #Y) detected and removed at t=Zh (activity: W shares/hour, platform: P)"
    
    Detection does NOT remove other bots in the same cluster
    → but it reduces cluster size and disrupts coordination
    → subsequent waves from the same cluster are smaller

Interaction with demographics (§2.6):
    Detection by USERS (not platform):
        node.demographic.bot_detection_intuition = digital_nativity × 0.3
        When receiving from a bot, user has bot_detection_intuition chance of
        applying the "Bot (if detected)" trust modifier (0.3x)
        → Tech-savvy users are harder for bots to influence

Effect on simulation:
    Aggressive bot networks get detected faster → self-limiting
    Stealthy bots (low activity per hour) survive longer but have less impact
    WhatsApp bots are nearly undetectable (realistic — no platform visibility)
    Creates tension: bot effectiveness vs bot survival
    Network autopsy can track: "X% of bot-caused infections occurred before detection"
    Counterfactual: "What if bots had been detected 1 hour earlier?"
```

---

## 5. Network Structure

### 5.1 Echo Chambers (Worldview-Based Clustering)
Network connections form based on worldview similarity in 4D space, NOT credibility:

```
Each node: worldview_vector = [political, health_trust, tech_trust, authority_trust]
    Generated from cluster centers using: np.random.normal(loc=cluster_center_4d, scale=0.3, size=4)
    
Number of echo chambers: 3-6 (configurable)
    Each chamber has a centroid in 4D worldview space
    
    Example cluster centers:
        Bubble 1: [0.7, 0.3, 0.6, 0.4]   (right-leaning, health-skeptic, tech-friendly, moderate trust)
        Bubble 2: [-0.5, 0.8, 0.4, 0.7]   (left-leaning, health-trusting, moderate tech, high authority)
        Bubble 3: [0.1, 0.2, 0.2, 0.1]    (centrist, skeptic of everything)
        Bubble 4: [-0.3, 0.6, 0.8, 0.5]   (center-left, health-moderate, tech-optimist, moderate authority)
        Bubble 5: [0.5, 0.1, 0.1, 0.2]    (right-leaning, skeptic of health+tech+authority)
    
    4D space enables overlapping communities:
        Two people can share the SAME political bubble but differ on health trust
        Someone can be in the health-skeptic cluster AND the tech-optimist cluster
        This creates cross-cutting community structure that 2D couldn't model

Connection probability:
    base_connection_prob = 0.15  # increased from 0.1 to compensate for sparser 4D distances
    within_chamber_boost = 3.0
    
    same_chamber = (assigned_chamber(A) == assigned_chamber(B))
    
    if same_chamber:
        p(edge) = base_prob × exp(-distance_4d(A.worldview, B.worldview)) × within_chamber_boost
    else:
        p(edge) = base_prob × exp(-distance_4d(A.worldview, B.worldview))
        # No boost — only organic worldview proximity drives cross-chamber edges
    
Key properties:
    Dense connections WITHIN bubbles (people talk to like-minded people)
    Sparse bridges BETWEEN bubbles (some weak ties cross boundaries)
    Credibility varies WITHIN each bubble (skeptics and gullible mixed)
    Each bubble has its own mix of agent types
    4D proximity can create unexpected bridges:
        two people in different political bubbles but similar health_trust
        → weak connection that a health rumor can traverse

Visualization (D3):
    Can't display 4D directly, but:
    - Force-directed layout uses 4D distance for spring constants → clusters emerge naturally
    - Color echo chambers by dominant dimension (political = red/blue gradient)
    - Node shape indicates secondary dimension category:
        ● circle = high health_trust, ■ square = low health_trust
    - Tooltip shows full 4D vector on hover
    - Echo chamber boundaries still render as dotted circles around cluster regions

This means:
    A rumor can explode in one bubble while another bubble is untouched
    Bridge nodes are critical — they're the only path between chambers
    A mutation that shifts alignment can unlock a previously immune chamber
    Health rumors can spread through health-skeptic clusters regardless of political alignment
    Tech-paranoid clusters resist tech-related misinformation independently
```

### 5.2 Edge Properties
Every connection (edge) carries:

```
edge = {
    source: node_id,
    target: node_id,
    relationship_strength: float,   # 0.0-1.0 (Granovetter strong/weak tie)
    trust_weight: float,            # how much source trusts target's judgment
    platform: str,                  # which platform this connection exists on
    bidirectional: bool             # WhatsApp groups = True, Twitter follow = not necessarily
}
```

### 5.3 Edge Cases & Error Handling

```
Network partition handling:
    If network generation creates disconnected components:
        → connect them with 1-2 weak-tie bridge edges
        → ensures every node is theoretically reachable
        → log warning: "partition resolved with N bridge edges"
    
    If rumor seed node has 0 connections (shouldn't happen but defensive):
        → re-select seed node from connected nodes

Fact-checker distribution:
    If all fact-checkers land in the same echo chamber (by chance):
        → this is a VALID outcome, not an error
        → it models real-world uneven debunking coverage
        → but log it: "warning: all fact-checkers in bubble #X"
    
    If a platform has 0 fact-checkers (possible with small networks):
        → valid: that platform has no native correction mechanism
        → corrections can only arrive via platform hop from another platform

Queue overflow protection:
    If a node's message queue exceeds 1,000 messages:
        → drop oldest unprocessed messages (FIFO overflow)
        → log: "queue overflow at node #X, dropped N messages"
        → this models real-world attention limits

Numerical stability:
    All thresholds clamped to [0.001, 0.999] → prevents division by zero or certainty
    Emotional values clamped to [0.0, 1.0] after mutation
    Susceptibility values clamped to [original × 0.3, original × 1.5] after priming/fatigue
    Attention budget clamped to [0.0, 1.0]
    Virality score clamped to [0.0, 5.0] → prevents runaway amplification
    Backfire multipliers capped (see §4.6 and §6.4.4)
    
Reactive rewiring edge cases (§4.13):
    If unfollowing creates a disconnected component:
        → still allowed (realistic — people DO isolate themselves)
        → disconnected nodes become unreachable for new messages
        → logged: "Node #X became isolated after unfollowing"
    If rewiring_events reaches 3:
        → no further rewiring allowed for this node
        → additional triggers logged but not executed
    
Bot detection edge cases (§4.16):
    If all bots in a cluster are detected:
        → cluster effectively destroyed
        → log: "Bot cluster #X fully neutralized at t=Yh"
    If a bot is detected while messages are in transit:
        → messages already delivered are NOT recalled
        → only queued/pending messages are cancelled
    
Monte Carlo edge cases:
    If a run produces 0 infections (seed node rejected by all neighbors):
        → valid outcome, counted in distribution
        → contributes to the "Starved" death type
    If a run hits simulation time limit with >90% infection:
        → classified as "Saturated" if R₀ < 1.0, else "Still Alive"
```

---

## 6. Analysis Metrics

### 6.1 R₀ for Information

```
R₀ = (avg_contacts_per_node) × (probability_of_sharing) × (duration_of_believing)

Calculated per platform and per rumor type.

Dynamic R₀: recalculated at each timestep as the simulation evolves
    early R₀ may be high (fresh rumor, unsaturated network)
    late R₀ drops as nodes get corrected or saturated

Output:
    R₀ timeline plot per platform
    R₀ comparison bar chart across platforms
    Threshold analysis: at what parameters does R₀ cross 1.0
```

### 6.2 Tipping Point Detection

```
For each Monte Carlo run:
    Track infection_rate at each timestep
    Compute d(infection_rate)/dt at each step
    Tipping point = timestep where derivative is MAXIMUM
    
    Also compute: "point of no return"
    = the infection % beyond which correction becomes ineffective
    = found by: if infection > X%, final_infection stays above 50% in 95% of runs
    
Output:
    Average tipping point across all Monte Carlo runs
    Distribution of tipping points (histogram)
    "If correction doesn't arrive before [X]% infection, the rumor becomes uncontrollable"
    Tipping point comparison across platforms
```

### 6.3 Network Resilience Score

```
resilience = 1 - (avg_final_infection_rate / max_possible_infection_rate)
Scale: 0 to 100

Computed per platform, per rumor type, per configuration.

Factors that increase resilience:
    Forwarding limits (WhatsApp)
    Community notes / moderators (Twitter, Reddit)
    Story decay (Instagram)
    Higher average credibility threshold
    Better fact-checker placement
    
Output:
    Resilience comparison bar chart across platforms
    Resilience sensitivity to parameter changes
    Resilience ranking under different rumor types
```

### 6.4 Herd Immunity Threshold (Realistic Model)

**6.4.1 Topic-Specific Literacy**
```
node.literacy_vector = {health, financial, political, celebrity, campus, tech}
Immunity is NOT binary — same person can be immune to health rumors
but susceptible to financial panic.

Effective resistance per rumor = node.literacy_vector[rumor.topic]
```

**6.4.2 Placement Strategy Comparison**
```
Test 4 strategies at same literacy percentage:
    1. Random placement → scatter media-literate nodes randomly
    2. Bridge targeting → place at high-betweenness-centrality nodes
    3. Influencer targeting → make highest-connection nodes literate
    4. Echo chamber seeding → at least one literate node per bubble

Monte Carlo for each: same %, different strategy → compare final infection

Output: strategy effectiveness comparison table and graph
Finding: WHERE you place immunity matters more than HOW MUCH
```

**6.4.3 Literacy Degradation Under Pressure**
```
effective_literacy = base_literacy × (decay_rate ^ (exposure_count × pressure_modifier))

pressure_modifier depends on:
    source credibility of sender (trusted = more pressure)
    emotional charge (fear = more pressure)
    worldview alignment (confirmed bias = more pressure)

Even highly literate people crack under sustained multi-source exposure
```

**6.4.4 Backfire Effect (Literacy-Based)**
```
Distinct from source credibility backfire (§4.6) — this triggers on LITERACY, not sender type.

When a HIGH-LITERACY node (base_literacy > 0.7) finally shares:
    backfire_multiplier = min(0.7, node.base_literacy × 0.5)
    All receivers: threshold *= (1 - backfire_multiplier)

"The skeptic falling becomes the most dangerous super spreader"

Stacking with source credibility backfire (§4.6):
    If the same share triggers BOTH bacfkire types:
    total_backfire = min(0.85, source_credibility_backfire + literacy_backfire)
    Capped to prevent threshold going to zero
```

**6.4.5 Dynamic Threshold During Crisis**
```
crisis_modifier = 1.0 + (crisis_intensity × topic_relevance)

Crisis event specification:
    crisis_time: Uniform(6h, 36h) into simulation  # random trigger point
    crisis_intensity: Uniform(0.3, 0.8)             # how severe the crisis is
    crisis_duration: Exponential(μ=4h)               # how long heightened state lasts
    crisis_topic: str                                # which topic is affected
    
    topic_relevance:
        If crisis_topic matches rumor.scenario: relevance = 1.0
        If adjacent topic (health/financial during pandemic): relevance = 0.5
        If unrelated topic: relevance = 0.1

Effect on simulation:
    During crisis window [crisis_time, crisis_time + crisis_duration]:
        All nodes: effective_threshold *= (1 - crisis_modifier × 0.3)
        Emotional susceptibility for fear and urgency: boosted by +0.2
        Fact-checker correction delay: increased by 2x (overwhelmed)
    
    After crisis window:
        Thresholds recover to pre-crisis levels over Exponential(μ=2h) cooldown
        But trust decay accumulated DURING crisis is permanent

Configurable: crisis can be disabled (crisis_enabled = False) for baseline runs

Output: threshold over time with crisis events marked
```

**6.4.6 Cross-Topic Analysis**
```
Run Monte Carlo grid:
    For each topic in [health, financial, political, celebrity, campus, tech]:
        For each platform in [WhatsApp, Twitter, Instagram, Reddit]:
            For each literacy % in [5%, 10%, 15%, ..., 95%]:
                For each strategy in [random, bridge, influencer, echo_seed]:
                    Run 1000 simulations → record avg infection

Output: 4D heatmap (one per platform, x=literacy%, y=topic, color=infection)
```

### 6.5 Confidence Intervals

```
After N Monte Carlo runs:
    mean_infection = sum(results) / N
    std_error = std(results) / sqrt(N)
    CI_95 = mean ± 1.96 × std_error

Show convergence:
    After 100 runs:   43.2% ± 4.1%
    After 1000 runs:  42.8% ± 1.3%
    After 10000 runs: 42.9% ± 0.4%

Output: convergence plot showing CI narrowing as N increases
```

### 6.6 Distribution Fitting Justification

| Distribution | Used For | Justification |
|-------------|----------|---------------|
| Poisson | Message arrivals per unit time | Independent events at constant avg rate — classic Poisson process |
| Exponential | Processing/reading/service time, boost duration, detection delay, crisis duration, adaptive termination cooldowns | Memoryless: time to next action independent of elapsed time |
| Uniform | Credibility thresholds, initial assignments, crisis timing/intensity, bot cluster assignment, age group distribution | No prior knowledge — maximum entropy assumption |
| Bernoulli | Share/reject binary decision, rewiring trigger, bot detection check | Single trial with known probability |
| Geometric | Trust decay model | Repeated independent exposures until success |
| Normal | 4D worldview vector generation (per cluster) | Central limit theorem — beliefs cluster around community mean |
| Barabási-Albert | Twitter network topology | Real social networks follow power-law degree distributions |
| Watts-Strogatz | WhatsApp network topology | High clustering + short path lengths = small-world model |
| Beta | Emotional susceptibility, hop tendency, digital nativity (age-parameterized) | Flexible shape for bounded [0,1] parameters with skew |
| Stochastic Block Model | Reddit community structure | Models community-based connectivity with inter/intra probabilities |

### 6.7 Death of a Rumor Tracker

**Classification per run:**
```
Death types:
    1. "Starved" → R₀ dropped below 1, no susceptible nodes reachable
    2. "Corrected" → fact-checkers killed it before saturation
    3. "Saturated" → reached everyone possible, nothing left
    4. "Mutated Away" → mutations made it unrecognizable from original
    5. "Time Decayed" → Instagram story expiry, interest faded
    6. "Still Alive" → simulation ended but rumor still active
```

**Detailed per-run metrics:**
```
peak_infection_rate: float          # highest % believing at any point
peak_time: float                    # when peak occurred
time_to_death: float                # how long until spread rate hit zero
total_mutations: int                # mutation count
final_mutation_distance: float      # euclidean distance from original emotional profile
correction_effectiveness: float     # % of believers who were corrected
platform_of_death: str              # which platform it died on first/last
killer_node: node_id                # fact-checker or correction source that dealt final blow
zombie_nodes: int                   # silent believers who still believe at end
super_spread_events_count: int      # how many SSE occurred
platform_hops_count: int            # cross-platform jumps
bots_detected: int                  # how many bots were removed by platforms
rewiring_events_total: int          # total unfollow + seek events across all nodes
avg_attention_budget_at_end: float  # mean attention budget at termination
termination_reason: str             # "max_time" | "rumor_dead" | "saturated" | "corrected"
actual_duration_hours: float        # how long the simulation actually ran (adaptive)
```

**Aggregate across Monte Carlo:**
```
death_type_distribution: pie chart
avg_lifespan_by_platform: bar chart
avg_lifespan_by_emotion: bar chart
survival_curve: Kaplan-Meier style — % of rumors surviving past time T
```

### 6.8 Network Autopsy

**Critical Path Analysis:**
```
After each run:
    Trace backwards from the most-infected cluster
    Find: Node 0 → Node 7 → Node 23 → Node 156 → ...
    Identify "patient zero to pandemic" path
    Highlight in red on network graph
```

**Bottleneck Identification:**
```
Find bridge nodes — the ONLY path between two echo chambers
    If that node hadn't shared, an entire bubble would be spared
    Label as "bridge super spreaders"
    Metric: betweenness centrality of each infected node
```

**What-If Counterfactual Analysis:**
```
After finding critical path, re-run with modifications:
    "What if we removed the top 3 bridge nodes?"
    "What if the first influencer had rejected?"
    "What if the platform hop hadn't occurred?"
    "What if bots were removed?"

Output: counterfactual comparison table
    Scenario              | Infection Change
    Without bridge nodes  | -61%
    Without platform hop  | -28%
    Without bots          | -15%
    Without influencer    | -34%
    Bots detected 1h earlier | -9%
    No network rewiring   | +7%
    Attention budget off   | +12%
```

**Mutation Forensics:**
```
Track: which mutation was the "deadliest" (caused most downstream infections)
Identify: mutation points on the network graph
Show: mutation chain from v1 → v2 → v3 with emotional profile changes
```

**Autopsy Report:**
```
┌─────────────────────────────────────────────────┐
│ NETWORK AUTOPSY REPORT                          │
│                                                 │
│ Critical path length: 7 nodes                   │
│ Key bridge node: Node #23 (influencer)          │
│ Responsible for: 34% of total spread            │
│                                                 │
│ Counterfactual analysis:                        │
│ Without Node #23: infection drops 61%           │
│ Without platform hop: infection drops 28%       │
│ Without bots: infection drops 15%               │
│ Bots detected 1h earlier: infection drops 9%    │
│ Attention budget disabled: infection rises 12%  │
│                                                 │
│ Bot network analysis:                           │
│ Cluster #1: 4/5 bots detected by t=8h          │
│ Cluster #2: 2/3 bots survived (WhatsApp-only)  │
│ Bot-caused infections before detection: 18%     │
│                                                 │
│ Deadliest mutation: v3 at Node #89              │
│ Mutation increased fear by 0.4                  │
│ Caused 2nd wave of spread                       │
│                                                 │
│ Network adaptation: 12 unfollow events          │
│ 3 seek events (rabbit hole behavior)            │
│ Unfollows reduced late-stage spread by ~7%      │
│                                                 │
│ Adaptive termination: rumor died at t=14.2h     │
│ Reason: R₀ < 0.1 for 30+ min (starved)        │
│                                                 │
│ Verdict: Influencer-driven cascade              │
│ with emotional mutation amplification           │
│ Bot cluster #1 accelerated but was neutralized  │
└─────────────────────────────────────────────────┘
```

### 6.9 Sensitivity Analysis

```
Systematic parameter sweeps to identify which parameters most affect outcomes.

Primary sweep parameters:
    1. correction_delay: [0min, 5min, 15min, 30min, 1h, 2h, 6h, 12h, never]
    2. sharing_probability_modifier: [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    3. bot_percentage: [0%, 3%, 7%, 15%, 25%]
    4. network_density: [sparse(0.5x), normal(1x), dense(2x)] → scales connection counts
    5. forward_limit (WhatsApp): [1, 3, 5, 10, unlimited]
    6. algorithmic_amplification: [off, low(2x), normal(3-5x), aggressive(5-10x)]
    7. echo_chamber_count: [2, 3, 4, 6, 8]
    8. fact_checker_percentage: [0%, 1%, 3%, 5%, 10%]

Secondary sweep parameters (new in v2.1):
    9. attention_budget_enabled: [off, on] → toggle to measure attention budget impact
    10. bot_detection_enabled: [off, on] → toggle to measure detection impact
    11. framing_bonus_enabled: [off, on] → toggle to measure framing impact
    12. rewiring_enabled: [off, on] → toggle to measure network adaptation impact
    13. emotional_dynamics_mode: [static, priming_only, fatigue_only, both]
    14. worldview_dimensions: [2D, 4D] → compare echo chamber penetration patterns

Sweep methodology:
    For each parameter:
        Hold all others at default
        Sweep through values listed above
        Run 1,000 Monte Carlo simulations per value
        Record: mean_infection, CI_95, R₀, tipping_point, death_type_distribution

    For 2D heatmaps (Graph #6):
        Sweep correction_delay × sharing_probability as primary pair
        Run 500 Monte Carlo simulations per cell
        Heatmap color = mean final infection rate

Output:
    Tornado chart: rank parameters by impact on final infection rate
    2D heatmaps: correction_delay × sharing_probability (Graph #6)
    Parameter sensitivity curves: each parameter vs infection rate with CI bands
    
Performance note:
    Full grid sweep at 1,000 runs per cell is expensive
    Use Medium network (2,000 nodes) for sensitivity analysis
    Estimated: ~50,000 total runs for full sweep → parallelize with multiprocessing
```

---

## 7. Website Visualization

### 7.1 Website Two-Mode System

The website operates in **two modes**, toggled via a tab bar at the top of the main panel:

**Mode 1: Visual Mode (default)**
```
Purpose: Animated single-run demonstration of all simulation mechanics

What runs:
    Full simulation engine in JavaScript (Web Worker)
    ALL core mechanics present and faithful to Python version:
        ✓ Trust decay, confirmation bias, emotional charge
        ✓ Selective sharing, silent believers, correction fatigue
        ✓ Mutation, platform hopping, algorithmic amplification
        ✓ Time of day, source credibility, super spreader events
        ✓ Correction launch & propagation
        ✓ Reddit karma/moderation, WhatsApp forward limits, Instagram story decay

What displays:
    Force-directed network graph with animated edge pulses
    Live feed panel (curated highlights, see §7.5)
    Simulation concepts panel with live distribution callouts
    Spread curve (live), R₀ gauge, queue length chart
    Timeline scrubber with key moment markers

Network size: 500 nodes (D3 force-directed performance limit)
Single run per interaction — no statistical aggregation
```

**Mode 2: Monte Carlo Mode**
```
Purpose: Statistical analysis with live-building visualizations
Demonstrates: Monte Carlo estimation, convergence, confidence intervals, CDFs

How it works:
    Network graph panel is REPLACED by a statistics dashboard
    Simulation runs in Web Worker WITHOUT animation (no D3 rendering overhead)
    Each run: 500 nodes, adaptive duration (up to 48h), all mechanics active
    Runs execute back-to-back: ~100-200ms per run

    User clicks "Run Monte Carlo" → batch begins
    Run count selector: [100, 500, 1000, 2500] (default: 500)
    
    Estimated times:
        100 runs:  ~15 seconds
        500 runs:  ~75 seconds
        1000 runs: ~150 seconds (2.5 min)
        2500 runs: ~375 seconds (6.25 min)
    
    Show: "Run 847/2500 — ~4 min remaining"

What displays (all update LIVE as runs complete):
    1. Histogram: final infection rate distribution
       → bars fill in one-by-one as each run completes
    2. Convergence plot: mean infection estimate + 95% CI bands
       → line extends rightward with each run, CI bands visibly narrow
    3. R₀ distribution: histogram of R₀ values across runs
    4. Tipping point distribution: histogram of tipping point times
    5. Death type pie chart: fills in as runs classify
    6. Resilience score: computed and displayed after batch completes
    7. Live counter: "Run 147/500 — Mean: 42.3% ± 2.1% (95% CI)"

Progressive statistics with early-stop suggestion:
    After every 50 runs, compute convergence metric:
        convergence = abs(mean_last_50 - mean_cumulative) / std_cumulative
        
    If convergence < 0.02 for 3 consecutive checks:
        Show: "✓ Results have stabilized. Continue for tighter CIs or stop now?"
        User can stop early with valid results
    
    Most runs stabilize around 300-500 runs
    Users who need tail statistics can let it run to 2500

Tail statistics panel (after batch completes):
    "Extreme outcomes (>2σ from mean):"
    "  3.2% of runs had infection > 80% (possible super-spread cascade)"
    "  1.6% of runs had infection < 10% (early containment success)"
    
    Honesty about statistical limitations:
    If N < 1000: show warning icon next to tail percentiles
    "⚠️ Tail percentile estimates require 1000+ runs for reliability"
    
    This turns the limitation into an educational moment about sample size

Course concept callouts (shown alongside):
    ✦ "Monte Carlo Estimation: averaging N=147 independent runs"
    ✦ "CI = mean ± 1.96 × (std / √N) = 42.3% ± 2.1%"
    ✦ "Convergence: estimate stabilizing as N increases"
    ✦ "Tail analysis: rare events need large N for reliable estimates"

Notable Run Detector:
    After batch completes, flag runs with unusual events:
    "Run #347 had a backfire cascade at t=18h — click to view"
    "Run #89 had zero platform hops despite high virality — click to view"
    "Run #201 had the highest infection rate (94%) — click to view"
    "Run #512 saw all bots detected within 2 hours — click to view"
    Clicking switches to Visual Mode and loads that specific run

Interaction:
    User can STOP batch early → results are valid for completed runs
    User can change scenario/platform and re-run
    Previous batch results persist until overwritten
```

**Shared between modes:**
```
Scenario panel (left sidebar): identical in both modes
    Rumor selector, platform selector, media type, seed persona
    Parameter sliders (decay rate, bot %, etc.)
    
The tab bar: [ Visual Mode ] [ Monte Carlo Mode ]
    Switching modes preserves scenario settings
    Switching to Visual Mode from a Notable Run loads that run
```

### 7.2 Layout Modes

Three layout modes for the network visualization area (Visual Mode only):

```
Toggle in top-right corner: 🔍 Focus | ⚖️ Compare | 📊 Overview

FOCUS MODE (default):
    1 main panel (500 nodes, full D3 force-directed with animations)
    3 thumbnail panels (500 nodes each, color-only updates, simplified rendering)
    
    Main panel: ~10ms/frame (full animations, edge pulses, node labels on hover)
    Thumbnails: ~2ms/frame each (color blobs, no animations, no labels)
    Total: ~16ms/frame → 60fps target met
    
    Thumbnail stats overlay:
        ┌─── WhatsApp ──────────┐
        │  [network thumbnail]   │
        │                        │
        │  🔴 34%  🔵 8%  R₀ 2.1│
        └────────────────────────┘
    
    Alert flash on thumbnail border when important event occurs:
        Red flash: super spread event
        Magenta flash: platform hop arrived
        Blue flash: correction launched
        Green pulse: mutation detected
    
    Click thumbnail → swap with main panel

COMPARE MODE:
    2 equal panels (500 nodes each, both with full animations)
    2 thumbnail panels
    Same rumor, same random seed for node attributes
    Different platform parameters
    Synced timeline — both advance together
    
    Use cases:
        WhatsApp (clustered, slow) vs Twitter (hub-based, explosive)
        Same platform with vs without bots
        Same platform with vs without algorithmic amplification

OVERVIEW MODE:
    4 equal panels (500 nodes each, reduced animation detail)
    All at simplified rendering (200-300 node detail, rest as color dots)
    "Surveillance dashboard" — see everything at a glance
    
    Each panel has its mini stats overlay
    Good for watching platform hops propagate across all platforms
```

### 7.3 Feed Panel

```
Replaces the traditional event log with a curated message feed:

STRUCTURE:
    Left sidebar position (or bottom in Compare Mode)
    Scrolling feed of events and messages
    
CONTENT MIX:
    Major events get FULL message text (from §2.5 message generation):
        Rumor seed, mutations, super spread events, corrections,
        platform hops, backfire cascades, influencer shares
    
    Routine shares get COMPACT log format:
        "Node #47 → Node #89, shared ✅ (v2, fear: 0.71)"
    
    This creates a curated highlight feed:
        ~50-100 real messages per run instead of thousands
        Feels like a social media feed without the scale problem

FEATURES:
    Virtual scrolling: only render ~20-30 visible messages
    Auto-scroll with manual override
    "↓ Jump to latest" button when manually scrolled up
    Speed-aware: at 10x, batch routine events with collapse indicators
    
    Filter bar: All | Rumor | Corrections | Mutations | Hops
    Platform filter: All | WhatsApp | Twitter | Instagram | Reddit
    "Hops only" filter: shows only cross-platform events
    Agent type filter: show only influencer messages, etc.
    
    Collapsible and resizable: drag handle to expand/shrink
    Collapsed state: single line with most recent event + badge count
    
    In Focus Mode: feed sits to the right of main panel (vertical scroll)
    In Compare Mode: feed sits below both panels (horizontal, shorter)
    In Overview Mode: collapsed by default

TIMELINE INTEGRATION:
    Timeline scrub rewinds the feed (all messages stored in memory)
    Can replay feed from any point in simulation
```

### 7.4 Node Inspection
Click any node → popup card with full detail + message inbox:

```
┌────────────────────────────────┐
│ 👤 Node #47                    │
│ Type: Regular User             │
│ Connections: 12 (unfollowed: 1)│
│ Echo Chamber: Bubble #2        │
│ Demographic: middle, DN: 0.52  │
│                                │
│ Credibility: 0.62              │
│ Worldview: [0.7, 0.3, 0.6, 0.4]│
│   political: 0.7 (right-lean)  │
│   health_trust: 0.3 (skeptic)  │
│   tech_trust: 0.6 (moderate)   │
│   authority: 0.4 (moderate)    │
│ Emotional Susceptibility:      │
│   fear: 0.8→0.91 (primed +0.11)│
│   outrage: 0.3 (stable)       │
│                                │
│ Attention Budget: 0.42 ⚠️     │
│ Messages Processed: 29         │
│                                │
│ Times Exposed: 3               │
│ Effective Threshold: 0.38      │
│ Trust Decay Applied: yes       │
│ Confirmation Bias: +0.12       │
│                                │
│ Status: 🔴 Believing (active) │
│ Rumor Version: v3 (mutated)    │
│ Infected by: Node #23          │
│ Infected at: t=12.4s           │
│ Downstream infections: 7       │
│                                │
│ Queue: 2 messages pending      │
│ Processing: Exp(μ=5min)→2.3min │
│ Corrections seen: 1            │
│ Correction fatigue: 70%        │
│                                │
│ ─── MESSAGE INBOX ──────────── │
│                                │
│ [t=8.1s] Node #12 (weak tie)  │
│ "⚠️ Celebrity X found dead"   │
│ ❌ REJECTED (threshold: 0.62) │
│                                │
│ [t=10.3s] Node #31 (stranger) │
│ "bro is this real?? 💀"       │
│ ❌ REJECTED (threshold: 0.53) │
│                                │
│ [t=12.4s] Node #23 (close)    │
│ "my cousin confirmed this is   │
│  real, the family hasn't made  │
│  a statement yet"              │
│ ✅ BELIEVED → shared to 8     │
│                                │
│ Decision math:                 │
│  Base: 0.62                    │
│  Trust decay (3rd): ×0.85³     │
│  Conf. bias: ×(1-0.12)        │
│  Source cred: ×(1/1.5)        │
│  Final threshold: 0.38         │
│  Rumor credibility: 0.72       │
│  Framing bonus: +15% (Personal)│
│  Effective cred: 0.83          │
│  0.83 > 0.38 → SHARED         │
└────────────────────────────────┘

Node inbox features:
    Generated ON DEMAND when node is clicked (not pre-computed)
    Shows every message received, chronologically
    Each message shows: sender, tie strength, message text, outcome
    Decision math breakdown for the decisive message
    
    Optional "What if?" toggle:
        "What if this message came from a stranger instead of a close friend?"
        → recalculates with 0.6x source credibility instead of 1.5x
        → "Result: ❌ Would have rejected"
```

### 7.5 Edge Tooltip
Hover over any pulsing edge → compact floating card:

```
┌─────────────────────────────────────┐
│ Node #23 → Node #47                 │
│ "bro Celebrity X is dead fr fr" 📱  │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ 🔗 Strong tie (0.82)  ⏱ t=12.4s    │
│ Rumor v2 · fear: 0.71 · 🖼️ Image   │
└─────────────────────────────────────┘

During correction propagation (blue styling):
┌─────────────────────────────────────┐
│ Node #112 → Node #47     🔵 CORR.  │
│ "This is confirmed fake, X is fine" │
│ ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ │
│ 🔗 Weak tie (0.45)   ⏱ t=34.1s     │
│ Correction #2 for this node         │
│ Fatigue: 56% effective              │
└─────────────────────────────────────┘

Generated on hover, not pre-computed. Pure hover — no click required.
One message at a time, truncated with ellipsis if long.
```

### 7.6 Timeline Replay
```
Bottom bar scrubber like a video player
Controls: ◀◀ ◀ ▶/⏸ ▶ ▶▶
Speed: 0.5x, 1x, 2x, 5x, 10x
Drag to any point → network reconstructs to that state

Auto-marked key moments on timeline:
    🔴 "Rumor Seeded" (t=0)
    🧬 "Mutation #1" (t=X)
    📈 "Tipping Point" (t=X)
    💥 "Super Spread Event" (t=X)
    🔀 "Platform Hop" (t=X)
    🔵 "Correction Launched" (t=X)
    ⚡ "Backfire Cascade" (t=X)
    🤖 "Bot Detected" (t=X)
    ✂️ "Node Unfollowed Source" (t=X)
    💀 "Rumor Death" (t=X)
```

### 7.7 Simulation Concepts Panel
```
Live-updating panel showing active simulation mechanics with real values:

    ✦ Poisson(λ=15) arrivals → 14 messages this tick
    ✦ Exponential(μ=30s) service time → Node #7 processed in 0.8s
    ✦ Bernoulli(p=0.72) → SHARED ✅
    ✦ Trust decay: 0.7 → 0.43 (4th exposure)
    ✦ Emotional impact: 1.40 (fear × susceptibility)
    ✦ Confirmation bias: -0.12 threshold reduction
    ✦ Queue at Node 14: 7 messages (utilization: 84%)
    ✦ Current R₀: 3.2
    ✦ Edge weight: 0.82 (strong tie)
    ✦ Source credibility: 1.5x (close friend)
    ✦ Framing bonus: +15% (Personal Frame shape)
    ✦ Attention budget: Node #14 at 0.42 (skipping 27% of messages)
    ✦ Emotional priming: fear susceptibility 0.65 → 0.78 (+0.13 primed)
    ✦ Bot detection: Bot #203 detected at 15 shares/hour
    ✦ Adaptive termination: R₀ < 0.1 for 35 min → approaching end condition
    
Values update in real-time as simulation runs.
Clicking any concept highlights the relevant node/edge on the graph.
```

### 7.8 Color Scheme
```
Theme: Dark (cyber/data aesthetic)
Background: #0a0a0f
Accent: neon tones

Node colors:
    Unaware:         #4a4a5a (grey)
    Believing:       #ff4444 (red)
    Silent Believer: #ffaa00 (amber/yellow)
    Corrected:       #44aaff (blue)
    Fact-Checker:    #44ff88 (green)
    Bot:             #aa44ff (purple)
    Immune:          #ffffff (white)
    Removed (bot detected): #333333 (dark grey, shrinks to 50% size with ✕ marker)
    
Edge colors:
    Inactive:        #1a1a2e (dark)
    Rumor traveling:  #ff4444 → animated pulse
    Correction:      #44aaff → animated pulse
    Platform hop:    #ff44ff (magenta flash)
    
Echo chamber boundaries: subtle dotted circles in #2a2a3e
```

### 7.9 Analytics Dashboard (Visual Mode)
```
Shown alongside or below the network graph:

Live charts:
    1. Spread curve: % believing vs time (rumor red, correction blue, silent amber)
    2. R₀ gauge: current dynamic R₀ per platform
    3. Queue length over time
    
Live stats:
    Believing: 23% 🔴  Corrected: 12% 🔵
    Silent: 8% 🟡      Unaware: 57% ⚪
    Avg Queue: 3.2    Utilization: 76%
    Mutations: 4      Platform hops: 1
    R₀: 3.2           Tipping: not yet
```

---

## 8. Simulation & Modeling Concepts Map

| Course Concept | Where It Appears |
|---------------|-----------------|
| Random Number Generation | Every decision: share/reject, processing time, network generation, mutations, rewiring, bot detection |
| Poisson Distribution | Message arrival rates per platform, platform hop timing, super spread triggers |
| Exponential Distribution | Service/processing times, boost duration, fact-checker detection delay, crisis duration, bot wave cooldown, community note delay, adaptive termination cooldowns |
| Uniform Distribution | Credibility thresholds, initial parameter assignments, crisis timing/intensity, bot cluster assignment, age group distribution |
| Normal Distribution | 4D worldview vector generation per echo chamber cluster |
| Bernoulli Trials | Share/reject decision at each node, rewiring trigger, bot detection check, attention skip check |
| Geometric Distribution | Trust decay model (repeated trials until belief) |
| Beta Distribution | Emotional susceptibility assignment, hop tendency, digital nativity (age-parameterized) |
| Stochastic Block Model | Reddit community topology |
| Monte Carlo Estimation | 10,000+ runs to estimate spread probability distributions (Python), up to 2,500 runs in browser with progressive convergence |
| Cumulative Distribution Function | CDF of infection rates, convergence analysis |
| Event-Driven Simulation | Priority queue of events: message arrivals, processing completions, shares, rewiring, bot detection |
| Queuing Theory | Message queues at each node, arrival rate vs service rate, utilization, attention budget as queue processing modifier |
| Server Utilization | % time each node is busy processing messages vs idle |
| Confidence Intervals | 95% CI on all Monte Carlo estimates, narrowing with N, tail statistics with sample size warnings |
| Sensitivity Analysis | Parameter sweeps: correction delay, sharing probability, network density |
| Convergence Analysis | Monte Carlo estimate stabilizing as run count increases, progressive convergence detection |
| Kaplan-Meier Survival | Rumor survival curve: % of rumors alive past time T |
| Dynamic Systems Feedback | Attention budget depletion-recovery cycle, emotional priming-fatigue equilibrium, bot detection escalation |

---

## 9. Output Deliverables

### 9.1 Python Outputs (Graphs)

| # | Graph | Type |
|---|-------|------|
| 1 | Spread curve: % believing vs time (rumor red, correction blue, silent amber) | Line plot |
| 2 | Monte Carlo histogram: final infection rate distribution across 10K runs | Histogram |
| 3 | Platform comparison: same rumor on 4 platforms, overlaid spread curves | Multi-line plot |
| 4 | Convergence plot: Monte Carlo estimate + CI narrowing as N increases | Line plot with bands |
| 5 | Queue length over time: message backlogs during viral moments | Line plot |
| 6 | Sensitivity heatmap: correction delay × sharing probability → infection rate | Heatmap |
| 7 | CDF plot: cumulative probability of reaching X% infection | CDF curve |
| 8 | Utilization rate by platform: how overloaded nodes get | Bar chart |
| 9 | R₀ timeline: dynamic R₀ per platform over simulation time | Multi-line plot |
| 10 | Tipping point distribution across Monte Carlo runs | Histogram |
| 11 | Network resilience comparison across platforms | Bar chart |
| 12 | Herd immunity: literacy % vs infection rate per strategy | Multi-line plot |
| 13 | Herd immunity heatmap: topic × platform × literacy % → infection | Heatmap |
| 14 | Death type distribution: how rumors die | Pie chart |
| 15 | Survival curve: Kaplan-Meier style rumor lifespan | Survival curve |
| 16 | Network autopsy: critical path highlighted on network graph | Network visualization |
| 17 | Counterfactual comparison: what-if scenario impact | Bar chart |
| 18 | Mutation chain: emotional profile evolution across versions | Connected scatter plot |
| 19 | Time of day effect: spread rate vs hour with activity overlay | Dual-axis plot |
| 20 | Echo chamber penetration: infection rate per bubble over time | Stacked area chart |
| 21 | Sensitivity tornado chart: parameter impact ranking | Tornado chart |
| 22 | Attention budget depletion: avg attention budget over time by node centrality | Multi-line plot |
| 23 | Emotional susceptibility drift: mean fear/outrage susceptibility over time (priming + fatigue) | Dual-line plot |
| 24 | Bot survival curve: % of bots remaining over time by platform | Multi-line plot |
| 25 | Rewiring events timeline: cumulative unfollows and seeks over simulation time | Stacked area chart |
| 26 | Demographic breakdown: infection rate by age group × platform | Grouped bar chart |
| 27 | Adaptive termination distribution: histogram of simulation durations across MC runs | Histogram |
| 28 | Framing modifier impact: infection rate with vs without framing bonus (paired MC comparison) | Paired bar chart |

### 9.2 Submission Files
```
├── simulation.py              # Core simulation engine
├── simulation.ipynb           # Jupyter notebook with analysis + graphs  
├── visualization.html         # Interactive web visualization (single file)
├── report.pdf                 # 4-5 page project report
└── README.md                  # Project overview and instructions
```

---

## 10. Tech Stack & Performance Targets

### Python (Core Simulation)
| Library | Purpose |
|---------|---------|
| numpy | Random number generation, all distributions |
| networkx | Graph creation (Watts-Strogatz, Barabási-Albert, SBM), centrality metrics |
| matplotlib | All static graphs and plots |
| seaborn | Heatmaps, styled statistical plots |
| pandas | Monte Carlo results storage and analysis |
| scipy.stats | Distribution fitting, confidence intervals, statistical tests |
| collections.deque | Message queues at each node |
| heapq | Priority queue for event-driven simulation |

### Website (Visualization)
| Technology | Purpose |
|------------|---------|
| HTML5 Canvas / D3.js | Network graph rendering and animation |
| JavaScript (vanilla) | Simulation logic running in browser |
| CSS3 | Dark theme, neon accents, responsive layout |
| Chart.js or D3 | Dashboard graphs (spread curves, histograms, gauges) |
| Web Workers | Run simulation without blocking UI (both Visual and Monte Carlo modes) |
| Single .html file | No build tools, no backend, fully portable |

### Performance Targets

```
Python simulation:
    Single run (2,000 nodes, 48h sim): < 2 seconds
    1,000 Monte Carlo runs (2,000 nodes): < 30 minutes
    Full sensitivity grid (50K runs): parallelize with multiprocessing → < 4 hours

Browser simulation:
    Rendering: ≥ 30fps with 500 nodes (target 60fps)
    Per simulation step: < 16ms (to maintain frame budget)
    Initial page load: < 3 seconds
    Monte Carlo mode: 500 runs in < 120 seconds, 2500 runs in < 10 minutes
    
    Web Worker architecture:
        [Web Worker: simulation engine] → postMessage(snapshot) → [Main Thread: D3 renderer]
        User controls (play/pause/speed) → postMessage(command) → [Web Worker]
        Worker posts state snapshots to main thread at 30fps
        Each snapshot: all node statuses, active edges, queue lengths, metrics
```

---

## 11. Build Order & Timeline

### Phase 1: Core Engine (2.5 days)
```
□ Basic node class with all attributes (§2.2) including:
    □ 4D worldview vector
    □ Attention budget field
    □ Emotional priming/fatigue tracking fields
    □ Demographic attributes (§2.6)
    □ Rewiring counters
    □ Bot-specific fields
□ Rumor object class (§2.3) with 4D alignment vector
□ Single platform network generation (start with Twitter/Barabási-Albert)
□ Event-driven simulation loop with priority queue
□ Poisson arrivals + Exponential service times (resolved per §2.1)
□ Bernoulli share/reject with basic credibility threshold
□ Trust decay mechanic
□ Silent believer state
□ Selective sharing (edge weights)
□ Correction mechanics + correction fatigue
□ Source credibility inheritance
□ Correction launch & propagation (§4.11)
□ Attention budget system (§4.12)
□ Adaptive termination with checkpoint snapshots (§1.5)
□ Test: single run produces valid spread curve with adaptive termination
```

### Phase 2: Full Feature Set (2.5 days)
```
□ All 5 agent types implemented with service time resolution
□ All 4 platform topologies (Watts-Strogatz, BA, Ring-lattice, SBM)
□ 4D worldview echo chamber generation (§5.1) with example cluster centers
□ Instagram story decay (§3.4.2)
□ Reddit karma/upvote system + moderator intervention (§3.5)
□ Confirmation bias using 4D worldview alignment (§4.7)
□ Emotional charge system (multi-dimensional)
□ Emotional priming & fatigue (§4.14)
□ Mutation with version tracking (reset forward count)
□ Algorithmic amplification (scaled engagement thresholds)
□ Group dynamics (WhatsApp batch processing)
□ Platform hopping with enhanced mechanics:
    □ Per-node hop tendency
    □ Topic-weighted destination selection
    □ Correction follow probability
□ Platform-attached corrections (§4.11):
    □ Twitter community notes
    □ WhatsApp self-correction on forwarded messages
    □ Refined pre-bunking with correction_quality
□ Multi-platform node identity (§2.4)
□ Demographic layer (§2.6):
    □ Age group + digital nativity assignment
    □ Platform distribution bias
    □ Sharing modifier, correction receptivity, bot detection intuition
□ Reactive network rewiring (§4.13):
    □ Unfollow mechanic
    □ Seek mechanic
    □ Max 3 events per node, edge logging
□ Coordinated bot networks & detection (§4.16):
    □ Bot cluster formation and wave mechanics
    □ Apparent credibility growth
    □ Platform detection with activity-based escalation
    □ Coordinated wave penalty
□ Message framing bonus (§4.15)
□ Time of day activity profiles
□ Media type modifier (quality × platform fit × emotion)
□ Super spreader event detection
□ Seed persona system (§1.6)
□ Crisis event system (§6.4.5)
□ Edge cases & error handling (§5.3) including rewiring and bot detection edge cases
□ Test: multi-platform simulation runs without errors, all new mechanics verified
```

### Phase 3: Monte Carlo + Analysis (2 days)
```
□ Monte Carlo runner (10,000+ runs) with adaptive termination
□ R₀ calculation (static and dynamic)
□ Tipping point detection
□ Network resilience score
□ Herd immunity (all 6 sub-features)
□ Confidence intervals + convergence
□ Death of a rumor classification
□ Network autopsy + counterfactual analysis (including rewiring impact)
□ Sensitivity analysis (§6.9) — all 8 parameters + new mechanics:
    □ Add attention_budget_enabled as toggle parameter
    □ Add bot_detection_enabled as toggle parameter
    □ Add framing_bonus_enabled as toggle parameter
□ All 28 graphs generated (including 7 new graphs #22-#28)
□ Distribution fitting documentation
□ Reproducibility: seed logging and replay
□ Checkpoint-based cross-run comparison at fixed time points
```

### Phase 4: Website Core (2 days)
```
□ Dark theme layout structure
□ Two-mode tab system (Visual + Monte Carlo)
□ Three layout modes (Focus/Compare/Overview)
□ Force-directed network graph with D3 (500 nodes)
□ Node coloring by status
□ Edge animation (pulse effects)
□ Echo chamber visual clustering
□ Thumbnail panels with stats overlay + alert flashes
□ Scenario panel (rumor + platform + media + persona selectors)
□ Parameter sliders
□ Play/pause/reset controls
□ Web Worker architecture (simulation off main thread)
□ Feed panel (curated highlights, §7.3)
□ Simulation concepts panel with live values (§7.7)
□ Basic analytics dashboard (spread curve, live stats, R₀ gauge)
```

### Phase 5: Website Advanced (2 days)
```
□ Node inspection popup with message inbox (§7.4):
    □ 4D worldview display
    □ Attention budget indicator
    □ Emotional priming/fatigue display
    □ Demographic info
    □ Framing bonus in decision math
□ Edge tooltip on hover (§7.5)
□ Timeline replay with scrubber + key moment markers (§7.6):
    □ Including new markers: bot detection, unfollow events
□ Message generation system (§2.5):
    □ 5 rumor shapes + 4 correction shapes
    □ 3-layer phrase pool
    □ Platform tone modifiers
    □ Media type references + simple media cards
□ Monte Carlo Mode:
    □ Headless batch runner in Web Worker
    □ Extended run count: [100, 500, 1000, 2500]
    □ Histogram building in real-time (live bar-by-bar fill)
    □ Convergence plot with CI
    □ Progressive convergence detection with early-stop suggestion
    □ Tail statistics panel with sample size warnings
    □ R₀ distribution, tipping point histogram, death type pie chart
    □ Notable Run detector with click-to-view (including bot detection events)
□ Feed panel filtering (All/Rumor/Corrections/Mutations/Hops + platform filter)
□ Mutation chain visualization
□ Network autopsy view (critical path highlighting)
□ Bot detection visualization (removed nodes shrink with ✕ marker)
□ Rewiring visualization (unfollowed edges fade out, new edges flash in)
```

### Phase 6: Report (1 day)
```
□ Problem description (0.5 page)
    → What is misinformation spread? Why model it? Real-world motivation.

□ Assumptions + distribution justification (1 page)
    → Table of all distributions used (from §6.6) with justification
    → Key simplifying assumptions (§1.8)
    → Limitations acknowledged

□ Model design + flowchart (1 page)
    → System architecture diagram (platforms → nodes → queues → decisions)
    → Event-driven simulation flow
    → Agent type summary table

□ Results + graphs + interpretation (1.5 pages)
    → Must include at minimum:
        Graph #1 (Spread curve), Graph #2 (Monte Carlo histogram),
        Graph #3 (Platform comparison), Graph #6 (Sensitivity heatmap),
        Graph #9 (R₀ timeline), Graph #14 (Death types),
        Graph #24 (Bot survival) or Graph #26 (Demographic breakdown)
    → Each graph gets 2-3 sentences of interpretation
    → Key finding highlighted: which platform is most vulnerable and why

□ Conclusion + improvements (0.5 page)
    → Summary of key quantitative findings
    → Real-world implications
    → Future work: larger networks, real data calibration, more platforms

□ Screenshots from website embedded
    → At least 2: network visualization mid-spread + analytics dashboard
```

### Total Timeline: ~12 days
### Compressed (hard push): ~9-10 days

---

## Feature Lock ✅

All features listed in this document are LOCKED. No additions without timeline review.

### Locked Features Checklist:
- [x] 5 Agent Types (Regular, Influencer, Fact-Checker, Bot, Lurker)
- [x] Service Time Resolution (agent modifier × platform base)
- [x] 4 Platform Models (WhatsApp, Twitter, Instagram, Reddit)
- [x] Instagram Story Decay (24hr TTL with gradual fade)
- [x] Reddit Karma/Upvote System + Moderator Intervention
- [x] Trust Decay Over Time
- [x] Mutation with Version Tracking (forward count reset)
- [x] Selective Sharing (Strong/Weak Ties)
- [x] Read But Don't Act (Silent Believers)
- [x] Correction Fatigue
- [x] Correction Launch & Propagation (§4.11)
- [x] Platform-Attached Corrections (Community Notes, WhatsApp self-correction, refined pre-bunking)
- [x] Algorithmic Amplification (scaled thresholds)
- [x] Group Dynamics (WhatsApp)
- [x] Source Credibility Inheritance (with backfire cascade)
- [x] Confirmation Bias (4D worldview space)
- [x] Echo Chambers (4D Worldview-Based, §5.1)
- [x] Emotional Charge System (Multi-Dimensional)
- [x] Emotional Priming & Fatigue (§4.14)
- [x] Media Type Modifier (Quality × Platform Fit × Emotion)
- [x] Time of Day Effect (Per-Node Activity Profiles)
- [x] Super Spreader Events (Emergent)
- [x] Platform Hopping (Enhanced: hop tendency, topic-weighted, correction follow)
- [x] Multi-Platform Node Identity (§2.4)
- [x] Rumor Object Schema (§2.3, 4D alignment)
- [x] Seed Persona System (§1.6)
- [x] Custom Rumor Input
- [x] Reproducibility & Random Seeds (§1.7)
- [x] Simplifying Assumptions Documented (§1.8, updated for v2.1 features)
- [x] Attention Budget System (§4.12)
- [x] Reactive Network Rewiring (§4.13, unfollow + seek, max 3 per node)
- [x] Message Framing Bonus (§4.15, bridges narrative and mechanical layers)
- [x] Coordinated Bot Networks (§4.16, clustered waves, credibility building)
- [x] Platform Bot Detection (§4.16, activity-based escalation, platform multipliers)
- [x] Lightweight Demographic Layer (§2.6, age group + digital nativity)
- [x] Adaptive Simulation Duration (§1.5, early termination + checkpoints)
- [x] R₀ for Information
- [x] Tipping Point Detection
- [x] Network Resilience Score
- [x] Herd Immunity Threshold (6 Sub-Features including Crisis Events)
- [x] Confidence Intervals
- [x] Distribution Fitting Justification (with SBM + Beta additions)
- [x] Death of a Rumor Tracker
- [x] Network Autopsy with Counterfactuals
- [x] Sensitivity Analysis (8+ parameters, tornado chart)
- [x] Edge Cases & Error Handling (§5.3, including rewiring + bot detection cases)
- [x] Website Two-Mode System (Visual + Monte Carlo)
- [x] Three Layout Modes (Focus/Compare/Overview)
- [x] Feed Panel (curated highlights with filtering)
- [x] Node Inspection with Message Inbox (4D worldview, attention, demographics, framing)
- [x] Edge Tooltip on Hover
- [x] Timeline Replay with Key Moment Markers (including bot detection, rewiring)
- [x] Simulation Concepts Panel (live distribution callouts + new mechanics)
- [x] Message Generation System (5 shapes, 3-layer phrase pools, platform tones)
- [x] Notable Run Detector (Monte Carlo Mode, including bot detection events)
- [x] Thumbnail Stats Overlay with Alert Flashes
- [x] Extended Browser Monte Carlo (up to 2500 runs, progressive convergence, tail stats)
- [x] Performance Targets Defined
- [x] Interactive Web Visualization
- [x] 28 Output Graphs Specified
