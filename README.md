```
     _     _____ __        __     _    _   _
    / \   |  ___|\ \      / /    / \  | | | |
   / _ \  | |_    \ \ /\ / /    / _ \ | |_| |
  / ___ \ |  _|    \ V  V /    / ___ \|  _  |
 /_/   \_\|_|       \_/\_/    /_/   \_\_| |_|

 Multi-Platform Misinformation Cascade Simulation
```

---

## What Is This?

Afwah (Arabic for "rumors") is a discrete-event Monte Carlo simulation that models how misinformation spreads across **four social media platforms** — each with its own network topology, algorithmic rules, and user behavior.

A rumor gets seeded on a few nodes, then the simulation tracks how it propagates through followers, hops between platforms, mutates along the way, gets fought by fact-checkers, and eventually dies. Run it 1,000 times and you get a statistical picture of which platforms are most vulnerable and which interventions actually work.

---

## Architecture

```
+------------------+     +-------------------+     +------------------+
|    simulation.py |     | visualization.html|     |    report.md     |
|                  |     |                   |     |                  |
|  Core Engine     |     |  Interactive Web  |     |  Project Report  |
|  238KB Python    |     |  Single-File App  |     |  6-Page Analysis |
|                  |     |                   |     |                  |
|  - Event-driven  |     |  - D3.js Network  |     |  - 7 Key Graphs  |
|  - MinHeap Queue |     |  - Chart.js MC    |     |  - Distribution  |
|  - 4 Platforms   |     |  - Real-time Viz  |     |    Justification |
|  - Monte Carlo   |     |  - Retro CRT UI   |     |  - Assumptions   |
+------------------+     +-------------------+     +------------------+
         |                        |                        |
         v                        v                        v
+------------------+     +-------------------+     +------------------+
|   simulation_    |     |    graphs/        |     |   spec_v2.1.md   |
|   executed.ipynb |     |                   |     |                  |
|                  |     |  28 Academic PNGs  |     |  Full Spec       |
|  46 Cells        |     |  28 Visual PNGs   |     |  ~2900 Lines     |
|  All Outputs     |     |  Interactive HTML  |     |  6 Phases        |
|  28 Graphs       |     |                   |     |                  |
+------------------+     +-------------------+     +------------------+
```

---

## Platforms Simulated

```
+------------------+  +------------------+  +------------------+  +------------------+
|     TWITTER      |  |    INSTAGRAM     |  |     WHATSAPP     |  |      REDDIT      |
|                  |  |                  |  |                  |  |                  |
|  Barabasi-Albert |  |  Barabasi-Albert |  |  Watts-Strogatz  |  | Stochastic Block |
|  (Scale-Free)    |  |  (Scale-Free)    |  |  (Small-World)   |  |  Model (SBM)     |
|                  |  |                  |  |                  |  |                  |
|  - Algo Amp      |  |  - Stories       |  |  - Forward Limit |  |  - Karma System  |
|  - Super-Spreader|  |  - 24h Expiry    |  |  - Self-Correct  |  |  - Mod Actions   |
|    Events (SSE)  |  |  - Engagement    |  |  - "Forwarded"   |  |  - Community     |
|  - Community     |  |    Threshold     |  |    Tag Penalty   |  |    Notes         |
|    Notes         |  |  - Hop Urgency   |  |  - Group Chains  |  |  - Vote Gating   |
|                  |  |                  |  |                  |  |                  |
|  Infection: 61%  |  |  Infection: 58%  |  |  Infection: 52%  |  |  Infection: 24%  |
+------------------+  +------------------+  +------------------+  +------------------+
```

---

## Simulation Pipeline

```
 SEED (20 nodes)
      |
      v
 +-----------+    +-----------+    +-----------+    +-----------+
 |  RECEIVE  |--->|  DECIDE   |--->|   SHARE   |--->|    HOP    |
 |           |    |           |    |           |    |           |
 | Node gets |    | Belief    |    | Schedule  |    | Bridge    |
 | message,  |    | prob from |    | shares to |    | nodes     |
 | exposure  |    | trust,    |    | neighbors |    | carry to  |
 | count ++  |    | emotion,  |    | w/ delays |    | other     |
 |           |    | confirm   |    |           |    | platforms |
 |           |    | bias      |    |           |    |           |
 +-----------+    +-----------+    +-----------+    +-----------+
                       |                                  |
                       v                                  v
                 +-----------+                     +-----------+
                 |  CORRECT  |                     |  MUTATE   |
                 |           |                     |           |
                 | FCs push  |                     | After 5+  |
                 | counter-  |                     | forwards, |
                 | narrative |                     | emotion & |
                 | after     |                     | cred      |
                 | delay     |                     | change    |
                 +-----------+                     +-----------+
                                                        |
                                                        v
                                                  +-----------+
                                                  | TERMINATE |
                                                  |           |
                                                  | Converge  |
                                                  | or 48h    |
                                                  | cap       |
                                                  +-----------+
```

---

## Agent Types

```
 REGULAR (75%)      FACT-CHECKER (5%)    BOT (3%)           INFLUENCER (2%)     LURKER (15%)
 +-----------+      +-----------+        +-----------+      +-----------+       +-----------+
 | Standard  |      | Never     |        | Always    |      | High cred |       | Reads but |
 | belief &  |      | believes  |        | believes  |       | more     |       | rarely    |
 | share     |      | Spreads   |        | Shares at |      | connects  |       | shares    |
 | mechanics |      | correct-  |        | 10x speed |      | via pref  |       | (10%)     |
 |           |      | ions      |        | Clustered |      | attach    |       | Max 8     |
 |           |      |           |        |           |      |           |       | connects  |
 +-----------+      +-----------+        +-----------+      +-----------+       +-----------+
```

---

## Key Results (1,000 Monte Carlo Runs)

```
 INFECTION RATES                           HOW RUMORS DIE
 ===============                           ==============

 Twitter   |==================== 60.6%     Starved     |======================== 57%
 Instagram |=================== 58.0%      Corrected   |============= 32%
 WhatsApp  |================ 52.1%         Saturated   |== 6%
 Reddit    |======== 24.0%                 Mutated     |== 5%

 Mean: 58.3% +/- 1.4% (95% CI)            Most die because network runs out
 Median: 59.1%                             of susceptible nodes, not correction


 SENSITIVITY ANALYSIS                      EPIDEMIOLOGICAL
 ====================                      ===============

 No corrections  -> +13.6% infection       Mean R0: 0.93 (peaks > 1.0 early)
 No framing      -> -5.3% infection        Point of no return: ~5% infected
 No rewiring     -> -1.9% infection        50% literacy -> still 53% infected
```

---

## Project Structure

```
Afwah/
|
|-- simulation.py               # Core simulation engine (238KB)
|                                  Event-driven, MinHeap priority queue
|                                  All platform mechanics, agent types
|                                  Monte Carlo runner, metrics, plotting
|
|-- simulation.ipynb             # Jupyter notebook (source)
|-- simulation_executed.ipynb    # Fully executed notebook with all outputs
|                                  46 cells, 28 graphs, zero errors
|
|-- visualization.html           # Interactive web app (single file)
|                                  D3.js force-directed network graph
|                                  Chart.js Monte Carlo dashboard
|                                  Real-time simulation playback
|                                  Feed panel, timeline scrub, compare mode
|
|-- report.md                    # Project report (6 pages)
|                                  Problem, assumptions, model design
|                                  Results with 7 key graphs
|                                  Conclusion and future work
|
|-- spec_v2.1.md                 # Full project specification (~2900 lines)
|                                  All 6 phases, feature specs
|                                  Distribution justifications
|                                  Simplifying assumptions
|
|-- graphs/                      # Generated output graphs
|   |-- run_2026-02-18_114023/
|       |-- academic/            # 28 publication-style PNGs
|       |-- visual/              # 28 stylized PNGs
|       |-- interactive/         # Interactive HTML charts
|
|-- generate_notebook.py         # Script to generate notebook from engine
|-- .gitignore
|-- README.md
```

---

## 28 Output Graphs

| # | Graph | What It Shows |
|---|-------|--------------|
| 01 | Spread Curve | Infection % over time (S-curve) |
| 02 | Infection Histogram | MC distribution of final infection rates |
| 03 | Platform Comparison | Side-by-side infection rates per platform |
| 04 | Convergence | Running mean + 95% CI narrowing over runs |
| 05 | Queue Length | Event queue size over simulation time |
| 06 | Sensitivity Heatmap | Two-parameter interaction effects |
| 07 | CDF | Cumulative distribution of infection rates |
| 08 | Utilization by Platform | Per-platform event processing rates |
| 09 | R0 Timeline | Dynamic reproduction number over time |
| 10 | Tipping Point | Infection threshold for self-sustaining cascade |
| 11 | Platform Resilience | Recovery dynamics per platform |
| 12 | Herd Immunity | Literacy level vs infection rate curve |
| 13 | Herd Immunity Heatmap | 2D literacy + vaccination interaction |
| 14 | Death Types | How rumors terminate (starved/corrected/etc.) |
| 15 | Kaplan-Meier | Rumor survival probability over time |
| 16 | Network Autopsy | Critical path analysis post-simulation |
| 17 | Counterfactual | What-if scenarios (remove hubs, bridges, etc.) |
| 18 | Mutation Chain | Rumor evolution through successive mutations |
| 19 | Time of Day | Activity patterns and spread by hour |
| 20 | Echo Chambers | Belief clustering and polarization |
| 21 | Tornado Chart | Single-parameter sensitivity ranking |
| 22 | Attention Budget | Cognitive bandwidth utilization per node |
| 23 | Emotional Drift | Rumor sentiment shift over time |
| 24 | Bot Survival | Bot detection and removal timeline |
| 25 | Rewiring Events | Network topology changes over time |
| 26 | Demographic Breakdown | Age-stratified infection rates |
| 27 | Termination Time | Distribution of when simulations end |
| 28 | Framing Impact | Effect of message framing on spread |

---

## How to Run

### Website (No Setup Needed)
```
Just open visualization.html in any modern browser.
Everything is self-contained in a single file.
```

### Jupyter Notebook
```bash
pip install numpy matplotlib scipy networkx
jupyter notebook simulation_executed.ipynb
```
The executed notebook already has all outputs — no need to re-run (takes ~1.5 hours for 1,000 MC runs).

### Simulation Engine Directly
```python
from simulation import SimulationEngine, SimulationConfig

config = SimulationConfig(network_size=500, seed=42)
engine = SimulationEngine(config)
results = engine.run()
```

---

## Distributions Used

| Distribution | Purpose |
|-------------|---------|
| Poisson | Message arrival timing |
| Exponential | Processing delays, boost duration |
| Uniform | Thresholds, initial values |
| Bernoulli | Share/reject decisions |
| Geometric | Trust decay |
| Normal | Worldview vectors |
| Barabasi-Albert | Twitter/Instagram topology |
| Watts-Strogatz | WhatsApp topology |
| Beta | Susceptibility, nativity |
| Stochastic Block Model | Reddit communities |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Simulation Engine | Python (NumPy, SciPy, NetworkX, Matplotlib) |
| Network Visualization | D3.js v7 (force-directed graph) |
| Monte Carlo Charts | Chart.js with custom retro plugins |
| Web App | Single-file HTML with inline Web Worker |
| Notebook | Jupyter with Gaussian KDE, SciPy stats |

---

## Features

- **Event-driven simulation** with MinHeap priority queue
- **4 platform topologies**: Barabasi-Albert, Watts-Strogatz, Stochastic Block Model, Ring Lattice
- **5 agent types**: Regular, Fact-Checker, Bot, Influencer, Lurker
- **4D worldview model**: political, health_trust, tech_trust, authority_trust
- **Platform-specific mechanics**: algorithmic amplification, story expiry, forward limits, karma, mods
- **Cross-platform hops** with correction follow probability
- **Rumor mutation** after 5+ forwards (emotional drift, credibility change)
- **Crisis events** that modify spread dynamics mid-simulation
- **Demographic layer** with age-based digital nativity modifiers
- **Attention budget** system limiting cognitive bandwidth per node
- **Network rewiring**: unfollow aggressive sharers, seek behavior after belief
- **Super-Spreader Events** triggered by engagement/influencer/emotion thresholds
- **Community Notes** (Twitter) and **Moderator Actions** (Reddit)
- **WhatsApp self-correction** after forward count threshold
- **Monte Carlo runner** with adaptive convergence detection
- **28 output graphs** in academic, visual, and interactive formats
- **Interactive web visualization** with real-time playback, timeline scrub, and compare mode
- **Retro CRT-styled** Monte Carlo dashboard with light/dark theme support

---

