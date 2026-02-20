"""Generate the complete simulation.ipynb notebook with all 28 graphs."""
import json
import uuid

def cell_id():
    return str(uuid.uuid4())[:8]

def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": [source], "id": cell_id()}

def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.split('\n'), "id": cell_id()}

# Fix: source must be list of lines with \n
def code_lines(source):
    lines = source.split('\n')
    src = [l + '\n' for l in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src, "id": cell_id()}

def md_lines(source):
    lines = source.split('\n')
    src = [l + '\n' for l in lines[:-1]]
    if lines[-1]:
        src.append(lines[-1])
    return {"cell_type": "markdown", "metadata": {}, "source": src, "id": cell_id()}

cells = []

# =====================================================================
# Cell 0: Title
# =====================================================================
cells.append(md_lines("""# Afwah — Multi-Platform Misinformation Cascade Simulation
### Phase 3: Monte Carlo Analysis & Visualization"""))

# =====================================================================
# Cell 1: Setup & Imports
# =====================================================================
cells.append(code_lines(r"""# Section 0: Setup & Imports
import sys, os, time, warnings, shutil
from datetime import datetime
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend when not in Jupyter (prevents Tk/multiprocessing crash)
try:
    get_ipython()
except NameError:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patheffects as pe
import seaborn as sns
import networkx as nx
from collections import Counter, defaultdict

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from simulation import (
    Platform, SimulationConfig, SimulationEngine, SimulationResult,
    MonteCarloResult, DeathType, AgentType, NodeStatus,
    run_single_simulation, run_monte_carlo,
    compute_tipping_point, compute_point_of_no_return,
    compute_network_autopsy, run_counterfactual_analysis,
    run_sensitivity_sweep, run_herd_immunity_sweep,
    PLATFORM_CONFIG, CHECKPOINT_TIMES,
)

# === STYLE DEFINITIONS ===
DARK_STYLE = {
    'figure.facecolor': '#050508', 'axes.facecolor': '#050508',
    'axes.edgecolor': '#1a1a2a', 'axes.labelcolor': '#909098',
    'text.color': '#b0b0b8', 'xtick.color': '#606068', 'ytick.color': '#606068',
    'grid.color': '#111118', 'grid.alpha': 0.25, 'grid.linewidth': 0.4,
    'grid.linestyle': ':', 'lines.linewidth': 1.2,
    'font.family': 'monospace', 'font.size': 9,
    'axes.titlesize': 13, 'axes.titleweight': 'normal', 'axes.labelsize': 10,
    'legend.facecolor': '#08080c', 'legend.edgecolor': '#1a1a2a',
    'legend.fontsize': 8, 'legend.framealpha': 0.6, 'figure.dpi': 150,
}

ACADEMIC_STYLE = {
    'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#222222',
    'text.color': '#222222', 'xtick.color': '#444444', 'ytick.color': '#444444',
    'grid.color': '#dddddd', 'grid.alpha': 0.6, 'grid.linewidth': 0.5,
    'grid.linestyle': '-', 'lines.linewidth': 1.8,
    'font.family': 'serif', 'font.size': 10,
    'axes.titlesize': 13, 'axes.labelsize': 11,
    'legend.facecolor': '#ffffff', 'legend.edgecolor': '#cccccc',
    'legend.fontsize': 9, 'figure.dpi': 150,
}

PLOTLY_TEMPLATE = dict(
    template='plotly_dark',
    paper_bgcolor='#050508', plot_bgcolor='#050508',
    font=dict(family='Courier New, monospace', color='#b0b0b8', size=11),
    title_font=dict(color='#c8c8d0', size=14),
    xaxis=dict(gridcolor='#111118', gridwidth=0.4, griddash='dot',
               zerolinecolor='#1a1a2a', linecolor='#1a1a2a',
               tickfont=dict(color='#606068')),
    yaxis=dict(gridcolor='#111118', gridwidth=0.4, griddash='dot',
               zerolinecolor='#1a1a2a', linecolor='#1a1a2a',
               tickfont=dict(color='#606068')),
    legend=dict(bgcolor='rgba(8,8,12,0.6)', bordercolor='#1a1a2a', borderwidth=0.5),
)

# Color palette
C_PRIMARY  = '#d0d0d8'
C_RUMOR    = '#cc6666'
C_CORRECT  = '#6699cc'
C_SILENT   = '#cc9944'
C_FC       = '#66bb88'
C_BOT      = '#9966bb'
C_UNAWARE  = '#555566'
C_ACCENT   = '#ffffff'
PLATFORM_COLORS = {
    Platform.TWITTER: '#5599cc', Platform.WHATSAPP: '#55aa77',
    Platform.INSTAGRAM: '#cc5577', Platform.REDDIT: '#cc7744',
}
PLATFORM_NAMES = {
    Platform.TWITTER: 'Twitter', Platform.WHATSAPP: 'WhatsApp',
    Platform.INSTAGRAM: 'Instagram', Platform.REDDIT: 'Reddit',
}
ACAD_PLAT_COLORS = {
    Platform.TWITTER: '#2266aa', Platform.WHATSAPP: '#228855',
    Platform.INSTAGRAM: '#aa2255', Platform.REDDIT: '#aa5522',
}

# Output directories
RUN_TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H%M%S')
GRAPHS_DIR = os.path.join('graphs', f'run_{RUN_TIMESTAMP}')
VISUAL_DIR = os.path.join(GRAPHS_DIR, 'visual')
ACADEMIC_DIR = os.path.join(GRAPHS_DIR, 'academic')
INTERACTIVE_DIR = os.path.join(GRAPHS_DIR, 'interactive')
for d in [VISUAL_DIR, ACADEMIC_DIR, INTERACTIVE_DIR]:
    os.makedirs(d, exist_ok=True)

def add_subtle_glow(line, color=None, glow_width=3, glow_alpha=0.25):
    c = color or line.get_color()
    line.set_path_effects([
        pe.withStroke(linewidth=glow_width, foreground=c, alpha=glow_alpha),
        pe.Normal()
    ])

def save_visual(fig, num, name):
    path = os.path.join(VISUAL_DIR, f'graph_{num:02d}_{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f'  [visual] {path}')
    plt.close(fig)

def save_academic(fig, num, name):
    path = os.path.join(ACADEMIC_DIR, f'graph_{num:02d}_{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'  [academic] {path}')
    plt.close(fig)

def save_interactive(fig_plotly, num, name):
    path = os.path.join(INTERACTIVE_DIR, f'graph_{num:02d}_{name}.html')
    fig_plotly.write_html(path, include_plotlyjs='cdn')
    print(f'  [interactive] {path}')

def plot_dual(num, name, plot_fn, plotly_fn=None, figsize=(10, 6)):
    # Helper: render visual + academic + optional plotly for a graph.
    with plt.rc_context(DARK_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        plot_fn(fig, ax, dark=True)
        save_visual(fig, num, name)
    with plt.rc_context(ACADEMIC_STYLE):
        fig, ax = plt.subplots(figsize=figsize)
        plot_fn(fig, ax, dark=False)
        save_academic(fig, num, name)
    if plotly_fn:
        fig_p = plotly_fn()
        fig_p.update_layout(**PLOTLY_TEMPLATE)
        fig_p.show()
        save_interactive(fig_p, num, name)

print(f'Setup complete. Output: {GRAPHS_DIR}/')"""))

# =====================================================================
# Cell 2: Section 1 header
# =====================================================================
cells.append(md_lines("## Section 1: Single-Run Demonstration"))

# =====================================================================
# Cell 3: Run simulations
# =====================================================================
cells.append(code_lines(r"""# Run detailed multi-platform sim + per-platform single-platform sims
print("Running detailed multi-platform simulation (seed=42)...")
t0 = time.perf_counter()
cfg_detail = SimulationConfig(
    scenario="celebrity", seed_platform=Platform.TWITTER,
    active_platforms=[Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT],
    network_size=500, master_seed=42, detailed_tracking=True,
)
engine_detail = SimulationEngine(cfg_detail)
engine_detail.setup()
result_detail = engine_detail.run()
print(f"  Done in {time.perf_counter()-t0:.1f}s | infection={result_detail.final_infection_rate:.1%} "
      f"| R0_peak={result_detail.r0_final:.2f} | death={result_detail.death_type.value}")

platform_results = {}
for plat in [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]:
    cfg = SimulationConfig(
        scenario="celebrity", seed_platform=plat, active_platforms=[plat],
        network_size=500, master_seed=42, detailed_tracking=True,
    )
    eng = SimulationEngine(cfg)
    eng.setup()
    r = eng.run()
    platform_results[plat] = r
    print(f"  {plat.value}: infection={r.final_infection_rate:.1%}, R0_peak={r.r0_final:.2f}")"""))

# =====================================================================
# Cell 4: Graph #1 — Spread Curve
# =====================================================================
cells.append(code_lines(r"""# Graph #1: Spread Curve (60s resolution sub-status data)
tl = result_detail.detailed_timelines
times_h = [t / 3600 for t in tl["time"]]

def plot_g1(fig, ax, dark=True):
    c = (C_RUMOR, C_SILENT, C_CORRECT) if dark else ('#cc3333', '#cc8800', '#3366cc')
    l1, = ax.plot(times_h, [x*100 for x in tl["total_infected_frac"]], color=c[0], label='Infected (total)')
    l2, = ax.plot(times_h, [x*100 for x in tl["believing_frac"]], color=c[0], alpha=0.6, linestyle='--', label='Believing')
    l3, = ax.plot(times_h, [x*100 for x in tl["silent_believer_frac"]], color=c[1], label='Silent Believers')
    l4, = ax.plot(times_h, [x*100 for x in tl["corrected_frac"]], color=c[2], label='Corrected')
    if dark:
        for line in [l1, l2, l3, l4]: add_subtle_glow(line)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Population %')
    ax.set_title('Graph #1: Misinformation Spread Curve'); ax.legend(loc='upper left')
    ax.set_ylim(0, 100); ax.grid(True)

def plotly_g1():
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=times_h, y=[x*100 for x in tl["total_infected_frac"]], mode='lines', name='Infected', line=dict(color=C_RUMOR, width=2)))
    fig_p.add_trace(go.Scatter(x=times_h, y=[x*100 for x in tl["believing_frac"]], mode='lines', name='Believing', line=dict(color=C_RUMOR, width=1, dash='dash')))
    fig_p.add_trace(go.Scatter(x=times_h, y=[x*100 for x in tl["silent_believer_frac"]], mode='lines', name='Silent Believers', line=dict(color=C_SILENT, width=1.5)))
    fig_p.add_trace(go.Scatter(x=times_h, y=[x*100 for x in tl["corrected_frac"]], mode='lines', name='Corrected', line=dict(color=C_CORRECT, width=1.5)))
    fig_p.update_layout(title='Graph #1: Misinformation Spread Curve', xaxis_title='Time (hours)', yaxis_title='Population %', yaxis_range=[0, 100])
    return fig_p

plot_dual(1, 'spread_curve', plot_g1, plotly_g1)"""))

# =====================================================================
# Cell 5: Graph #5 — Queue Length
# =====================================================================
cells.append(code_lines(r"""# Graph #5: Queue Length Over Time
def plot_g5(fig, ax, dark=True):
    c = C_PRIMARY if dark else '#333333'
    l1, = ax.plot(times_h, tl["queue_length_avg"], color=c, label='Avg Queue Length')
    if dark: add_subtle_glow(l1)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Avg Events per Node')
    ax.set_title('Graph #5: Queue Length Over Time'); ax.legend(); ax.grid(True)

plot_dual(5, 'queue_length', plot_g5)"""))

# =====================================================================
# Cell 6: Graph #8 — Utilization (FIXED)
# =====================================================================
cells.append(code_lines(r"""# Graph #8: Platform Utilization Rate (Fix 8: smart formatting for tiny values)
util_data = tl.get("utilization_per_platform", [])
peak_util, mean_util = {}, {}
for plat in [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]:
    vals = [u.get(plat, 0) for u in util_data if plat in u]
    peak_util[plat] = max(vals) if vals else 0
    mean_util[plat] = float(np.mean(vals)) if vals else 0

plats = list(peak_util.keys())
pnames = [PLATFORM_NAMES[p] for p in plats]
pvals = [peak_util[p] for p in plats]
mvals = [mean_util[p] for p in plats]

def fmt_util(v):
    # Smart formatting: show enough decimals so values aren't all '0.00'
    if v >= 0.01:
        return f'{v:.2f}'
    elif v >= 0.001:
        return f'{v:.3f}'
    elif v > 0:
        return f'{v:.1e}'
    return '0'

def plot_g8(fig, ax, dark=True):
    x = np.arange(len(pnames))
    colors = [PLATFORM_COLORS[p] for p in plats] if dark else ['#3366cc','#339966','#cc3366','#cc6633']
    ax.bar(x - 0.15, pvals, 0.3, label='Peak', color=colors, alpha=0.85, edgecolor='none')
    ax.bar(x + 0.15, mvals, 0.3, label='Mean', color=colors, alpha=0.45, edgecolor='none')
    ax.set_xticks(x); ax.set_xticklabels(pnames)
    ax.set_ylabel('Server Utilization (0-1)'); ax.set_title('Graph #8: Platform Utilization Rate')
    ax.legend(); ax.set_ylim(0, max(max(pvals) * 1.3, 0.1)); ax.grid(True, axis='y')
    tc = '#b0b0b8' if dark else '#333333'
    for i, (pv, mv) in enumerate(zip(pvals, mvals)):
        ax.text(i-0.15, pv+0.01, fmt_util(pv), ha='center', fontsize=8, color=tc)
        ax.text(i+0.15, mv+0.01, fmt_util(mv), ha='center', fontsize=8, color=tc)

plot_dual(8, 'utilization_by_platform', plot_g8)"""))

# =====================================================================
# Cell 7: Graph #9 — R0 Timeline
# =====================================================================
cells.append(code_lines(r"""# Graph #9: R0 Timeline Per Platform (Fix 2: handle edge cases gracefully)
def plot_g9(fig, ax, dark=True):
    plotted = False
    for plat in [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]:
        r = platform_results.get(plat)
        if r is None:
            continue
        c = PLATFORM_COLORS[plat] if dark else ACAD_PLAT_COLORS[plat]
        if r.r0_timeline and len(r.r0_timeline) > 1:
            ts = [t/3600 for t, _ in r.r0_timeline]; vs = [v for _, v in r.r0_timeline]
            l, = ax.plot(ts, vs, color=c, label=f'{PLATFORM_NAMES[plat]} (peak={r.r0_final:.2f})')
            if dark: add_subtle_glow(l)
            plotted = True
        else:
            # Show as annotation for platforms with minimal R0 data
            ax.scatter([0], [r.r0_final], color=c, marker='x', s=40, zorder=3,
                       label=f'{PLATFORM_NAMES[plat]} (peak={r.r0_final:.2f})')
    ax.axhline(y=1.0, color='#555566' if dark else '#888888', linestyle='--', linewidth=0.8, label='R0=1 threshold')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Effective R0')
    ax.set_title('Graph #9: Effective Reproduction Number'); ax.legend(fontsize=7); ax.grid(True)

def plotly_g9():
    fig_p = go.Figure()
    for plat in [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]:
        r = platform_results.get(plat)
        if r is None:
            continue
        if r.r0_timeline and len(r.r0_timeline) > 1:
            ts = [t/3600 for t, _ in r.r0_timeline]; vs = [v for _, v in r.r0_timeline]
            fig_p.add_trace(go.Scatter(x=ts, y=vs, mode='lines',
                name=f'{PLATFORM_NAMES[plat]} (peak={r.r0_final:.2f})',
                line=dict(color=PLATFORM_COLORS[plat], width=1.5)))
        else:
            fig_p.add_trace(go.Scatter(x=[0], y=[r.r0_final], mode='markers',
                name=f'{PLATFORM_NAMES[plat]} (peak={r.r0_final:.2f})',
                marker=dict(color=PLATFORM_COLORS[plat], size=8, symbol='x')))
    fig_p.add_hline(y=1.0, line_dash='dash', line_color='#555566', annotation_text='R0=1')
    fig_p.update_layout(title='Graph #9: Effective R0', xaxis_title='Time (hours)', yaxis_title='R0')
    return fig_p

plot_dual(9, 'r0_timeline', plot_g9, plotly_g9)"""))

# =====================================================================
# Cell 8: Graph #19 — Time-of-Day + Health subplot
# =====================================================================
cells.append(code_lines(r"""# Graph #19: Time-of-Day Effect (Fix 7: dual-axis with activity overlay, Celebrity + Health)
# Run health scenario for comparison
cfg_health = SimulationConfig(
    scenario="health", seed_platform=Platform.TWITTER,
    active_platforms=[Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT],
    network_size=500, master_seed=42, detailed_tracking=True,
)
eng_health = SimulationEngine(cfg_health)
eng_health.setup()
result_health = eng_health.run()
print(f"Health scenario: infection={result_health.final_infection_rate:.1%}")

# Activity profile from simulation.py TIME_OF_DAY_ACTIVITY
ACTIVITY_PROFILE = {0:0.1,1:0.05,2:0.03,3:0.02,4:0.03,5:0.1,6:0.25,7:0.45,8:0.65,
    9:0.8,10:0.85,11:0.9,12:0.85,13:0.8,14:0.75,15:0.7,16:0.75,17:0.8,
    18:0.85,19:0.9,20:0.95,21:0.9,22:0.7,23:0.4}

def plot_g19(fig, ax, dark=True):
    fig.set_size_inches(12, 5)
    ax.remove()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    for ax_i, res, title, start_h in [(ax1, result_detail, 'Celebrity (start 10:00)', 10),
                                       (ax2, result_health, 'Health (start 10:00)', 10)]:
        tl_i = res.detailed_timelines
        if tl_i and tl_i.get("infections_this_period"):
            times_i = tl_i["time"]
            infections_i = tl_i["infections_this_period"]
            hour_buckets = defaultdict(int)
            for t, inf in zip(times_i, infections_i):
                hod = int((start_h + t / 3600) % 24)
                hour_buckets[hod] += inf
            hours = list(range(24))
            counts = [hour_buckets.get(h, 0) for h in hours]
            c = C_RUMOR if dark else '#cc3333'
            bars = ax_i.bar(hours, counts, color=c, alpha=0.75, edgecolor='none', label='Infections')
            # Activity overlay on secondary axis
            ax2_i = ax_i.twinx()
            act_vals = [ACTIVITY_PROFILE[h] for h in hours]
            c2 = C_CORRECT if dark else '#3366cc'
            l_act, = ax2_i.plot(hours, act_vals, color=c2, alpha=0.6, linewidth=1.5, linestyle='--', label='Activity')
            ax2_i.set_ylim(0, 1.2)
            ax2_i.set_ylabel('Activity Level', fontsize=8, color=c2)
            ax2_i.tick_params(axis='y', labelcolor=c2, labelsize=7)
        ax_i.set_xlabel('Hour of Day'); ax_i.set_ylabel('Infections')
        ax_i.set_title(title, fontsize=10); ax_i.grid(True, axis='y', alpha=0.3)
        ax_i.set_xticks(range(0, 24, 3))
    fig.suptitle('Graph #19: Time-of-Day Infection Pattern (Dual-Axis)', fontsize=13,
                 color='#b0b0b8' if dark else '#222222')
    fig.tight_layout(rect=[0, 0, 1, 0.93])

plot_dual(19, 'time_of_day', plot_g19, figsize=(12, 5))"""))

# =====================================================================
# Cell 9: Graph #20 — Echo Chamber Penetration
# =====================================================================
cells.append(code_lines(r"""# Graph #20: Echo Chamber Penetration (Fix 9: stacked area chart)
tl = result_detail.detailed_timelines
chamber_data = tl.get("infection_per_chamber", [])
n_chambers = max((max(d.keys()) for d in chamber_data if d), default=3) + 1

def plotly_g20():
    fig_p = go.Figure()
    chamber_colors = ['#cc6666', '#6699cc', '#cc9944', '#66bb88', '#9966bb']
    for cidx in range(n_chambers):
        vals = [d.get(cidx, 0)*100 for d in chamber_data]
        fig_p.add_trace(go.Scatter(x=times_h, y=vals, mode='lines', name=f'Chamber {cidx}',
            stackgroup='one', line=dict(color=chamber_colors[cidx % len(chamber_colors)], width=0.5)))
    fig_p.update_layout(title='Graph #20: Echo Chamber Penetration (Stacked Area)',
        xaxis_title='Time (hours)', yaxis_title='Infection %', yaxis_range=[0, 100])
    return fig_p

def plot_g20(fig, ax, dark=True):
    chamber_colors = [C_RUMOR, C_CORRECT, C_SILENT, C_FC, C_BOT] if dark else ['#cc3333','#3366cc','#cc8800','#339966','#663399']
    # Stacked area using fill_between
    bottoms = np.zeros(len(chamber_data))
    for cidx in range(n_chambers):
        vals = np.array([d.get(cidx, 0)*100 for d in chamber_data])
        c = chamber_colors[cidx % len(chamber_colors)]
        ax.fill_between(times_h, bottoms, bottoms + vals, alpha=0.6, color=c, label=f'Chamber {cidx}', linewidth=0)
        l, = ax.plot(times_h, bottoms + vals, color=c, linewidth=0.8, alpha=0.8)
        if dark: add_subtle_glow(l, glow_width=2, glow_alpha=0.15)
        bottoms = bottoms + vals
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Infection % (stacked)')
    ax.set_title('Graph #20: Echo Chamber Penetration (Stacked Area)'); ax.legend(fontsize=7); ax.grid(True)

plot_dual(20, 'echo_chambers', plot_g20, plotly_g20)"""))

# =====================================================================
# Cell 10: Graph #22 — Attention Budget
# =====================================================================
cells.append(code_lines(r"""# Graph #22: Attention Budget Depletion (Fix 11: add p10 percentile for hubs)
def plot_g22(fig, ax, dark=True):
    c1 = C_PRIMARY if dark else '#333333'; c2 = C_RUMOR if dark else '#cc3333'; c3 = C_SILENT if dark else '#cc8800'
    l1, = ax.plot(times_h, tl["attention_budget_mean_all"], color=c1, label='Mean (all nodes)')
    l2, = ax.plot(times_h, tl["attention_budget_mean_hubs"], color=c2, label='Mean (hub nodes)')
    p10 = tl.get("attention_budget_p10_hubs", [])
    if p10 and len(p10) == len(times_h):
        l3, = ax.plot(times_h, p10, color=c3, linestyle='--', label='p10 (hub nodes)')
        if dark: add_subtle_glow(l3, glow_width=2)
    if dark:
        add_subtle_glow(l1); add_subtle_glow(l2)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Attention Budget (0-1)')
    ax.set_title('Graph #22: Attention Budget Depletion'); ax.legend(); ax.grid(True)
    ax.set_ylim(0, 1.05)

def plotly_g22():
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=times_h, y=tl["attention_budget_mean_all"], mode='lines', name='All Nodes', line=dict(color=C_PRIMARY, width=1.5)))
    fig_p.add_trace(go.Scatter(x=times_h, y=tl["attention_budget_mean_hubs"], mode='lines', name='Hub Nodes (mean)', line=dict(color=C_RUMOR, width=1.5)))
    p10 = tl.get("attention_budget_p10_hubs", [])
    if p10 and len(p10) == len(times_h):
        fig_p.add_trace(go.Scatter(x=times_h, y=p10, mode='lines', name='Hub Nodes (p10)', line=dict(color=C_SILENT, width=1.2, dash='dash')))
    fig_p.update_layout(title='Graph #22: Attention Budget', xaxis_title='Time (hours)', yaxis_title='Budget', yaxis_range=[0, 1.05])
    return fig_p

plot_dual(22, 'attention_budget', plot_g22, plotly_g22)"""))

# =====================================================================
# Cell 11: Graph #23 — Emotional Drift (FIXED: dominant emotions)
# =====================================================================
cells.append(code_lines(r"""# Graph #23: Emotional Susceptibility Drift (dominant emotions for celebrity scenario)
# Celebrity: curiosity (0.8) + urgency (0.6) are dominant
def plot_g23(fig, ax, dark=True):
    curiosity = tl.get("curiosity_susceptibility_mean", [])
    urgency = tl.get("urgency_susceptibility_mean", [])
    fear = tl.get("fear_susceptibility_mean", tl.get("fear_mean", []))
    if curiosity and len(curiosity) == len(times_h):
        c1 = C_CORRECT if dark else '#3366cc'; c2 = C_SILENT if dark else '#cc8800'; c3 = C_RUMOR if dark else '#cc3333'
        l1, = ax.plot(times_h, curiosity, color=c1, label='Curiosity (dominant)')
        l2, = ax.plot(times_h, urgency, color=c2, label='Urgency')
        l3, = ax.plot(times_h, fear, color=c3, alpha=0.6, label='Fear')
        if dark:
            for l in [l1, l2, l3]: add_subtle_glow(l)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Mean Susceptibility')
    ax.set_title('Graph #23: Emotional Susceptibility Drift (Celebrity)'); ax.legend(); ax.grid(True)

plot_dual(23, 'emotional_drift', plot_g23)"""))

# =====================================================================
# Cell 12: Graph #24 — Bot Survival
# =====================================================================
cells.append(code_lines(r"""# Graph #24: Bot Survival Curve (Fix 4: per-platform lines)
bot_per_plat = tl.get("bot_survival_per_platform", [])

def plot_g24(fig, ax, dark=True):
    # Overall aggregate
    c_agg = C_PRIMARY if dark else '#555555'
    l_agg, = ax.plot(times_h, [x*100 for x in tl["bot_survival_fraction"]], color=c_agg, linewidth=1.8, label='Overall')
    if dark: add_subtle_glow(l_agg)
    # Per platform
    if bot_per_plat:
        all_plats = set()
        for d in bot_per_plat:
            all_plats.update(d.keys())
        for plat in sorted(all_plats, key=lambda p: p.value):
            vals = [d.get(plat, 1.0)*100 for d in bot_per_plat]
            c = PLATFORM_COLORS.get(plat, C_BOT) if dark else ACAD_PLAT_COLORS.get(plat, '#663399')
            l, = ax.plot(times_h, vals, color=c, linewidth=1.0, alpha=0.8, label=PLATFORM_NAMES.get(plat, plat.value))
            if dark: add_subtle_glow(l, glow_width=2, glow_alpha=0.15)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Surviving Bots (%)')
    ax.set_title('Graph #24: Bot Survival by Platform'); ax.legend(fontsize=7); ax.grid(True)
    ax.set_ylim(0, 105)

plot_dual(24, 'bot_survival', plot_g24)"""))

# =====================================================================
# Cell 13: Graph #25 — Rewiring Events
# =====================================================================
cells.append(code_lines(r"""# Graph #25: Rewiring Events (Fix 10: stacked area chart)
def plotly_g25():
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=times_h, y=tl["cumulative_unfollows"], mode='lines', name='Unfollows',
        stackgroup='one', line=dict(color=C_RUMOR, width=0.5)))
    fig_p.add_trace(go.Scatter(x=times_h, y=tl["cumulative_seeks"], mode='lines', name='Seeks',
        stackgroup='one', line=dict(color=C_FC, width=0.5)))
    fig_p.update_layout(title='Graph #25: Cumulative Rewiring Events (Stacked Area)', xaxis_title='Time (hours)', yaxis_title='Cumulative Count')
    return fig_p

def plot_g25(fig, ax, dark=True):
    c1 = C_RUMOR if dark else '#cc3333'; c2 = C_FC if dark else '#339966'
    unfollows = np.array(tl["cumulative_unfollows"], dtype=float)
    seeks = np.array(tl["cumulative_seeks"], dtype=float)
    ax.fill_between(times_h, 0, unfollows, alpha=0.6, color=c1, label='Unfollows', linewidth=0)
    ax.fill_between(times_h, unfollows, unfollows + seeks, alpha=0.6, color=c2, label='Seeks', linewidth=0)
    l1, = ax.plot(times_h, unfollows, color=c1, linewidth=0.8)
    l2, = ax.plot(times_h, unfollows + seeks, color=c2, linewidth=0.8)
    if dark:
        add_subtle_glow(l1, glow_width=2, glow_alpha=0.15)
        add_subtle_glow(l2, glow_width=2, glow_alpha=0.15)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Cumulative Count (stacked)')
    ax.set_title('Graph #25: Cumulative Rewiring Events (Stacked Area)'); ax.legend(); ax.grid(True)

plot_dual(25, 'rewiring_events', plot_g25, plotly_g25)"""))

# =====================================================================
# Cell 14: Graph #26 — Demographic Breakdown (with Reddit annotation)
# =====================================================================
cells.append(code_lines(r"""# Graph #26: Demographic Breakdown (infection by age_group x platform)
snapshot = result_detail.node_data_snapshot
if snapshot:
    age_plat = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for nid, data in snapshot.items():
        ag = data.get("age_group", "unknown")
        for pv, cnt in data.get("connections_count", {}).items():
            if cnt > 0:
                plat_key = pv
                infected = 1 if data["status"] in ("believing", "silent_believer") else 0
                age_plat[ag][plat_key][0] += 1
                age_plat[ag][plat_key][1] += infected

    age_groups = sorted(age_plat.keys())
    plat_keys = ["twitter", "whatsapp", "instagram", "reddit"]

    def plot_g26(fig, ax, dark=True):
        x = np.arange(len(age_groups))
        width = 0.18
        colors = list(PLATFORM_COLORS.values()) if dark else ['#2266aa','#228855','#aa2255','#aa5522']
        for i, pk in enumerate(plat_keys):
            rates = []
            for ag in age_groups:
                total, inf = age_plat[ag].get(pk, [0, 0])
                rates.append(inf/total*100 if total > 0 else 0)
            ax.bar(x + i*width, rates, width, label=pk.capitalize(), color=colors[i], alpha=0.85, edgecolor='none')
        ax.set_xticks(x + 1.5*width); ax.set_xticklabels(age_groups)
        ax.set_ylabel('Infection Rate (%)'); ax.legend(fontsize=7)
        ax.set_title('Graph #26: Infection Rate by Age Group x Platform\n(Multi-Platform Run - Reddit via platform hop, not isolated)',
                     fontsize=11 if dark else 12)
        ax.grid(True, axis='y')

    plot_dual(26, 'demographic_breakdown', plot_g26)
else:
    print("  No node_data_snapshot available for Graph #26")"""))

# =====================================================================
# Cell 15: Section 2 header
# =====================================================================
cells.append(md_lines("## Section 2: Platform Comparison"))

# =====================================================================
# Cell 16: Graph #3 — Platform Comparison
# =====================================================================
cells.append(code_lines(r"""# Graph #3: Platform Comparison (Fix 1: always show all 4 platforms, handle edge cases)
def plot_g3(fig, ax, dark=True):
    for plat in [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]:
        r = platform_results.get(plat)
        if r is None:
            continue
        c = PLATFORM_COLORS[plat] if dark else ACAD_PLAT_COLORS[plat]
        if r.infection_timeline and len(r.infection_timeline) > 1:
            ts = [t/3600 for t, _ in r.infection_timeline]; vs = [v*100 for _, v in r.infection_timeline]
            l, = ax.plot(ts, vs, color=c, label=f'{PLATFORM_NAMES[plat]} ({r.final_infection_rate:.0%})')
            if dark: add_subtle_glow(l)
        else:
            # Platform had minimal/no spread — show as flat line at final value
            ax.axhline(r.final_infection_rate * 100, color=c, linestyle=':', alpha=0.5,
                        label=f'{PLATFORM_NAMES[plat]} ({r.final_infection_rate:.0%})')
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Infection Rate (%)')
    ax.set_title('Graph #3: Platform Comparison (Isolated Single-Platform Sims)'); ax.legend(); ax.grid(True)

def plotly_g3():
    fig_p = go.Figure()
    for plat in [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]:
        r = platform_results.get(plat)
        if r is None:
            continue
        if r.infection_timeline and len(r.infection_timeline) > 1:
            ts = [t/3600 for t, _ in r.infection_timeline]; vs = [v*100 for _, v in r.infection_timeline]
            fig_p.add_trace(go.Scatter(x=ts, y=vs, mode='lines',
                name=f'{PLATFORM_NAMES[plat]} ({r.final_infection_rate:.0%})',
                line=dict(color=PLATFORM_COLORS[plat], width=1.5)))
        else:
            fig_p.add_hline(y=r.final_infection_rate * 100, line_dash='dot',
                line_color=PLATFORM_COLORS[plat],
                annotation_text=f'{PLATFORM_NAMES[plat]} ({r.final_infection_rate:.0%})')
    fig_p.update_layout(title='Graph #3: Platform Comparison', xaxis_title='Time (hours)', yaxis_title='Infection %')
    return fig_p

plot_dual(3, 'platform_comparison', plot_g3, plotly_g3)"""))

# =====================================================================
# Cell 17: Section 3 header
# =====================================================================
cells.append(md_lines("## Section 3: Network Autopsy"))

# =====================================================================
# Cell 18: Autopsy data + counterfactual run
# =====================================================================
cells.append(code_lines(r"""# Compute network autopsy and run counterfactual analysis
print("Computing network autopsy...")
autopsy = compute_network_autopsy(result_detail)
print(f"  Bridge nodes: {len(autopsy.get('bridge_nodes', []))}")
print(f"  Deadliest mutation: v{autopsy.get('deadliest_mutation', {}).get('version', '?')}")
print(f"  Mutation chain length: {len(autopsy.get('mutation_chain', []))}")

print("\nRunning counterfactual analysis (10 scenarios x 200 runs)...")
t0 = time.perf_counter()
counterfactual_results = run_counterfactual_analysis(
    baseline_result=result_detail, n_runs=200, scenario="celebrity",
    seed_platform=Platform.TWITTER, network_size=500, base_seed=42,
)
print(f"  Done in {time.perf_counter()-t0:.1f}s")
baseline_infection = counterfactual_results.get("baseline", None)
if baseline_infection:
    baseline_infection = baseline_infection.mean_infection"""))

# =====================================================================
# Cell 19: Graph #16 — Network Graph
# =====================================================================
cells.append(code_lines(r"""# Graph #16: Network Graph with Critical Path
snapshot = result_detail.node_data_snapshot
if snapshot:
    def plot_g16(fig, ax, dark=True):
        G = nx.Graph()
        for nid, data in snapshot.items():
            G.add_node(nid)
        # Add edges from infected_by relationships
        for nid, data in snapshot.items():
            if data["infected_by"] is not None and data["infected_by"] in snapshot:
                G.add_edge(data["infected_by"], nid)

        try:
            pos = nx.kamada_kawai_layout(G)
        except ImportError:
            pos = nx.spring_layout(G, seed=42)
        # Color by status
        status_colors = {'believing': C_RUMOR if dark else '#cc3333', 'silent_believer': C_SILENT if dark else '#cc8800',
                        'corrected': C_CORRECT if dark else '#3366cc', 'unaware': C_UNAWARE if dark else '#888888',
                        'immune': C_FC if dark else '#339966', 'removed': C_BOT if dark else '#663399'}
        node_colors = [status_colors.get(snapshot[n]["status"], '#444444') for n in G.nodes()]
        node_sizes = [max(5, snapshot[n]["downstream_infections"] * 3 + 2) for n in G.nodes()]

        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color='#1a1a2a' if dark else '#cccccc', width=0.3)
        # Critical path in red
        critical = autopsy.get("critical_path", [])
        if len(critical) > 1:
            path_edges = [(critical[i], critical[i+1]) for i in range(len(critical)-1) if critical[i] in G and critical[i+1] in G]
            if path_edges:
                nx.draw_networkx_edges(G, pos, edgelist=path_edges, ax=ax, edge_color=C_RUMOR if dark else '#cc3333', width=1.5, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8, linewidths=0)
        ax.set_title('Graph #16: Network Autopsy (Critical Path in Red)')
        ax.axis('off')

    plot_dual(16, 'network_autopsy', plot_g16, figsize=(10, 10))
else:
    print("  No snapshot for Graph #16")"""))

# =====================================================================
# Cell 20: Graph #17 — Counterfactual (FIXED: sorted, diverging colors)
# =====================================================================
cells.append(code_lines(r"""# Graph #17: Counterfactual Comparison (sorted by delta, diverging colors)
if baseline_infection is not None:
    deltas = {}
    for k, v in counterfactual_results.items():
        if k != "baseline" and hasattr(v, 'mean_infection'):
            deltas[k] = v.mean_infection - baseline_infection
    sorted_scenarios = sorted(deltas, key=deltas.get)

    def plot_g17(fig, ax, dark=True):
        vals = [deltas[s] * 100 for s in sorted_scenarios]
        colors = ['#66bb88' if d < 0 else '#cc6666' for d in vals] if dark else ['#228855' if d < 0 else '#cc3333' for d in vals]
        ax.barh(sorted_scenarios, vals, color=colors, alpha=0.85, edgecolor='none')
        ax.axvline(0, color='#808090' if dark else '#333333', linewidth=0.8)
        ax.set_xlabel('Change in Infection Rate (pp)')
        ax.set_title('Graph #17: Counterfactual Analysis')
        tc = '#b0b0b8' if dark else '#333333'
        for i, s in enumerate(sorted_scenarios):
            d = deltas[s] * 100
            ax.text(d + (0.3 if d >= 0 else -0.3), i, f'{d:+.1f}%', va='center',
                    ha='left' if d >= 0 else 'right', fontsize=8, color=tc)
        ax.grid(True, axis='x')

    def plotly_g17():
        fig_p = go.Figure()
        vals = [deltas[s] * 100 for s in sorted_scenarios]
        colors = ['#66bb88' if d < 0 else '#cc6666' for d in vals]
        fig_p.add_trace(go.Bar(y=sorted_scenarios, x=vals, orientation='h',
            marker_color=colors, text=[f'{d:+.1f}%' for d in vals], textposition='outside'))
        fig_p.update_layout(title='Graph #17: Counterfactual Analysis', xaxis_title='Change (pp)')
        return fig_p

    plot_dual(17, 'counterfactual', plot_g17, plotly_g17)"""))

# =====================================================================
# Cell 21: Graph #18 — Mutation Chain (FIXED: connected scatter)
# =====================================================================
cells.append(code_lines(r"""# Graph #18: Mutation Chain — Emotional Profile Evolution (connected scatter)
# Run with elevated mutation probability for demonstration
cfg_mut = SimulationConfig(
    scenario="celebrity", seed_platform=Platform.TWITTER,
    active_platforms=[Platform.TWITTER], network_size=500,
    master_seed=42, detailed_tracking=True, mutation_probability=0.15,
)
eng_mut = SimulationEngine(cfg_mut)
eng_mut.setup()
res_mut = eng_mut.run()
# Extract mutation chain from engine's rumor_versions (not autopsy)
# Build mutation data: list of dicts with version, chain, emotions
mutations = []
for vid in sorted(eng_mut.rumor_versions.keys()):
    rv = eng_mut.rumor_versions[vid]
    emo = rv.emotions.__dict__ if hasattr(rv.emotions, '__dict__') else {}
    mutations.append({"version": vid, "chain": rv.mutation_chain, "emotions": emo})
print(f"Mutation chain: {len(mutations)} versions (mutation_prob=0.15)")

# Find the longest chain for connected scatter
chains_by_depth = {}
for m in mutations:
    d = len(m["chain"])
    chains_by_depth.setdefault(d, []).append(m)
# Build the main lineage: follow v0 -> deepest chain
main_chain = [m for m in mutations if m["chain"] == [0]]  # root
if main_chain:
    root = main_chain[0]
    # Find longest chain starting from root
    longest = max(mutations, key=lambda m: len(m["chain"]))
    lineage_versions = set(longest["chain"])
    lineage = [m for m in mutations if m["version"] in lineage_versions]
    lineage.sort(key=lambda m: m["chain"].index(m["version"]) if m["version"] in m["chain"] else 999)
else:
    lineage = mutations[:10]  # fallback

def plot_g18(fig, ax, dark=True):
    if len(mutations) > 1:
        # Plot ALL versions as background scatter
        all_v = [m["version"] for m in mutations]
        emotions_map = {"fear": C_RUMOR if dark else '#cc3333', "curiosity": C_CORRECT if dark else '#3366cc',
                    "urgency": C_SILENT if dark else '#cc8800', "outrage": C_BOT if dark else '#9944cc',
                    "humor": C_FC if dark else '#339966'}
        for emo, color in emotions_map.items():
            vals = [m["emotions"].get(emo, 0) for m in mutations]
            ax.scatter(all_v, vals, color=color, alpha=0.25, s=15, zorder=1)
            # Connected scatter for main lineage
            lin_v = [m["version"] for m in lineage]
            lin_vals = [m["emotions"].get(emo, 0) for m in lineage]
            if len(lin_v) > 1:
                l, = ax.plot(lin_v, lin_vals, 'o-', color=color, label=emo.capitalize(),
                             markersize=5, linewidth=1.2, zorder=2)
                if dark: add_subtle_glow(l)
        ax.set_xlabel('Mutation Version'); ax.set_ylabel('Emotion Intensity')
        ax.set_title('Graph #18: Mutation Chain - Emotional Profile Evolution\n(mutation_probability=0.15)')
        ax.legend(loc='upper right', ncol=2); ax.grid(True)
    else:
        ax.text(0.5, 0.5, 'Insufficient mutations\n(only 1 version)', ha='center', va='center',
                transform=ax.transAxes, fontsize=12, color='#808080')
        ax.set_title('Graph #18: Mutation Chain')

plot_dual(18, 'mutation_chain', plot_g18)"""))

# =====================================================================
# Cell 22: Section 4 header
# =====================================================================
cells.append(md_lines("## Section 4: Monte Carlo Analysis"))

# =====================================================================
# Cell 23: Run MC batch
# =====================================================================
cells.append(code_lines(r"""# Run 1000-run Monte Carlo batch
print("Running Monte Carlo batch (1000 runs, 500 nodes, celebrity/twitter)...")
t0 = time.perf_counter()
mc_result = run_monte_carlo(n_runs=1000, network_size=500, scenario='celebrity',
    seed_platform=Platform.TWITTER, base_seed=42)
elapsed = time.perf_counter() - t0
print(f"  Done in {elapsed:.1f}s ({elapsed/1000*1000:.0f}ms/run)")
print(f"  Mean infection: {mc_result.mean_infection:.1%} +/- {mc_result.ci_95_upper - mc_result.mean_infection:.1%}")
print(f"  Mean R0 peak: {mc_result.mean_r0:.2f}")
print(f"  Death types: {mc_result.death_type_counts}")"""))

# =====================================================================
# Cell 24: Graph #2 — Infection Histogram
# =====================================================================
cells.append(code_lines(r"""# Graph #2: Infection Rate Histogram
def plot_g2(fig, ax, dark=True):
    c = C_PRIMARY if dark else '#555555'
    ax.hist(mc_result.infection_rates * 100, bins=30, color=c, alpha=0.7, edgecolor='#2a2a3a' if dark else '#999999', linewidth=0.5)
    ax.axvline(mc_result.mean_infection * 100, color=C_RUMOR if dark else '#cc3333', linestyle='--', label=f'Mean={mc_result.mean_infection:.1%}')
    ax.set_xlabel('Final Infection Rate (%)'); ax.set_ylabel('Count')
    ax.set_title('Graph #2: MC Infection Rate Distribution (N=1000)'); ax.legend(); ax.grid(True, axis='y')

def plotly_g2():
    fig_p = go.Figure()
    fig_p.add_trace(go.Histogram(x=mc_result.infection_rates * 100, nbinsx=30, marker_color=C_PRIMARY, opacity=0.7))
    fig_p.add_vline(x=mc_result.mean_infection * 100, line_dash='dash', line_color=C_RUMOR, annotation_text=f'Mean={mc_result.mean_infection:.1%}')
    fig_p.update_layout(title='Graph #2: Infection Rate Distribution', xaxis_title='Infection %', yaxis_title='Count')
    return fig_p

plot_dual(2, 'infection_histogram', plot_g2, plotly_g2)"""))

# =====================================================================
# Cell 25: Graph #4 — Convergence Plot
# =====================================================================
cells.append(code_lines(r"""# Graph #4: Convergence Plot (running mean + 95% CI)
rm = mc_result.running_means
if rm:
    ns = [x[0] for x in rm]; means = [x[1]*100 for x in rm]
    ci_lo = [x[2]*100 for x in rm]; ci_hi = [x[3]*100 for x in rm]

    def plot_g4(fig, ax, dark=True):
        c = C_PRIMARY if dark else '#333333'; cb = C_CORRECT if dark else '#3366cc'
        l, = ax.plot(ns, means, color=c, label='Running Mean')
        ax.fill_between(ns, ci_lo, ci_hi, alpha=0.2, color=cb, label='95% CI')
        if dark: add_subtle_glow(l)
        ax.set_xlabel('Number of Runs'); ax.set_ylabel('Mean Infection Rate (%)')
        ax.set_title('Graph #4: MC Convergence'); ax.legend(); ax.grid(True)

    def plotly_g4():
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=ns, y=means, mode='lines', name='Running Mean', line=dict(color=C_PRIMARY, width=1.5)))
        fig_p.add_trace(go.Scatter(x=ns+ns[::-1], y=ci_hi+ci_lo[::-1], fill='toself', name='95% CI', fillcolor='rgba(102,153,204,0.2)', line=dict(width=0)))
        fig_p.update_layout(title='Graph #4: MC Convergence', xaxis_title='Runs', yaxis_title='Mean Infection %')
        return fig_p

    plot_dual(4, 'convergence', plot_g4, plotly_g4)"""))

# =====================================================================
# Cell 26: Graph #7 — CDF
# =====================================================================
cells.append(code_lines(r"""# Graph #7: CDF of Infection Rates
sorted_rates = np.sort(mc_result.infection_rates) * 100
cdf = np.arange(1, len(sorted_rates)+1) / len(sorted_rates)

def plot_g7(fig, ax, dark=True):
    c = C_PRIMARY if dark else '#333333'
    l, = ax.plot(sorted_rates, cdf, color=c)
    if dark: add_subtle_glow(l)
    ax.set_xlabel('Infection Rate (%)'); ax.set_ylabel('Cumulative Probability')
    ax.set_title('Graph #7: CDF of Infection Rates'); ax.grid(True)

def plotly_g7():
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=sorted_rates.tolist(), y=cdf.tolist(), mode='lines', name='CDF', line=dict(color=C_PRIMARY, width=1.5)))
    fig_p.update_layout(title='Graph #7: CDF of Infection Rates', xaxis_title='Infection %', yaxis_title='Cumulative Probability')
    return fig_p

plot_dual(7, 'cdf', plot_g7, plotly_g7)"""))

# =====================================================================
# Cell 27: Graph #10 — Tipping Point
# =====================================================================
cells.append(code_lines(r"""# Graph #10: Tipping Point Distribution + Point of No Return
tipping_points = np.array([compute_tipping_point(r.infection_timeline) for r in mc_result.results if r.infection_timeline])
ponr = compute_point_of_no_return(mc_result)
print(f"Point of no return: {ponr:.1%}")

def plot_g10(fig, ax, dark=True):
    tp_pct = tipping_points[tipping_points > 0] / 3600  # convert to hours
    if len(tp_pct) > 0:
        c = C_PRIMARY if dark else '#555555'
        ax.hist(tp_pct, bins=20, color=c, alpha=0.7, edgecolor='#2a2a3a' if dark else '#999999')
    if ponr > 0:
        ax.axvline(ponr * 100, color=C_RUMOR if dark else '#cc3333', linestyle='--', linewidth=1.5, label=f'Point of No Return: {ponr:.0%}')
    ax.set_xlabel('Tipping Point (hours)'); ax.set_ylabel('Count')
    ax.set_title('Graph #10: Tipping Point Distribution'); ax.legend(); ax.grid(True, axis='y')

def plotly_g10():
    fig_p = go.Figure()
    tp_pct = tipping_points[tipping_points > 0] / 3600
    if len(tp_pct) > 0:
        fig_p.add_trace(go.Histogram(x=tp_pct.tolist(), nbinsx=20, marker_color=C_PRIMARY, opacity=0.7))
    fig_p.update_layout(title='Graph #10: Tipping Point Distribution', xaxis_title='Hours', yaxis_title='Count')
    return fig_p

plot_dual(10, 'tipping_point', plot_g10, plotly_g10)"""))

# =====================================================================
# Cell 28: Graph #14 — Death Type Pie
# =====================================================================
cells.append(code_lines(r"""# Graph #14: Death Type Pie Chart
dt_counts = mc_result.death_type_counts
labels = list(dt_counts.keys()); sizes = list(dt_counts.values())
pie_colors_dark = [C_RUMOR, C_CORRECT, C_SILENT, C_FC, C_BOT, C_UNAWARE][:len(labels)]
pie_colors_acad = ['#cc3333', '#3366cc', '#cc8800', '#339966', '#663399', '#888888'][:len(labels)]

def plot_g14(fig, ax, dark=True):
    colors = pie_colors_dark if dark else pie_colors_acad
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
        wedgeprops=dict(edgecolor='#d0d0d8' if dark else '#ffffff', linewidth=0.5),
        textprops=dict(color='#b0b0b8' if dark else '#222222', fontsize=9))
    ax.set_title('Graph #14: Termination Type Distribution')

plot_dual(14, 'death_types', plot_g14)"""))

# =====================================================================
# Cell 29: Graph #15 — Kaplan-Meier
# =====================================================================
cells.append(code_lines(r"""# Graph #15: Kaplan-Meier Survival Curve
term_times = np.sort(mc_result.termination_times) / 3600
survival = 1.0 - np.arange(1, len(term_times)+1) / len(term_times)

def plot_g15(fig, ax, dark=True):
    c = C_PRIMARY if dark else '#333333'
    l, = ax.step(term_times, survival, color=c, where='post')
    if dark: add_subtle_glow(l)
    ax.set_xlabel('Time (hours)'); ax.set_ylabel('Survival Probability')
    ax.set_title('Graph #15: Kaplan-Meier Survival Curve'); ax.grid(True)

def plotly_g15():
    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(x=term_times.tolist(), y=survival.tolist(), mode='lines', name='Survival', line=dict(color=C_PRIMARY, width=1.5, shape='hv')))
    fig_p.update_layout(title='Graph #15: Kaplan-Meier Survival', xaxis_title='Hours', yaxis_title='Survival Prob')
    return fig_p

plot_dual(15, 'kaplan_meier', plot_g15, plotly_g15)"""))

# =====================================================================
# Cell 30: Graph #27 — Termination Time
# =====================================================================
cells.append(code_lines(r"""# Graph #27: Termination Time Distribution
def plot_g27(fig, ax, dark=True):
    c = C_PRIMARY if dark else '#555555'
    ax.hist(mc_result.termination_times / 3600, bins=25, color=c, alpha=0.7, edgecolor='#2a2a3a' if dark else '#999999')
    mean_t = np.mean(mc_result.termination_times) / 3600
    ax.axvline(mean_t, color=C_RUMOR if dark else '#cc3333', linestyle='--', label=f'Mean={mean_t:.1f}h')
    ax.set_xlabel('Termination Time (hours)'); ax.set_ylabel('Count')
    ax.set_title('Graph #27: Termination Time Distribution'); ax.legend(); ax.grid(True, axis='y')

def plotly_g27():
    fig_p = go.Figure()
    fig_p.add_trace(go.Histogram(x=(mc_result.termination_times/3600).tolist(), nbinsx=25, marker_color=C_PRIMARY, opacity=0.7))
    fig_p.update_layout(title='Graph #27: Termination Time Distribution', xaxis_title='Hours', yaxis_title='Count')
    return fig_p

plot_dual(27, 'termination_time', plot_g27, plotly_g27)"""))

# =====================================================================
# Cell 31: Section 5 header
# =====================================================================
cells.append(md_lines("## Section 5: Platform Resilience"))

# =====================================================================
# Cell 32: Graph #11 — Platform Resilience
# =====================================================================
cells.append(code_lines(r"""# Graph #11: Platform Resilience (Fix 5: show resilience score = 1 - infection, per spec §6.3)
print("Running per-platform MC (4 x 200 runs)...")
plat_mc = {}
for plat in [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]:
    mc = run_monte_carlo(n_runs=200, network_size=500, scenario='celebrity',
        seed_platform=plat, active_platforms=[plat], base_seed=42)
    plat_mc[plat] = mc
    resilience = (1.0 - mc.mean_infection) * 100
    print(f"  {plat.value}: infection={mc.mean_infection:.1%}, resilience={resilience:.1f}")

# Sort by resilience (highest first)
plats_sorted = sorted(plat_mc.keys(), key=lambda p: plat_mc[p].mean_infection)

def plot_g11(fig, ax, dark=True):
    names = [PLATFORM_NAMES[p] for p in plats_sorted]
    vals = [(1.0 - plat_mc[p].mean_infection) * 100 for p in plats_sorted]
    colors = [PLATFORM_COLORS[p] if dark else ACAD_PLAT_COLORS[p] for p in plats_sorted]
    ax.barh(names, vals, color=colors, alpha=0.85, edgecolor='none')
    tc = '#b0b0b8' if dark else '#333333'
    for i, v in enumerate(vals):
        ax.text(v + 0.5, i, f'{v:.1f}', va='center', fontsize=9, color=tc)
    ax.set_xlabel('Resilience Score (0-100)'); ax.set_title('Graph #11: Platform Resilience Score')
    ax.set_xlim(0, 100); ax.grid(True, axis='x')

def plotly_g11():
    fig_p = go.Figure()
    names = [PLATFORM_NAMES[p] for p in plats_sorted]
    vals = [(1.0 - plat_mc[p].mean_infection) * 100 for p in plats_sorted]
    colors = [PLATFORM_COLORS[p] for p in plats_sorted]
    fig_p.add_trace(go.Bar(y=names, x=vals, orientation='h', marker_color=colors,
        text=[f'{v:.1f}' for v in vals], textposition='outside'))
    fig_p.update_layout(title='Graph #11: Platform Resilience Score (higher = more resistant)',
        xaxis_title='Resilience Score (0-100)', xaxis_range=[0, 100])
    return fig_p

plot_dual(11, 'platform_resilience', plot_g11, plotly_g11)"""))

# =====================================================================
# Cell 33: Section 6 header
# =====================================================================
cells.append(md_lines("## Section 6: Herd Immunity Analysis"))

# =====================================================================
# Cell 34: Graph #12 — Herd Immunity Lines
# =====================================================================
cells.append(code_lines(r"""# Graph #12: Herd Immunity (Fix 3: more runs for smoother curves, finer literacy levels)
print("Running herd immunity sweep (4 strategies x 11 pcts x 150 runs)...")
t0 = time.perf_counter()
herd_results = run_herd_immunity_sweep(
    strategies=['random', 'bridge', 'influencer', 'echo_seed'],
    literacy_pcts=[0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
    n_runs_per_cell=150, scenario='celebrity', seed_platform=Platform.TWITTER,
    network_size=500, base_seed=42,
)
print(f"  Done in {time.perf_counter()-t0:.1f}s")

strategy_colors_dark = {'random': C_UNAWARE, 'bridge': C_CORRECT, 'influencer': C_RUMOR, 'echo_seed': C_FC}
strategy_colors_acad = {'random': '#888888', 'bridge': '#3366cc', 'influencer': '#cc3333', 'echo_seed': '#339966'}

def plot_g12(fig, ax, dark=True):
    for strat, pct_dict in herd_results.items():
        pcts = sorted(pct_dict.keys())
        means = [pct_dict[p].mean_infection * 100 for p in pcts]
        c = strategy_colors_dark[strat] if dark else strategy_colors_acad[strat]
        l, = ax.plot([p*100 for p in pcts], means, 'o-', color=c, label=strat.replace('_', ' ').title(), markersize=4)
        if dark: add_subtle_glow(l)
    ax.set_xlabel('Literacy Placement (%)'); ax.set_ylabel('Mean Infection Rate (%)')
    ax.set_title('Graph #12: Herd Immunity Analysis'); ax.legend(); ax.grid(True)

def plotly_g12():
    fig_p = go.Figure()
    for strat, pct_dict in herd_results.items():
        pcts = sorted(pct_dict.keys())
        means = [pct_dict[p].mean_infection * 100 for p in pcts]
        fig_p.add_trace(go.Scatter(x=[p*100 for p in pcts], y=means, mode='lines+markers', name=strat.replace('_', ' ').title(),
            line=dict(color=strategy_colors_dark[strat], width=1.5), marker=dict(size=5)))
    fig_p.update_layout(title='Graph #12: Herd Immunity', xaxis_title='Literacy %', yaxis_title='Infection %')
    return fig_p

plot_dual(12, 'herd_immunity', plot_g12, plotly_g12)"""))

# =====================================================================
# Cell 35: Graph #13 — Cross-Topic Heatmap (FIXED: multi-topic)
# =====================================================================
cells.append(code_lines(r"""# Graph #13: Cross-Topic x Cross-Platform Heatmap
# Sweep 4 topics x 4 platforms x 5 literacy pcts (30 runs each)
print("Running cross-topic heatmap sweep (2400 total runs)...")
topics = ["celebrity", "financial", "health", "campus"]
platforms_list = [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]
literacy_pcts = [0.0, 0.10, 0.20, 0.30, 0.50]

heatmap_data = {}
for topic in topics:
    for plat in platforms_list:
        for pct in literacy_pcts:
            mc = run_monte_carlo(
                n_runs=30, scenario=topic, seed_platform=plat,
                active_platforms=[plat], network_size=500, base_seed=42,
                config_overrides={
                    "literacy_placement_strategy": "bridge",
                    "literacy_placement_pct": pct,
                    "literacy_placement_topic": topic,
                },
            )
            heatmap_data[(topic, plat.value, pct)] = mc.mean_infection
print("  Done")

def plot_g13(fig, ax, dark=True):
    fig.set_size_inches(14, 10)
    ax.remove()
    for idx, plat in enumerate(platforms_list):
        ax_i = fig.add_subplot(2, 2, idx+1)
        grid = np.zeros((len(topics), len(literacy_pcts)))
        for ti, topic in enumerate(topics):
            for pi, pct in enumerate(literacy_pcts):
                grid[ti, pi] = heatmap_data.get((topic, plat.value, pct), 0) * 100
        cmap = 'inferno' if dark else 'YlOrRd'
        im = ax_i.imshow(grid, cmap=cmap, aspect='auto', vmin=0, vmax=80)
        ax_i.set_xticks(range(len(literacy_pcts)))
        ax_i.set_xticklabels([f'{p:.0%}' for p in literacy_pcts])
        ax_i.set_yticks(range(len(topics)))
        ax_i.set_yticklabels(topics)
        ax_i.set_title(PLATFORM_NAMES[plat], fontsize=11)
        # Annotations
        for ti in range(len(topics)):
            for pi in range(len(literacy_pcts)):
                val = grid[ti, pi]
                tc = '#d0d0d0' if (dark and val < 50) else '#222222'
                ax_i.text(pi, ti, f'{val:.0f}', ha='center', va='center', fontsize=7, color=tc)
    fig.suptitle('Graph #13: Cross-Topic Herd Immunity Heatmap (Bridge Strategy)', fontsize=13,
                 color='#b0b0b8' if dark else '#222222')
    fig.tight_layout(rect=[0, 0, 1, 0.95])

plot_dual(13, 'herd_immunity_heatmap', plot_g13, figsize=(14, 10))"""))

# =====================================================================
# Cell 36: Section 7 header
# =====================================================================
cells.append(md_lines("## Section 7: Sensitivity Analysis"))

# =====================================================================
# Cell 37: Graph #6 — 2D Heatmap
# =====================================================================
cells.append(code_lines(r"""# Graph #6: 2D Sensitivity Heatmap (Fix 6: shorter delays that fit sim duration)
print("Running 2D sensitivity sweep...")
delays = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]  # hours (within sim duration)
sharing_mods = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
grid_6 = np.zeros((len(delays), len(sharing_mods)))

for di, delay in enumerate(delays):
    for si, sm in enumerate(sharing_mods):
        mc = run_monte_carlo(
            n_runs=100, scenario='celebrity', seed_platform=Platform.TWITTER,
            network_size=500, base_seed=42,
            config_overrides={
                "correction_delay_override": delay * 3600,
                "sharing_probability_modifier": sm,
            },
        )
        grid_6[di, si] = mc.mean_infection * 100
print("  Done")

def plot_g6(fig, ax, dark=True):
    cmap = 'inferno' if dark else 'YlOrRd'
    im = ax.imshow(grid_6, cmap=cmap, aspect='auto', origin='lower')
    ax.set_xticks(range(len(sharing_mods))); ax.set_xticklabels([f'{s:.1f}x' for s in sharing_mods])
    ax.set_yticks(range(len(delays))); ax.set_yticklabels([f'{d}h' for d in delays])
    ax.set_xlabel('Sharing Probability Modifier'); ax.set_ylabel('Correction Delay')
    ax.set_title('Graph #6: Sensitivity Heatmap')
    plt.colorbar(im, ax=ax, label='Mean Infection %')
    for di in range(len(delays)):
        for si in range(len(sharing_mods)):
            val = grid_6[di, si]
            tc = '#d0d0d0' if (dark and val < 50) else '#222222'
            ax.text(si, di, f'{val:.0f}', ha='center', va='center', fontsize=7, color=tc)

plot_dual(6, 'sensitivity_heatmap', plot_g6, figsize=(10, 7))"""))

# =====================================================================
# Cell 38: Graph #21 — Tornado Chart
# =====================================================================
cells.append(code_lines(r"""# Graph #21: Tornado Chart (parameter impact ranking)
print("Running toggle sensitivity sweeps...")
toggle_params = {
    'No Bot Detection': {"bot_detection_enabled": False},
    'No Rewiring': {"rewiring_enabled": False},
    'No Corrections': {"correction_enabled": False},
    'No Attention Budget': {"attention_budget_toggle": False},
    'No Algo Amplification': {"algorithmic_amplification_multiplier": 0.0},
    'No Framing Bonus': {"framing_bonus_enabled": False},
    'Single Platform': {"active_platforms": [Platform.TWITTER]},
}

toggle_results = {}
baseline_mc = run_monte_carlo(n_runs=200, scenario='celebrity', seed_platform=Platform.TWITTER,
    network_size=500, base_seed=42)
baseline_inf = baseline_mc.mean_infection

for name, overrides in toggle_params.items():
    mc = run_monte_carlo(n_runs=200, scenario='celebrity', seed_platform=Platform.TWITTER,
        network_size=500, base_seed=42, config_overrides=overrides)
    toggle_results[name] = mc.mean_infection - baseline_inf
    print(f"  {name}: delta={toggle_results[name]:+.1%}")

sorted_toggles = sorted(toggle_results, key=lambda k: abs(toggle_results[k]), reverse=True)

def plot_g21(fig, ax, dark=True):
    vals = [toggle_results[k] * 100 for k in sorted_toggles]
    colors = ['#cc6666' if v > 0 else '#66bb88' for v in vals] if dark else ['#cc3333' if v > 0 else '#228855' for v in vals]
    ax.barh(sorted_toggles, vals, color=colors, alpha=0.85, edgecolor='none')
    ax.axvline(0, color='#808090' if dark else '#333333', linewidth=0.8)
    tc = '#b0b0b8' if dark else '#333333'
    for i, v in enumerate(vals):
        ax.text(v + (0.3 if v >= 0 else -0.3), i, f'{v:+.1f}pp', va='center',
                ha='left' if v >= 0 else 'right', fontsize=8, color=tc)
    ax.set_xlabel('Change in Infection Rate (pp)')
    ax.set_title('Graph #21: Parameter Impact (Tornado Chart)'); ax.grid(True, axis='x')

plot_dual(21, 'tornado_chart', plot_g21)"""))

# =====================================================================
# Cell 39: Graph #28 — Framing Impact
# =====================================================================
cells.append(code_lines(r"""# Graph #28: Framing Modifier Impact (with vs without, per platform)
print("Running framing comparison...")
framing_data = {}
for plat in [Platform.TWITTER, Platform.WHATSAPP, Platform.INSTAGRAM, Platform.REDDIT]:
    mc_with = run_monte_carlo(n_runs=200, scenario='celebrity', seed_platform=plat,
        active_platforms=[plat], network_size=500, base_seed=42)
    mc_without = run_monte_carlo(n_runs=200, scenario='celebrity', seed_platform=plat,
        active_platforms=[plat], network_size=500, base_seed=42,
        config_overrides={"framing_bonus_enabled": False})
    framing_data[plat] = (mc_with.mean_infection, mc_without.mean_infection)
    print(f"  {plat.value}: with={mc_with.mean_infection:.1%}, without={mc_without.mean_infection:.1%}")

def plot_g28(fig, ax, dark=True):
    plats = list(framing_data.keys())
    x = np.arange(len(plats))
    with_vals = [framing_data[p][0]*100 for p in plats]
    without_vals = [framing_data[p][1]*100 for p in plats]
    c1 = C_RUMOR if dark else '#cc3333'; c2 = C_CORRECT if dark else '#3366cc'
    ax.bar(x - 0.15, with_vals, 0.3, label='With Framing', color=c1, alpha=0.85, edgecolor='none')
    ax.bar(x + 0.15, without_vals, 0.3, label='Without Framing', color=c2, alpha=0.85, edgecolor='none')
    ax.set_xticks(x); ax.set_xticklabels([PLATFORM_NAMES[p] for p in plats])
    ax.set_ylabel('Mean Infection Rate (%)'); ax.set_title('Graph #28: Framing Modifier Impact')
    ax.legend(); ax.grid(True, axis='y')

plot_dual(28, 'framing_impact', plot_g28)"""))

# =====================================================================
# Cell 40: Section 8 header
# =====================================================================
cells.append(md_lines("## Section 8: Distribution Fitting Documentation"))

# =====================================================================
# Cell 41: Distribution table
# =====================================================================
cells.append(code_lines(r"""# Distribution Fitting Documentation (Spec Section 6.6)
distributions = [
    ("Inter-arrival time", "Exponential", "lambda=15/min (Twitter)", "Poisson process for message arrivals"),
    ("Service time", "Exponential", "mu=30s base (platform-modified)", "Memoryless processing time"),
    ("Correction delay", "Exponential", "mu=2h", "FC response time follows heavy-tail"),
    ("Algorithmic boost duration", "Exponential", "mu=30min", "Platform amplification window"),
    ("Crisis duration", "Exponential", "mu=4h", "External shock duration"),
    ("Crisis intensity", "Uniform", "[0.3, 0.8]", "Random severity"),
    ("Credibility threshold", "Uniform/Fixed", "[0.4,0.8] regular, 0.01 bot", "Per agent type"),
    ("Emotional susceptibility", "Beta(2,5)", "mean ~0.29, right-skewed", "Most people moderately susceptible"),
    ("Digital nativity", "Beta(age-dependent)", "Young: Beta(7,3), Older: Beta(3,7)", "Age-correlated tech comfort"),
    ("Worldview dimensions", "Uniform[-1,1]^4", "4D hypercube", "Political, health, tech, authority trust"),
]
df = pd.DataFrame(distributions, columns=["Variable", "Distribution", "Parameters", "Justification"])
print(df.to_string(index=False))"""))

# =====================================================================
# Cell 42: Section 9 header
# =====================================================================
cells.append(md_lines("## Section 9: Checkpoint Cross-Run Comparison"))

# =====================================================================
# Cell 43: Checkpoint table
# =====================================================================
cells.append(code_lines(r"""# Checkpoint Cross-Run Comparison (Spec Section 1.5)
checkpoint_data = []
for cp_time in CHECKPOINT_TIMES:
    active = 0; inf_rates = []; r0_vals = []
    for r in mc_result.results:
        matching = [c for c in r.checkpoints if abs(c.time - cp_time) < 30]
        if matching:
            active += 1
            cp = matching[0]
            inf_rates.append(cp.infection_rate)
            r0_vals.append(cp.r0_estimate)
    if active > 10:
        checkpoint_data.append({
            "Time": f"{cp_time/3600:.1f}h",
            "Runs Active": active,
            "Mean Infection": f"{np.mean(inf_rates):.1%}",
            "95% CI": f"[{np.percentile(inf_rates, 2.5):.1%}, {np.percentile(inf_rates, 97.5):.1%}]",
            "Mean R0": f"{np.mean(r0_vals):.2f}",
        })

if checkpoint_data:
    df_cp = pd.DataFrame(checkpoint_data)
    print(df_cp.to_string(index=False))
else:
    print("No checkpoint data available (runs may terminate before first checkpoint)")"""))

# =====================================================================
# Cell 44: Section 10 header
# =====================================================================
cells.append(md_lines("## Section 10: Summary of Findings"))

# =====================================================================
# Cell 45: Summary
# =====================================================================
cells.append(code_lines(r"""# Summary of Key Findings
print("=" * 60)
print("KEY FINDINGS")
print("=" * 60)

print(f"\n1. Mean infection rate: {mc_result.mean_infection:.1%} +/- {mc_result.ci_95_upper - mc_result.mean_infection:.1%}")
print(f"   (1000 MC runs, 500 nodes, celebrity scenario, Twitter seed)")

print(f"\n2. Mean peak R0: {mc_result.mean_r0:.2f}")

print(f"\n3. Platform vulnerability ranking:")
for plat in sorted(plat_mc, key=lambda p: -plat_mc[p].mean_infection):
    print(f"   {PLATFORM_NAMES[plat]}: {plat_mc[plat].mean_infection:.1%}")

print(f"\n4. Death type distribution:")
for dt, count in sorted(mc_result.death_type_counts.items(), key=lambda x: -x[1]):
    print(f"   {dt}: {count} ({count/len(mc_result.results):.0%})")

print(f"\n5. Point of no return: {ponr:.1%}")

print(f"\n6. Most impactful parameters (tornado chart):")
for k in sorted_toggles[:3]:
    print(f"   {k}: {toggle_results[k]:+.1%}")

if herd_results:
    bridge_50 = herd_results.get('bridge', {}).get(0.50, None)
    random_50 = herd_results.get('random', {}).get(0.50, None)
    if bridge_50 and random_50:
        print(f"\n7. Herd immunity (50% literacy):")
        print(f"   Bridge strategy: {bridge_50.mean_infection:.1%}")
        print(f"   Random strategy: {random_50.mean_infection:.1%}")
        print(f"   Advantage: {(random_50.mean_infection - bridge_50.mean_infection):.1%} reduction")

print(f"\n8. Output saved to: {GRAPHS_DIR}/")
print(f"   visual/: {len([f for f in os.listdir(VISUAL_DIR) if f.endswith('.png')])} dark PNGs")
print(f"   academic/: {len([f for f in os.listdir(ACADEMIC_DIR) if f.endswith('.png')])} white PNGs")
print(f"   interactive/: {len([f for f in os.listdir(INTERACTIVE_DIR) if f.endswith('.html')])} Plotly HTMLs")"""))

# =====================================================================
# Build notebook JSON
# =====================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.13.0"
        }
    },
    "cells": cells
}

with open(r"C:\Users\xplod\Videos\FSWD\Afwah\simulation.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written with {len(cells)} cells")
